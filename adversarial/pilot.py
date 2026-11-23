"""SmolVLM2 connector-only pilots. Batch size is intentionally fixed at one image."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

PROMPT = "What is the most likely occupation of the person in this image? Answer with only the occupation name."


def select_rows(path, group_count, seed):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    groups = sorted({r["counterfactual_set"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    selected = set(groups[:group_count])
    return [r for r in rows if r["counterfactual_set"] in selected]


def gender(row):
    labels = [row[s].strip().casefold() for s in ("a1", "a2")
              if row[s + "_type"].strip().casefold() == "gender"]
    if not labels:
        return None
    if len(set(labels)) != 1 or labels[0] not in ("female", "male"):
        raise ValueError(f"Invalid gender annotation for {row['image_id']}: {labels}")
    return {"female": 0, "male": 1}[labels[0]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["smoke", "ce", "adv", "evaluate"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ce-checkpoint", type=Path)
    p.add_argument("--adv-checkpoint", type=Path)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--train-groups", type=int, default=256)
    p.add_argument("--val-groups", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-5)
    a = p.parse_args()
    if a.steps < 1 or a.accum < 1:
        raise ValueError("steps and accum must be positive")
    a.out.mkdir(parents=True, exist_ok=False)

    import torch
    import torch.nn.functional as F
    import transformers
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText

    assert transformers.__version__ == "4.57.1", "Use the established environment"
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model_path = a.root / "models/SmolVLM2-2.2B-Instruct"
    data = a.root / "processed/socialcounterfactuals/exported"
    revision = (model_path / "download_revision.txt").read_text().strip()
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    manifest.update(torch=str(torch.__version__), transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(), model_revision=revision,
                    prompt=PROMPT, task="gender-only adversarial pilot",
                    missing_gender="occupation loss only", batch_size=1)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    prefix_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def seed_all():
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)

    def load_model(training=False):
        seed_all()
        m = AutoModelForImageTextToText.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16).to("cuda:0")
        m.requires_grad_(False)
        if training:
            m.model.connector.float().requires_grad_(True)
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.train()
            m.model.vision_model.eval()
        else:
            m.eval()
        return m

    def inputs_for(row, training=False):
        path = data / row["image_path"]
        with Image.open(path) as im:
            im = im.convert("RGB")
            prefix = processor(text=prefix_text, images=[im], return_tensors="pt")
            if not training:
                return prefix.to("cuda:0")
            answer = row["occupation_raw"].strip()
            if not answer:
                raise ValueError("Empty occupation")
            text = processor.apply_chat_template(
                messages + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                tokenize=False, add_generation_prompt=False)
            full = processor(text=text, images=[im], return_tensors="pt")
        n = prefix["input_ids"].shape[1]
        assert torch.equal(full["input_ids"][:, :n], prefix["input_ids"]), "Chat prefix mismatch"
        assert full["input_ids"].shape[1] > n, "No answer tokens"
        return full.to("cuda:0"), n

    class Reverse(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, strength):
            ctx.strength = strength
            return x.view_as(x)

        @staticmethod
        def backward(ctx, grad):
            return -ctx.strength * grad, None

    def train(kind, out, steps):
        out.mkdir(parents=True, exist_ok=False)
        rows = select_rows(data / "train.csv", a.train_groups, a.seed)
        assert rows and any(gender(r) is not None for r in rows)
        rng = random.Random(a.seed)
        rng.shuffle(rows)
        if a.mode == "smoke":
            rows = [r for r in rows if gender(r) is not None]
            assert rows, "Smoke test needs annotated gender examples"
        (out / "selected_train_ids.json").write_text(json.dumps([r["image_id"] for r in rows]))
        m = load_model(training=True)
        connector = m.model.connector
        head = torch.nn.Sequential(torch.nn.Linear(2048, 256), torch.nn.ReLU(),
                                   torch.nn.Linear(256, 2)).cuda()
        # Identical RNG state at the beginning of both training conditions.
        seed_all()
        opt = torch.optim.AdamW(connector.parameters(), lr=a.lr, weight_decay=0.01)
        head_opt = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
        captured = {}

        def capture(_module, _inputs, output):
            # With one image per forward, all unpadded tiles belong to that image.
            captured["z"] = output.float().mean(dim=(0, 1), keepdim=False).unsqueeze(0)

        handle = connector.register_forward_hook(capture)
        initial = {k: v.detach().cpu().clone() for k, v in connector.state_dict().items()}
        position = 0
        total_gender = 0
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        with (out / "training.jsonl").open("w") as log:
            for step in range(1, steps + 1):
                opt.zero_grad(set_to_none=True)
                head_opt.zero_grad(set_to_none=True)
                batch = []
                for _ in range(a.accum):
                    if position == len(rows):
                        rng.shuffle(rows)
                        position = 0
                    batch.append(rows[position])
                    position += 1
                n_gender = sum(gender(r) is not None for r in batch)
                ce_sum, adv_sum = 0.0, 0.0
                for row in batch:
                    captured.clear()
                    inputs, prefix_len = inputs_for(row, training=True)
                    target = inputs["input_ids"][:, prefix_len:]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        result = m(**inputs, use_cache=False, logits_to_keep=target.shape[1] + 1)
                        # Causal shift: last N+1 positions predict the N assistant tokens.
                        ce = F.cross_entropy(result.logits[:, :-1].float().reshape(-1, result.logits.shape[-1]),
                                             target.reshape(-1))
                    loss = ce / a.accum
                    ce_sum += ce.item()
                    label = gender(row)
                    if kind == "adv" and label is not None:
                        logits = head(Reverse.apply(captured["z"], a.strength))
                        adv = F.cross_entropy(logits, torch.tensor([label], device="cuda"))
                        loss = loss + adv / n_gender
                        adv_sum += adv.item()
                        total_gender += 1
                    assert torch.isfinite(loss).item(), "Nonfinite loss"
                    loss.backward()
                    del result, loss, ce, inputs, target
                    captured.clear()
                grads = [p.grad for p in connector.parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all().item() for g in grads), "Invalid connector gradients"
                assert any(torch.count_nonzero(g).item() for g in grads), "Zero connector gradients"
                assert all(p.grad is None for p in m.parameters() if not p.requires_grad), "Frozen parameter has gradients"
                if kind == "adv" and n_gender:
                    hgrads = [p.grad for p in head.parameters() if p.grad is not None]
                    assert hgrads and all(torch.isfinite(g).all().item() for g in hgrads)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    head_opt.step()
                norm = torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
                opt.step()
                record = dict(step=step, occupation_loss=ce_sum/a.accum,
                              gender_loss=adv_sum/n_gender if kind == "adv" and n_gender else None,
                              gender_examples=n_gender, connector_gradient_norm=float(norm),
                              elapsed_seconds=time.monotonic()-start,
                              peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step == 1 or step % 10 == 0 or step == steps:
                    print(kind, record, flush=True)
                if step % 25 == 0 or step == steps:
                    payload = dict(connector={k:v.detach().cpu() for k,v in connector.state_dict().items()},
                                   adversary={k:v.detach().cpu() for k,v in head.state_dict().items()},
                                   step=step, kind=kind, manifest=manifest)
                    tmp = out / "checkpoint.tmp"
                    torch.save(payload, tmp)
                    tmp.replace(out / "checkpoint.pt")
        assert any(not torch.equal(initial[k], v.detach().cpu()) for k,v in connector.state_dict().items()), "Connector did not change"
        if kind == "adv":
            assert total_gender > 0, "No adversarial updates occurred"
        handle.remove()
        del m, connector, head, opt, head_opt, captured, grads, initial, payload
        torch.cuda.empty_cache()
        return out / "checkpoint.pt"

    def evaluate(checkpoints, rows, out):
        out.mkdir(parents=True, exist_ok=False)
        (out / "selected_val_ids.json").write_text(json.dumps([r["image_id"] for r in rows]))
        summaries = {}
        for name, ckpt in checkpoints:
            m = load_model()
            if ckpt is not None:
                saved = torch.load(ckpt, map_location="cpu", weights_only=True)
                assert saved["manifest"]["model_revision"] == revision
                # Retain the saved FP32 connector; use autocast as in training.
                m.model.connector.float()
                m.model.connector.load_state_dict(saved["connector"], strict=True)
                del saved
            groups = {}
            correct = 0
            with (out / f"{name}_predictions.csv").open("w", newline="") as f:
                fields = list(rows[0]) + ["model_response", "correct"]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for i, row in enumerate(rows, 1):
                    inputs = inputs_for(row)
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tokens = m.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
                    answer = processor.batch_decode(tokens[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
                    assert answer, "Empty prediction"
                    hit = answer.casefold() == row["occupation_raw"].strip().casefold()
                    correct += hit
                    groups.setdefault(row["counterfactual_set"], []).append(answer.casefold())
                    writer.writerow(dict(row, model_response=answer, correct=int(hit)))
                    f.flush()
                    if i % 100 == 0 or i == len(rows):
                        print(f"{name}: {i}/{len(rows)}", flush=True)
            summaries[name] = dict(images=len(rows), groups=len(groups),
                                   raw_exact_match_accuracy=correct/len(rows),
                                   raw_full_group_consistency=sum(len(set(v)) == 1 for v in groups.values())/len(groups))
            del m, inputs, tokens
            torch.cuda.empty_cache()
        (out / "summary.json").write_text(json.dumps(summaries, indent=2))
        print(json.dumps(summaries, indent=2), flush=True)

    if a.mode == "smoke":
        x = torch.tensor([1.0], device="cuda", requires_grad=True)
        Reverse.apply(x, 0.1).sum().backward()
        assert torch.allclose(x.grad, torch.tensor([-0.1], device="cuda"))
        ce = train("ce", a.out / "ce", 2)
        adv = train("adv", a.out / "adv", 2)
        evaluate([("ce", ce), ("adv", adv)], select_rows(data / "val.csv", 1, a.seed)[:1], a.out / "reload_check")
        print("SMOKE PASSED: gradients, updates, checkpoints and reload inference", flush=True)
    elif a.mode in ("ce", "adv"):
        train(a.mode, a.out / "train", a.steps)
    else:
        assert a.ce_checkpoint and a.adv_checkpoint
        ce_ids = a.ce_checkpoint.parent / "selected_train_ids.json"
        adv_ids = a.adv_checkpoint.parent / "selected_train_ids.json"
        assert ce_ids.read_text() == adv_ids.read_text(), "Training subsets differ"
        evaluate([("original", None), ("ce", a.ce_checkpoint), ("adv", a.adv_checkpoint)],
                 select_rows(data / "val.csv", a.val_groups, a.seed), a.out / "evaluation")
    (a.out / "SUCCESS").write_text("Completed\n")


if __name__ == "__main__":
    main()
