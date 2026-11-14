#!/usr/bin/env python3
"""Train vision-attention LoRA with group-balanced CE + counterfactual JS.

Use --smoke-test for a genuine offline optimizer/save/reload test on a tiny
random CLIP. That test is synthetic infrastructure evidence, not a VLM result.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, PeftModel, get_peft_model

from cf_model import (PROMPT, audit_splits, base_model_id, choose_device, dump_json,
                      file_hash, group_objective, group_rows, image_path, load_base,
                      load_labels, occupation_logits, package_versions, pixel_batch,
                      processed_dir, read_config, resolve_path, seed_everything,
                      text_features)


def create_smoke_config(original: dict, output_dir: str | Path | None = None) -> dict:
    """Create all artifacts locally; never download pretrained weights or data."""
    from PIL import Image, ImageDraw
    from transformers import (CLIPConfig, CLIPImageProcessor, CLIPModel, CLIPProcessor,
                              CLIPTextConfig, CLIPTokenizer, CLIPVisionConfig)
    from transformers.models.clip.tokenization_clip import bytes_to_unicode

    root = Path(output_dir).resolve() if output_dir is not None else resolve_path(original, "outputs/smoke")
    if (root / "training_logs.csv").exists() or (root / "best_adapter").exists():
        raise FileExistsError(f"Smoke output already exists: {root}. Choose a new project copy or preserve/rename it first.")
    base = root / "random_base"
    base.mkdir(parents=True, exist_ok=True)
    seed_everything(123)
    alphabet = list(bytes_to_unicode().values())
    vocab = {token: index for index, token in enumerate(alphabet + [x + "</w>" for x in alphabet] +
                                                        ["<|startoftext|>", "<|endoftext|>"])}
    dump_json(base / "vocab.json", vocab)
    (base / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(vocab_file=str(base / "vocab.json"), merges_file=str(base / "merges.txt"),
                              model_max_length=77)
    image_processor = CLIPImageProcessor(size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32})
    processor = CLIPProcessor(image_processor=image_processor, tokenizer=tokenizer)
    text_cfg = CLIPTextConfig(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
                              num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=77,
                              bos_token_id=vocab["<|startoftext|>"], eos_token_id=vocab["<|endoftext|>"],
                              pad_token_id=vocab["<|endoftext|>"])
    vision_cfg = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                  num_attention_heads=4, image_size=32, patch_size=8)
    model = CLIPModel(CLIPConfig.from_text_vision_configs(text_cfg, vision_cfg, projection_dim=16))
    model.save_pretrained(base)
    processor.save_pretrained(base)
    folder = root / "processed"
    folder.mkdir(parents=True, exist_ok=True)
    labels = ["nurse", "plumber"]
    dump_json(folder / "labels.json", labels)
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for label_id, label in enumerate(labels):
            for variant in range(2):
                sample_id = f"SYNTHETIC_{split}_{label}_{variant}"
                path = root / "images" / f"{sample_id}.png"
                path.parent.mkdir(exist_ok=True)
                # Deliberately artificial shapes; not SocialCounterfactuals people.
                image = Image.new("RGB", (32, 32), (40 + 80 * label_id, 40 + 70 * variant, 30 + 50 * split_index))
                draw = ImageDraw.Draw(image)
                draw.rectangle((3 + split_index, 4, 17 + label_id * 5, 20 + variant * 3), fill=(220, 110 + variant * 40, 100))
                image.save(path)
                rows.append({"sample_id": sample_id, "image_path": f"../images/{path.name}", "occupation": label,
                             "label_id": label_id, "group_id": f"SYNTHETIC_{split}_{label}",
                             "dataset_segment": "synthetic_random_smoke", "a1_type": "synthetic_color",
                             "a2_type": "synthetic_shape", "a1": str(variant), "a2": str(label_id),
                             "a1a2": f"{variant}_{label_id}", "image_sha256": file_hash(path),
                             "source_index": split_index * 4 + label_id * 2 + variant})
        (folder / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    cfg = {"data": {"processed_dir": "processed", "image_dir": "images", "source": "synthetic_random_smoke"},
           "model": {"backend": "clip", "name": "random_base", "revision": "local-random-initialization",
                     "local_files_only": True, "prompt_template": PROMPT},
           "lora": {"r": 2, "alpha": 4, "dropout": 0.0},
           "training": {"output_dir": ".", "device": "cpu", "seed": 123, "epochs": 2,
                        "groups_per_batch": 2, "max_images_per_group": None, "learning_rate": 0.001,
                        "weight_decay": 0.0, "max_grad_norm": 1.0, "consistency_weight": 0.1},
           "prediction": {"batch_size": 4},
           "experiment": {"status": "SYNTHETIC_RANDOM_SMOKE_TEST_NOT_A_PRETRAINED_RESULT"}}
    path = root / "smoke_config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return read_config(path)


def batches(groups: dict[str, list[dict]], size: int, rng: random.Random | None,
            max_images: int | None = None):
    ids = sorted(groups)
    if rng is not None:
        rng.shuffle(ids)
    for start in range(0, len(ids), size):
        selected = []
        for group_id in ids[start:start + size]:
            rows = sorted(groups[group_id], key=lambda row: row["sample_id"])
            if max_images is not None and len(rows) > max_images:
                rows = sorted(rng.sample(rows, max_images), key=lambda row: row["sample_id"])
            selected.extend(rows)
        yield selected


def run_validation(model, processor, classes, cfg, groups, device, weight, size):
    model.eval()
    total_loss = total_ce = total_js = 0.0
    correct = images = group_count = 0
    with torch.no_grad():
        for rows in batches(groups, size, None):
            logits = occupation_logits(model, pixel_batch(cfg, processor, rows, device), classes)
            targets = torch.tensor([row["label_id"] for row in rows], device=device)
            ids = [row["group_id"] for row in rows]
            loss, ce, js = group_objective(logits, targets, ids, weight)
            n_groups = len(set(ids))
            total_loss += loss.item() * n_groups
            total_ce += ce.item() * n_groups
            total_js += js.item() * n_groups
            group_count += n_groups
            correct += (logits.argmax(dim=-1) == targets).sum().item()
            images += len(rows)
    return {"val_loss": total_loss / group_count, "val_ce": total_ce / group_count,
            "val_js": total_js / group_count, "val_image_accuracy": correct / images,
            "val_images": images, "val_groups": group_count}


def configure_lora(model, lora: dict):
    """Attach adapters exclusively to CLIP vision q/v attention projections."""
    targets = [name for name, module in model.named_modules()
               if name.startswith("vision_model.") and name.endswith((".q_proj", ".v_proj"))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No supported CLIP vision q_proj/v_proj linear layers found.")
    return get_peft_model(model, LoraConfig(r=int(lora.get("r", 8)), lora_alpha=int(lora.get("alpha", 16)),
                                           lora_dropout=float(lora.get("dropout", 0.05)), target_modules=targets,
                                           bias="none"))


def train(cfg: dict) -> Path:
    tc = cfg["training"]
    seed = int(tc.get("seed", 42))
    seed_everything(seed)
    output = resolve_path(cfg, tc.get("output_dir", "outputs/run"))
    resume_path = output / "resume.pt"
    resuming = bool(tc.get("resume", False)) and resume_path.is_file()
    if not resuming and ((output / "training_logs.csv").exists() or (output / "best_adapter").exists()):
        raise FileExistsError(f"Training artifacts already exist in {output}; set a new training.output_dir to preserve them.")
    output.mkdir(parents=True, exist_ok=True)
    labels = load_labels(processed_dir(cfg))
    splits, audit = audit_splits(processed_dir(cfg), labels)
    dump_json(output / "split_audit.json", audit)
    train_rows, val_rows = splits["train"], splits["val"]
    # Test rows are deliberately discarded after metadata-only leakage auditing.
    del splits
    missing = sorted(set(labels) - {row["occupation"] for row in train_rows})
    if missing:
        raise ValueError(f"Training split does not cover full label vocabulary: {missing}")
    for row in train_rows + val_rows:
        if file_hash(image_path(cfg, row)) != row["image_sha256"]:
            raise ValueError(f"Image content changed since preparation: {row['sample_id']}")
    train_groups, val_groups = group_rows(train_rows), group_rows(val_rows)
    if any(len(rows) < 2 for rows in train_groups.values()):
        raise ValueError("Counterfactual training requires at least two images per group.")
    groups_per_batch = int(tc.get("groups_per_batch", 2))
    epochs = int(tc.get("epochs", 3))
    maximum = tc.get("max_images_per_group")
    maximum = int(maximum) if maximum is not None else None
    if groups_per_batch < 1 or epochs < 1 or (maximum is not None and maximum < 2):
        raise ValueError("epochs/groups_per_batch must be positive; max_images_per_group must be null or >=2")
    consistency = float(tc.get("consistency_weight", 0.1))
    if consistency < 0:
        raise ValueError("consistency_weight must be nonnegative")
    device = choose_device(tc.get("device", "auto"))
    model, processor = load_base(cfg)
    resolved_commit = getattr(model.config, "_commit_hash", None)
    lora = cfg.get("lora", {})
    targets = [name for name, module in model.named_modules()
               if name.startswith("vision_model.") and name.endswith((".q_proj", ".v_proj"))
               and isinstance(module, torch.nn.Linear)]
    if not targets:
        raise ValueError("No supported CLIP vision q_proj/v_proj linear layers found.")
    model = configure_lora(model, lora)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if not trainable or any("lora_" not in name or "vision_model." not in name for name in trainable):
        raise RuntimeError("Only vision LoRA parameters may be trainable.")
    if any(param.requires_grad for name, param in model.named_parameters() if "lora_" not in name):
        raise RuntimeError("A base-model parameter is unexpectedly trainable.")
    model.to(device)
    initial = {name: param.detach().cpu().clone() for name, param in trainable.items()}
    classes = text_features(model, processor, labels, device)
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=float(tc.get("learning_rate", 1e-4)),
                                 weight_decay=float(tc.get("weight_decay", 0.01)))
    max_grad_norm = float(tc.get("max_grad_norm", 1.0))
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")
    run_cfg = {key: value for key, value in cfg.items() if not key.startswith("_")}
    # A saved run config is usable from its new directory without changing path meaning.
    run_cfg["data"] = dict(run_cfg.get("data", {}), processed_dir=str(processed_dir(cfg)))
    configured_images = cfg.get("data", {}).get("image_dir")
    run_cfg["data"]["image_dir"] = str(resolve_path(cfg, configured_images) if configured_images else
                                        (processed_dir(cfg).parent / "images").resolve())
    run_cfg["model"] = dict(run_cfg["model"], name=base_model_id(cfg), revision=resolved_commit or cfg["model"].get("revision", "main"))
    run_cfg["training"] = dict(tc, output_dir=str(output))
    (output / "run_config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    metadata = {"status": cfg.get("experiment", {}).get("status", "COUNTERFACTUAL_CLIP_TRAINING"),
                "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "packages": package_versions(), "device": str(device), "seed": seed,
                "base_model": base_model_id(cfg), "requested_revision": cfg["model"].get("revision", "main"),
                "resolved_base_commit": resolved_commit, "prompt_template": PROMPT, "labels": labels,
                "trainable_parameters": sum(p.numel() for p in trainable.values()),
                "total_parameters": sum(p.numel() for p in model.parameters()), "target_modules": targets,
                "objective": "mean_over_groups(mean_image_CE + lambda * multiclass_Jensen_Shannon)",
                "consistency_weight": consistency, "group_sampling": "all" if maximum is None else f"up_to_{maximum}_without_replacement_per_epoch",
                "validation_sampling": "all_images_in_all_validation_groups", "split_audit": audit,
                "test_used_for_model_selection": False, "pretrained_effectiveness_claim": False}
    local_base = Path(base_model_id(cfg))
    if local_base.is_dir():
        metadata["local_base_files_sha256"] = {str(path.relative_to(local_base)): file_hash(path)
                                                for path in sorted(local_base.glob("*.safetensors"))}
        metadata["local_base_files_sha256"]["config.json"] = file_hash(local_base / "config.json")
    dump_json(output / "metadata.json", metadata)
    log_fields = ["epoch", "global_step", "train_loss", "train_ce", "train_js", "train_images", "train_groups",
                  "val_loss", "val_ce", "val_js", "val_image_accuracy", "val_images", "val_groups",
                  "mean_grad_norm", "nonzero_gradient_steps", "learning_rate", "epoch_seconds"]
    best = math.inf
    best_path = output / "best_adapter"
    global_step = nonzero_steps = 0
    start_epoch = 1
    restored = None
    if resuming:
        restored = torch.load(resume_path, map_location="cpu", weights_only=False)
        if restored["run_config"] != run_cfg:
            raise ValueError("Resume checkpoint configuration differs from this run")
        if restored["audit"] != audit:
            raise ValueError("Resume checkpoint manifests differ from this run")
        with torch.no_grad():
            for name, param in trainable.items():
                param.copy_(restored["parameters"][name].to(device))
        optimizer.load_state_dict(restored["optimizer"])
        global_step, nonzero_steps = restored["global_step"], restored["nonzero_steps"]
        best, start_epoch = restored["best"], restored["epoch"]
        metadata = restored["metadata"]
        torch.set_rng_state(restored["cpu_rng"])
        if device.type == "mps":
            torch.mps.set_rng_state(restored["device_rng"])
        elif device.type == "cuda":
            torch.cuda.set_rng_state_all(restored["device_rng"])
        for name, offset in restored["log_offsets"].items():
            with (output / name).open("r+b") as handle:
                handle.truncate(offset)
    started = time.monotonic()
    with (output / "training_logs.csv").open("a" if resuming else "w", newline="", encoding="utf-8") as log_handle, \
            (output / "training_batches.jsonl").open("a" if resuming else "w", encoding="utf-8") as batch_handle:
        writer = csv.DictWriter(log_handle, fieldnames=log_fields)
        if not resuming:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            loss_sum = ce_sum = js_sum = norm_sum = 0.0
            count_groups = count_images = steps = epoch_nonzero = 0
            skip = 0
            if restored and epoch == start_epoch:
                (loss_sum, ce_sum, js_sum, norm_sum, count_groups, count_images,
                 steps, epoch_nonzero) = restored["accumulators"]
                skip = restored["batch_index"]
            for batch_index, rows in enumerate(batches(train_groups, groups_per_batch, random.Random(seed + epoch), maximum), 1):
                if batch_index <= skip:
                    continue
                optimizer.zero_grad(set_to_none=True)
                targets_tensor = torch.tensor([row["label_id"] for row in rows], device=device)
                logits = occupation_logits(model, pixel_batch(cfg, processor, rows, device), classes)
                ids = [row["group_id"] for row in rows]
                loss, ce, js = group_objective(logits, targets_tensor, ids, consistency)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                if any(param.grad is not None for name, param in model.named_parameters() if "lora_" not in name):
                    raise RuntimeError("Frozen base-model parameter received a gradient.")
                norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), max_grad_norm)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                if norm.item() > 0:
                    nonzero_steps += 1
                    epoch_nonzero += 1
                optimizer.step()
                global_step += 1
                steps += 1
                count_images += len(rows)
                n_groups = len(set(ids))
                count_groups += n_groups
                loss_sum += loss.item() * n_groups
                ce_sum += ce.item() * n_groups
                js_sum += js.item() * n_groups
                norm_sum += norm.item()
                batch_handle.write(json.dumps({"epoch": epoch, "global_step": global_step,
                                               "sample_ids": [row["sample_id"] for row in rows],
                                               "group_ids": list(dict.fromkeys(ids))}) + "\n")
                if global_step % 100 == 0:
                    progress = {"epoch": epoch, "step": global_step, "epoch_groups": count_groups,
                                "total_train_groups": len(train_groups), "elapsed_seconds": time.monotonic() - started}
                    dump_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                interval = int(tc.get("checkpoint_steps", 0))
                if interval > 0 and global_step % interval == 0:
                    log_handle.flush()
                    batch_handle.flush()
                    state = {"run_config": run_cfg, "audit": audit, "epoch": epoch,
                             "batch_index": batch_index, "global_step": global_step,
                             "nonzero_steps": nonzero_steps, "best": best, "metadata": metadata,
                             "parameters": {n:p.detach().cpu() for n,p in trainable.items()},
                             "optimizer": optimizer.state_dict(), "cpu_rng": torch.get_rng_state(),
                             "device_rng": torch.mps.get_rng_state() if device.type == "mps" else (
                                 torch.cuda.get_rng_state_all() if device.type == "cuda" else None),
                             "accumulators": (loss_sum, ce_sum, js_sum, norm_sum, count_groups,
                                              count_images, steps, epoch_nonzero),
                             "log_offsets": {"training_logs.csv": log_handle.tell(),
                                             "training_batches.jsonl": batch_handle.tell()}}
                    torch.save(state, output / "resume.tmp")
                    os.replace(output / "resume.tmp", resume_path)
            validation = run_validation(model, processor, classes, cfg, val_groups, device, consistency, groups_per_batch)
            record = {"epoch": epoch, "global_step": global_step, "train_loss": loss_sum / count_groups,
                      "train_ce": ce_sum / count_groups, "train_js": js_sum / count_groups,
                      "train_images": count_images, "train_groups": count_groups, **validation,
                      "mean_grad_norm": norm_sum / steps, "nonzero_gradient_steps": epoch_nonzero,
                      "learning_rate": optimizer.param_groups[0]["lr"], "epoch_seconds": time.monotonic() - epoch_start}
            writer.writerow(record)
            log_handle.flush()
            batch_handle.flush()
            print(json.dumps(record), flush=True)
            if validation["val_loss"] < best:
                best = validation["val_loss"]
                best_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_path)
                processor.save_pretrained(best_path / "processor")
                dump_json(best_path / "labels.json", labels)
                (best_path / "run_config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
                metadata.update(best_epoch=epoch, best_val_loss=best)
                dump_json(best_path / "metadata.json", metadata)
    changed = [name for name, param in trainable.items() if not torch.equal(initial[name], param.detach().cpu())]
    if nonzero_steps == 0 or not changed:
        raise RuntimeError("Training was a no-op: no nonzero gradient or LoRA update.")
    # Reload saved adapter on the same base and verify its validation logits.
    probe = sorted(val_rows, key=lambda row: row["sample_id"])[:min(2, len(val_rows))]
    model.load_adapter(best_path, adapter_name="saved_check")
    model.set_adapter("saved_check")
    model.eval()
    with torch.no_grad():
        expected = occupation_logits(model, pixel_batch(cfg, processor, probe, device), classes).cpu()
    reloaded_base, reloaded_processor = load_base(cfg)
    reloaded = PeftModel.from_pretrained(reloaded_base, best_path).to(device).eval()
    reloaded_classes = text_features(reloaded, reloaded_processor, labels, device)
    with torch.no_grad():
        actual = occupation_logits(reloaded, pixel_batch(cfg, reloaded_processor, probe, device), reloaded_classes).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    metadata.update(nonzero_gradient_steps=nonzero_steps, changed_lora_tensors=len(changed),
                    optimizer_steps=global_step, elapsed_seconds=time.monotonic() - started,
                    adapter_reload_check={"passed": True, "validation_samples": len(probe),
                                          "max_abs_logit_difference": (actual - expected).abs().max().item()},
                    base_parameters_frozen=True)
    dump_json(output / "metadata.json", metadata)
    dump_json(best_path / "metadata.json", metadata)
    print(f"Saved best adapter: {best_path}", flush=True)
    return best_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smoke-test", action="store_true", help="Offline tiny RANDOM CLIP; no research claim")
    parser.add_argument("--output-dir", help="Fresh output directory, relative to the current working directory")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.smoke_test:
        cfg = create_smoke_config(cfg, args.output_dir)
    elif args.output_dir:
        cfg["training"]["output_dir"] = str(Path(args.output_dir).resolve())
    train(cfg)


if __name__ == "__main__":
    main()
