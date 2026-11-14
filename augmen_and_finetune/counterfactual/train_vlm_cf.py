"""Train a vision-side LoRA with the counterfactual consistency objective.

One optimizer step = one counterfactual group (same occupation and scene,
different attributes), subsampled to --images-per-group variants. Trains on
the TRAIN split only; val stays untouched for evaluation with eval/*.

Resumable: a checkpoint is written every --save-every steps; rerunning the same
command continues from it. A finished run writes DONE.json and is skipped.

Usage (from the repo root):
    python -m counterfactual.train_vlm_cf --backend qwen2_5_vl --model $MODEL \
        --data $DATA --out $CF/qwen2_5_vl/cf_lora --consistency-weight 0.1
    # CE-only control (isolates the effect of the consistency term):
    python -m counterfactual.train_vlm_cf ... --out $CF/qwen2_5_vl/ce_only --consistency-weight 0
"""

import argparse
import csv
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from counterfactual.cf_vlm import (answer_logits, answer_text, example_builder,
                                   group_objective, lora_targets)
from eval.occupation_map import EXCLUDED
from eval.score import normalize as label_normalize
from projection.common import read_rows
from projection.models import BACKENDS, DEVICE, load_backend

LOG_FIELDS = ["epoch", "global_step", "loss", "ce", "js", "grad_norm", "sec_per_step"]


def build_groups(data, max_groups, seed):
    groups = defaultdict(list)
    for r in read_rows(data, "train"):
        if label_normalize(r["occupation_raw"]) not in EXCLUDED:
            groups[r["counterfactual_set"]].append(r)
    ids = sorted(g for g, rows in groups.items() if len(rows) >= 2)
    if max_groups:
        rng = random.Random(seed)
        rng.shuffle(ids)
        ids = sorted(ids[:max_groups])
    return {g: sorted(groups[g], key=lambda r: r["image_id"]) for g in ids}


def epoch_order(group_ids, seed, epoch):
    ids = list(group_ids)
    random.Random(f"{seed}-{epoch}").shuffle(ids)
    return ids


def sample_variants(rows, k, seed, epoch, gid):
    if k is None or len(rows) <= k:
        return rows
    return sorted(random.Random(f"{seed}-{epoch}-{gid}").sample(rows, k), key=lambda r: r["image_id"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--consistency-weight", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--images-per-group", type=int, default=4)
    p.add_argument("--max-groups", type=int, default=None, help="subsample training groups")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    args = p.parse_args()

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    out = args.out
    if (out / "DONE.json").exists():
        print(f"already trained: {out}/DONE.json")
        return
    out.mkdir(parents=True, exist_ok=True)

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg_path = out / "train_args.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError(f"{out} holds a run with different arguments; use a new --out")
    cfg_path.write_text(json.dumps(config, indent=2))

    torch.manual_seed(args.seed)
    groups = build_groups(args.data, args.max_groups, args.seed)
    print(f"training groups: {len(groups):,} "
          f"({sum(len(v) for v in groups.values()):,} images)", flush=True)

    backend, _, _ = load_backend(args.backend, args.model)
    model = backend.model
    for prm in model.parameters():
        prm.requires_grad_(False)
    if not args.no_grad_checkpointing:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:  # InternVL remote code: enable on the language model
            model.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    build = example_builder(args.backend, backend)   # before wrapping (InternVL reads its module)
    targets = lora_targets(model, args.backend)
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                                             lora_dropout=args.lora_dropout,
                                             target_modules=targets, bias="none"))
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    for prm in trainable:
        prm.data = prm.data.float()          # LoRA weights in fp32, base stays bf16
    n_train = sum(prm.numel() for prm in trainable)
    print(f"LoRA on {len(targets)} vision-side layers, {n_train:,} trainable parameters", flush=True)
    backend.model = model

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, start_step, global_step = 1, 0, 0
    resume = out / "resume.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(model, load_file(str(out / "checkpoint" / "adapter_model.safetensors")))
        for prm in trainable:
            prm.data = prm.data.float()
        opt.load_state_dict(state["optimizer"])
        start_epoch, start_step, global_step = state["epoch"], state["step_in_epoch"], state["global_step"]
        print(f"resuming at epoch {start_epoch}, step {start_step} (global {global_step})", flush=True)

    log_path = out / "training_log.csv"
    new_log = not log_path.exists()
    log = log_path.open("a", newline="")
    writer = csv.DictWriter(log, fieldnames=LOG_FIELDS)
    if new_log:
        writer.writeheader()

    def checkpoint(epoch, step_in_epoch):
        model.save_pretrained(out / "checkpoint")
        torch.save({"optimizer": opt.state_dict(), "epoch": epoch,
                    "step_in_epoch": step_in_epoch, "global_step": global_step}, out / "resume.tmp")
        os.replace(out / "resume.tmp", resume)
        log.flush()

    model.train()
    for epoch in range(start_epoch, args.epochs + 1):
        order = epoch_order(groups, args.seed, epoch)
        acc = defaultdict(float)
        t0 = time.perf_counter()
        for step_in_epoch, gid in enumerate(order, 1):
            if epoch == start_epoch and step_in_epoch <= start_step:
                continue
            rows = sample_variants(groups[gid], args.images_per_group, args.seed, epoch, gid)
            answer = answer_text(rows[0]["occupation_raw"])
            logits, targets = [], []
            for r in rows:
                enc, labels = build(args.data / r["image_path"], answer, DEVICE)
                lg, tg = answer_logits(model, enc, labels)
                logits.append(lg)
                targets.append(tg)
            loss, ce, js = group_objective(logits, targets, args.consistency_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at group {gid}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            opt.step()
            global_step += 1
            for k, v in (("loss", loss), ("ce", ce), ("js", js), ("grad_norm", norm)):
                acc[k] += float(v)
            acc["n"] += 1

            if global_step % 20 == 0:
                dt = (time.perf_counter() - t0) / acc["n"]
                rec = {"epoch": epoch, "global_step": global_step,
                       **{k: round(acc[k] / acc["n"], 5) for k in ("loss", "ce", "js", "grad_norm")},
                       "sec_per_step": round(dt, 3)}
                writer.writerow(rec)
                left = (len(order) - step_in_epoch) * dt / 3600
                print(f"epoch {epoch} step {step_in_epoch:,}/{len(order):,} {rec} "
                      f"(~{left:.1f} h left in epoch)", flush=True)
                acc = defaultdict(float)
                t0 = time.perf_counter()
            if global_step % args.save_every == 0:
                checkpoint(epoch, step_in_epoch)
        checkpoint(epoch + 1, 0)

    model.save_pretrained(out / "adapter")
    log.close()
    (out / "DONE.json").write_text(json.dumps({
        "args": config, "lora_targets": targets, "trainable_parameters": n_train,
        "training_groups": len(groups), "global_steps": global_step,
        "objective": "mean_i CE(answer_i) + lambda * JS(answer distributions within group)",
        "split_used": "train only"}, indent=2))
    print(f"saved adapter to {out / 'adapter'}")


if __name__ == "__main__":
    main()
