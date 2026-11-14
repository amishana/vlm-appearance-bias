"""Occupation prediction with a counterfactual LoRA adapter merged in.

Same prompt, preprocessing and output columns as the baseline runners, so
eval/score.py and eval/metrics.py treat the result as one more model (named
after the output's parent folder). Resumable like the baseline runners.

Usage (from the repo root):
    python -m counterfactual.predict_vlm_cf --backend qwen2_5_vl --model $MODEL \
        --data $DATA --adapter $CF/qwen2_5_vl/cf_lora/adapter \
        --output projection/results/qwen2_5_vl_cf_lora/val_predictions.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from projection.common import FIELDS, normalize, read_rows
from projection.models import BACKENDS, load_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    from peft import PeftModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
    if not (args.adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"no LoRA adapter in {args.adapter}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.data, args.split)
    if args.limit is not None:
        rows = rows[:args.limit]
    selected = {r["image_id"] for r in rows}
    completed = set()
    if args.output.exists():
        with args.output.open(newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != FIELDS:
                raise ValueError("Existing output has unexpected columns")
            for r in reader:
                if r["image_id"] not in selected or r["image_id"] in completed:
                    raise ValueError("Unexpected or duplicate image ID in output")
                completed.add(r["image_id"])
    pending = [r for r in rows if r["image_id"] not in completed]
    print(f"Selected {len(rows):,}, done {len(completed):,}, remaining {len(pending):,}", flush=True)
    if not pending:
        return

    backend, _, _ = load_backend(args.backend, args.model)
    base_dtype = next(backend.model.parameters()).dtype
    merged = PeftModel.from_pretrained(backend.model, str(args.adapter)).merge_and_unload()
    backend.model = merged.to(base_dtype).eval()
    print(f"merged adapter {args.adapter}", flush=True)

    train_info = {}
    done = args.adapter.parent / "DONE.json"
    if done.exists():
        train_info = json.loads(done.read_text())["args"]
    settings = {
        **backend.baseline_settings(args.split),
        "intervention": "counterfactual_lora_vision",
        "adapter": str(args.adapter.resolve()),
        "consistency_weight": train_info.get("consistency_weight"),
        "train_args": train_info,
    }
    settings_path = args.output.with_suffix(".settings.json")
    if settings_path.exists():
        if json.loads(settings_path.read_text()) != settings:
            raise ValueError("Existing output uses different settings")
    elif args.output.exists():
        raise ValueError("Existing predictions have no settings file")
    else:
        settings_path.write_text(json.dumps(settings, indent=2))

    new_file = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
            f.flush()
        for i, row in enumerate(pending, 1):
            start = time.perf_counter()
            with torch.inference_mode():
                response = backend.generate(args.data / row["image_path"])
            result = {k: row[k] for k in FIELDS if k in row}
            result["model_response"] = response
            result["correct"] = int(normalize(response) == normalize(row["occupation_raw"]))
            result["inference_seconds"] = round(time.perf_counter() - start, 3)
            writer.writerow(result)
            f.flush()
            if i % 100 == 0 or i == len(pending):
                print(f"Processed {i:,}/{len(pending):,}", flush=True)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
