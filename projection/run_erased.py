"""Baseline occupation prediction with the eraser applied to the connector.

Output has exactly the baseline columns (FIELDS), so eval/score.py and
eval/metrics.py treat it as one more model. The model name used by eval is the
output's parent folder, e.g. projection/results/qwen2_5_vl_leace_all/.

Usage (from the repo root):
    python -m projection.run_erased --backend qwen2_5_vl --model $MODEL \
        --data $DATA --eraser $FEAT/eraser_gender_race_physical.pt --alpha 1 \
        --output projection/results/qwen2_5_vl_leace_all/val_predictions.csv

Sanity check: --alpha 0 must reproduce baseline/results/<backend>/ exactly.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from projection.common import FIELDS, AffineEraser, normalize, read_rows
from projection.models import BACKENDS, DEVICE, load_backend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--eraser", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="0 = no erasure (sanity check), 1 = full LEACE")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")
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
    print(f"Selected {len(rows):,}, done {len(completed):,}, "
          f"remaining {len(pending):,}", flush=True)
    if not pending:
        return

    backend, name, connector = load_backend(args.backend, args.model)
    eraser = AffineEraser.load(args.eraser, alpha=args.alpha, device=DEVICE)
    connector.register_forward_hook(eraser.hook())
    print(f"eraser {args.eraser.name} (types={eraser.meta.get('types')}, "
          f"alpha={args.alpha}) on {name}", flush=True)

    settings = {
        **backend.baseline_settings(args.split),
        "intervention": "leace_connector",
        "connector_module": name,
        "eraser": str(args.eraser.resolve()),
        "erased_types": eraser.meta.get("types"),
        "alpha": args.alpha,
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
