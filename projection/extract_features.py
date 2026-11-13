"""Extract connector features for one model (vision path only, no LM pass).

Writes to --out:
    <split>_pooled.npy   [N, d] float32, mean over the image's connector tokens
    <split>_meta.csv     image_id + labels, row-aligned with the .npy
    train_stats.pt       token-level LEACE statistics (only with --stats)

Usage (from the repo root):
    python -m projection.extract_features --backend qwen2_5_vl --model $MODEL \
        --data $DATA --split train --max-images 20000 --stats --out $FEAT
    python -m projection.extract_features --backend qwen2_5_vl --model $MODEL \
        --data $DATA --split val --out $FEAT
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from projection.common import (StopForward, capture_hook, image_attrs,
                               read_rows, sample_by_set)
from projection.leace import StatsAccumulator
from projection.models import BACKENDS, load_backend

META = ["image_id", "counterfactual_set", "occupation_raw", "dataset_segment",
        "a1_type", "a1", "a2_type", "a2", "n_tokens"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=list(BACKENDS), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-images", type=int, default=None,
                   help="sample whole counterfactual sets up to this many images")
    p.add_argument("--seed", type=int, default=566)
    p.add_argument("--stats", action="store_true",
                   help="accumulate LEACE statistics (train split only)")
    args = p.parse_args()
    if args.stats and args.split != "train":
        p.error("--stats is only allowed on the train split")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    args.out.mkdir(parents=True, exist_ok=True)
    rows = sample_by_set(read_rows(args.data, args.split), args.max_images, args.seed)
    print(f"{args.backend} {args.split}: {len(rows):,} images", flush=True)

    backend, name, connector = load_backend(args.backend, args.model)
    print(f"hooking {name}", flush=True)
    store = {}
    connector.register_forward_hook(capture_hook(store))

    pooled, meta, acc = [], [], None
    start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        with torch.inference_mode():
            try:
                backend.encode(args.data / row["image_path"])
            except StopForward:
                pass
        X = store.pop("x")  # [T, d]

        if args.stats:
            if acc is None:
                acc = StatsAccumulator(X.shape[1], X.device)
            acc.update(X, image_attrs(row))

        pooled.append(X.float().mean(0).cpu().numpy())
        meta.append({**{k: row[k] for k in META if k in row}, "n_tokens": X.shape[0]})

        if i % 500 == 0 or i == len(rows):
            rate = i / (time.perf_counter() - start)
            print(f"{i:,}/{len(rows):,} ({rate:.1f} img/s)", flush=True)

    np.save(args.out / f"{args.split}_pooled.npy", np.stack(pooled).astype(np.float32))
    with (args.out / f"{args.split}_meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META)
        w.writeheader()
        w.writerows(meta)
    if acc is not None:
        torch.save(acc.state_dict(), args.out / "train_stats.pt")

    n_tok = sorted({m["n_tokens"] for m in meta})
    print(f"connector tokens per image: {n_tok[:5]}{'...' if len(n_tok) > 5 else ''}, "
          f"dim {pooled[0].shape[0]}")
    if len(n_tok) > 1:
        print("note: token counts vary, so pooled erasure is approximate (see README)")
    print(f"wrote features to {args.out}")


if __name__ == "__main__":
    main()
