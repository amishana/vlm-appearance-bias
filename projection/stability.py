"""Compare erasers fit on nested training samples (5k, 10k, 20k, ... images).

For each eraser, against the reference (the largest sample by default):

    subspace_overlap   ||U_ref^T U||_F^2 / max(r_ref, r), in [0, 1]; U spans the
                       directions the eraser edits (range of D = A - I).
                       1 means the same edited subspace.
    rel_diff_D         ||D - D_ref|| / ||D_ref||
    rank               number of edited directions

Pair with the probe numbers in leakage.csv (column train_images) to show that
leakage and occupation decodability stop changing as the sample grows.

Usage (from the repo root):
    python -m projection.stability --model-name qwen2_5_vl \
        --erasers $FEAT/eraser_all_n5000.pt $FEAT/eraser_all_n10000.pt \
                  $FEAT/eraser_all_n20000.pt $FEAT/eraser_all_n40000.pt \
        --out projection/eval_results/stability.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch


def edited_subspace(D, tol=1e-6):
    U, s, _ = torch.linalg.svd(D.double(), full_matrices=False)
    return U[:, s > tol * s.max()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--erasers", type=Path, nargs="+", required=True)
    p.add_argument("--reference", type=Path, default=None,
                   help="defaults to the last eraser listed (the largest sample)")
    p.add_argument("--model-name", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    ref_path = args.reference or args.erasers[-1]
    D_ref = torch.load(ref_path)["D"].double()
    U_ref = edited_subspace(D_ref)

    rows = []
    for path in args.erasers:
        obj = torch.load(path)
        D = obj["D"].double()
        U = edited_subspace(D)
        overlap = float((U_ref.T @ U).pow(2).sum() / max(U_ref.shape[1], U.shape[1]))
        m = re.search(r"_n(\d+)", path.stem)
        rows.append({
            "model": args.model_name, "eraser": path.name,
            "train_images_target": int(m.group(1)) if m else None,
            "erased_types": " ".join(obj.get("meta", {}).get("types", [])),
            "reference": ref_path.name, "rank": U.shape[1],
            "subspace_overlap": overlap,
            "rel_diff_D": float((D - D_ref).norm() / D_ref.norm()),
        })

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        out = pd.concat([pd.read_csv(args.out), out], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
