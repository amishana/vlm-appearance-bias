"""Fit a composed LEACE eraser from train_stats.pt.

Usage:
    python -m projection.fit_eraser --stats $FEAT/train_stats.pt \
        --types gender race physical --out $FEAT/eraser_all.pt
    python -m projection.fit_eraser --stats $FEAT/train_stats.pt \
        --types gender --out $FEAT/eraser_gender.pt
"""

import argparse
import json
from pathlib import Path

import torch

from projection.common import ATTR_TYPES
from projection.leace import fit_sequential


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--types", nargs="+", choices=ATTR_TYPES, default=ATTR_TYPES)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tol", type=float, default=1e-7)
    args = p.parse_args()

    stats = torch.load(args.stats)
    D, b, report = fit_sequential(stats, args.types, args.tol)
    meta = {"types": args.types, "stats": str(args.stats.resolve()), "report": report}
    torch.save({"D": D.float(), "b": b.float(), "meta": meta}, args.out)
    print(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
