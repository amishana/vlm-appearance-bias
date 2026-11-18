"""
Make a small local test sample from the team's VALIDATION split.

Reads Amisha's baseline file (baseline/results/<model>/val_predictions.csv),
which lists every validation image, picks the N validation groups that appear
earliest in the dataset, and downloads only those images from Hugging Face.
Writes a val.csv with the same columns as the team's exported val.csv.

Run from the repo root:
    python make_local_sample.py --groups 5 --out data_local/sc_sample
"""
import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset

DATASET = "Intel/SocialCounterfactuals"
REVISION = "182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1"  # same pinned version as the team
BASELINE = Path("baseline/results/qwen2_5_vl/val_predictions.csv")
KEEP = ["image_id", "counterfactual_set", "occupation_raw", "dataset_segment",
        "a1_type", "a1", "a2_type", "a2", "image_path"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("data_local/sc_sample"))
    args = ap.parse_args()

    val = pd.read_csv(BASELINE, keep_default_na=False)[KEEP]
    val["source_row_index"] = val["image_id"].str[3:].astype(int)

    # choose the validation groups that appear earliest, so little is streamed
    order = val.groupby("counterfactual_set")["source_row_index"].max().sort_values()
    groups = order.index[:args.groups]
    sample = val[val["counterfactual_set"].isin(groups)].sort_values("source_row_index")
    wanted = dict(zip(sample["source_row_index"], sample.to_dict("records")))
    last = max(wanted)
    print(f"{len(sample)} validation images in {len(groups)} groups "
          f"(streaming the first {last + 1} dataset rows)")

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    ds = load_dataset(DATASET, split="train", streaming=True, revision=REVISION)

    saved = 0
    for idx, row in enumerate(ds):
        if idx > last:
            break
        if idx not in wanted:
            continue
        rec = wanted[idx]
        # make sure the dataset row matches the team's record
        for key in ("counterfactual_set", "a1", "a2"):
            if str(row[key]) != str(rec[key]):
                raise ValueError(f"Row {idx} mismatch on {key}; check dataset version")
        row["image"].convert("RGB").save(args.out / rec["image_path"])
        saved += 1

    sample.drop(columns="source_row_index").to_csv(args.out / "val.csv", index=False)
    print(f"Saved {saved} images and val.csv to {args.out}")


if __name__ == "__main__":
    main()