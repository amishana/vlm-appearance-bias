import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

EXPECTED_GROUP_SIZES = {
    "physical_gender": 10,
    "physical_race": 30,
    "race_gender": 12,
}


def validate_metadata(df):
    required = {
        "image_id",
        "counterfactual_set",
        "occupation_raw",
        "dataset_segment",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs found")

    if df[list(required)].isna().any().any():
        raise ValueError("Missing required metadata values")

    groups = df.groupby("counterfactual_set", sort=True).agg(
        group_size=("image_id", "size"),
        occupation=("occupation_raw", "first"),
        occupation_count=("occupation_raw", "nunique"),
        segment=("dataset_segment", "first"),
        segment_count=("dataset_segment", "nunique"),
    ).reset_index()

    if (groups["occupation_count"] != 1).any():
        raise ValueError("Inconsistent occupations within groups")

    if (groups["segment_count"] != 1).any():
        raise ValueError("Groups span multiple segments")

    expected = groups["segment"].map(EXPECTED_GROUP_SIZES)

    if expected.isna().any():
        raise ValueError("Unknown dataset segment")

    if (groups["group_size"] != expected).any():
        raise ValueError("Incomplete counterfactual groups")

    return groups


def create_assignments(groups, seed):
    """
    Split complete groups, stratified by segment and occupation.

    Rare occupation strata are assigned using the same
    proportional allocation method, with fewer guarantees
    about representation in every split.
    """
    rng = np.random.default_rng(seed)
    assignments = {}

    strata = groups.groupby(
        ["segment", "occupation"], sort=True
    )

    for _, stratum in strata:
        group_ids = sorted(stratum["counterfactual_set"].tolist())
        rng.shuffle(group_ids)

        n = len(group_ids)

        # Allocate group counts using largest remainders.
        raw = np.array(list(RATIOS.values())) * n
        counts = np.floor(raw).astype(int)

        remaining = n - counts.sum()
        remainders = raw - counts

        for idx in np.argsort(-remainders, kind="stable")[:remaining]:
            counts[idx] += 1

        start = 0

        for split_name, count in zip(RATIOS, counts):
            for group_id in group_ids[start:start + count]:
                assignments[group_id] = split_name

            start += count

    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Create group-aware SocialCounterfactuals splits."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False)

    groups = validate_metadata(df)
    assignments = create_assignments(groups, args.seed)

    if len(assignments) != len(groups):
        raise ValueError("Not all groups received a split")

    df["split"] = df["counterfactual_set"].map(assignments)

    if df["split"].isna().any():
        raise ValueError("Some images were not assigned a split")

    # Verify no group appears in multiple splits.
    group_split_counts = df.groupby(
        "counterfactual_set"
    )["split"].nunique()

    if (group_split_counts != 1).any():
        raise ValueError("Counterfactual group leakage detected")

    args.output.mkdir(parents=True, exist_ok=True)

    report = {
        "seed": args.seed,
        "target_ratios": RATIOS,
        "total_images": len(df),
        "total_groups": len(groups),
        "splits": {},
    }

    for name in RATIOS:
        subset = df[df["split"] == name].copy()

        output_path = args.output / f"{name}.csv"
        subset.to_csv(output_path, index=False)

        report["splits"][name] = {
            "images": len(subset),
            "image_fraction": round(len(subset) / len(df), 4),
            "groups": subset["counterfactual_set"].nunique(),
            "unique_occupations": subset["occupation_raw"].nunique(),
            "images_by_segment": subset[
                "dataset_segment"
            ].value_counts().to_dict(),
        }

        print(
            f"{name}: {len(subset):,} images, "
            f"{report['splits'][name]['groups']:,} groups",
            flush=True,
        )

    # Save an authoritative group-to-split mapping.
    mapping = groups[["counterfactual_set"]].copy()
    mapping["split"] = mapping["counterfactual_set"].map(assignments)
    mapping.to_csv(
        args.output / "group_assignments.csv", index=False
    )

    with (args.output / "split_report.json").open("w") as f:
        json.dump(report, f, indent=2)

    print("\nSPLITS CREATED SUCCESSFULLY")
    print(f"Total images: {len(df):,}")
    print(f"Total groups: {len(groups):,}")
    print(f"Output directory: {args.output}")


if __name__ == "__main__":
    main()
