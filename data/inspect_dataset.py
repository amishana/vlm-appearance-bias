import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "Intel/SocialCounterfactuals"

COLUMNS = [
    "dataset_segment",
    "a1_type",
    "a2_type",
    "counterfactual_set",
    "caption",
    "a1",
    "a2",
    "a1a2",
]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect SocialCounterfactuals metadata."
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch2/amishana/csci566/processed/inspection"
        ),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading SocialCounterfactuals...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    )

    # Inspect metadata without decoding images.
    dataset = dataset.select_columns(COLUMNS)

    segments = Counter()
    attribute_types = Counter()
    attributes = Counter()
    group_sizes = Counter()
    captions = []

    count = 0

    for row in dataset:
        if count >= args.limit:
            break

        segments[str(row["dataset_segment"])] += 1

        attribute_types[
            (str(row["a1_type"]), str(row["a2_type"]))
        ] += 1

        attributes[str(row["a1"])] += 1
        attributes[str(row["a2"])] += 1

        group_sizes[str(row["counterfactual_set"])] += 1

        if len(captions) < 100:
            captions.append({
                "caption": row["caption"],
                "a1": row["a1"],
                "a2": row["a2"],
                "counterfactual_set": row["counterfactual_set"],
            })

        count += 1

        if count % 100 == 0:
            print(f"Inspected {count} rows", flush=True)

    report = {
        "dataset": DATASET_NAME,
        "rows_inspected": count,
        "dataset_segments": dict(segments),
        "attribute_type_pairs": [
            {
                "a1_type": a,
                "a2_type": b,
                "count": n,
            }
            for (a, b), n in attribute_types.most_common()
        ],
        "attribute_values": dict(attributes.most_common()),
        "counterfactual_groups_seen": len(group_sizes),
        "group_sizes_in_sample": dict(group_sizes),
        "sample_captions": captions,
    }

    output_file = args.output / "inspection_report.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\nInspection complete.")
    print(f"Rows inspected: {count}")
    print(f"Counterfactual groups observed: {len(group_sizes)}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
