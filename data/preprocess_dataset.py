import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from validate_occupations import extract_occupation


DATASET_NAME = "Intel/SocialCounterfactuals"
DATASET_REVISION = "182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1"

COLUMNS = [
    "dataset_segment",
    "a1_type",
    "a2_type",
    "counterfactual_set",
    "caption",
    "a1",
    "a2",
]

OUTPUT_COLUMNS = [
    "image_id",
    "source_row_index",
    "image_path",
    "counterfactual_set",
    "occupation_raw",
    "caption",
    "dataset_segment",
    "a1_type",
    "a1",
    "a2_type",
    "a2",
]


def preprocess(output_dir, limit=None, revision=DATASET_REVISION):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    report_path = output_dir / "preprocessing_report.json"

    print("Loading SocialCounterfactuals...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        revision=revision,
    ).select_columns(COLUMNS)

    total = 0
    failed = 0

    occupations = Counter()
    segments = Counter()
    group_sizes = Counter()
    group_occupations = defaultdict(set)

    # Write incrementally to avoid keeping the dataset in RAM.
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for row_index, row in enumerate(dataset):
            if limit is not None and total >= limit:
                break

            occupation = extract_occupation(
                row["caption"],
                row["a1"],
                row["a2"],
            )

            if occupation is None:
                failed += 1
                raise ValueError(
                    f"Occupation extraction failed at row {row_index}: "
                    f"{row['caption']}"
                )

            group = str(row["counterfactual_set"])
            segment = str(row["dataset_segment"])

            writer.writerow({
                "image_id": f"sc_{row_index:06d}",
                "source_row_index": row_index,
                "image_path": "",
                "counterfactual_set": group,
                "occupation_raw": occupation,
                "caption": row["caption"],
                "dataset_segment": segment,
                "a1_type": row["a1_type"],
                "a1": row["a1"],
                "a2_type": row["a2_type"],
                "a2": row["a2"],
            })

            total += 1
            occupations[occupation] += 1
            segments[segment] += 1
            group_sizes[group] += 1
            group_occupations[group].add(occupation)

            if total % 10000 == 0:
                print(f"Processed {total:,} rows", flush=True)

    inconsistent_groups = {
        group: sorted(values)
        for group, values in group_occupations.items()
        if len(values) != 1
    }

    report = {
        "dataset": DATASET_NAME,
        "revision": revision,
        "total_rows": total,
        "failed_extractions": failed,
        "unique_occupations": len(occupations),
        "occupation_counts": dict(occupations.most_common()),
        "dataset_segments": dict(segments),
        "counterfactual_groups": len(group_sizes),
        "group_size_distribution": dict(
            Counter(group_sizes.values()).most_common()
        ),
        "inconsistent_groups": inconsistent_groups,
        "metadata_file": str(metadata_path),
        "images_exported": False,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nPREPROCESSING COMPLETE")
    print(f"Rows processed: {total:,}")
    print(f"Unique occupations: {len(occupations)}")
    print(f"Counterfactual groups: {len(group_sizes):,}")
    print(f"Inconsistent groups: {len(inconsistent_groups)}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SocialCounterfactuals metadata."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch2/amishana/csci566/processed/socialcounterfactuals"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing.",
    )

    parser.add_argument(
        "--revision",
        default=DATASET_REVISION,
        help="Hugging Face dataset revision.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")

    preprocess(args.output, args.limit, args.revision)


if __name__ == "__main__":
    main()
