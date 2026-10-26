import argparse
import json
import re
from collections import Counter, defaultdict
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
]


def extract_occupation(caption, a1, a2):
    """
    Extract the occupation from a caption by removing
    the known caption prefix and attribute values.

    Example:
    'A photo of a skinny male accountant' -> 'accountant'
    """
    if not all(isinstance(x, str) for x in (caption, a1, a2)):
        return None

    caption = caption.strip()

    pattern = (
        r"^(?:(?:a (?:photo|picture)|an image) of\s+)?an?\s+"
        + re.escape(a1.strip())
        + r"\s+"
        + re.escape(a2.strip())
        + r"\s+(.+?)\s*$"
    )

    match = re.match(pattern, caption, flags=re.IGNORECASE)

    if not match:
        return None

    occupation = match.group(1).strip().lower()

    if not occupation:
        return None

    return occupation


def main():
    parser = argparse.ArgumentParser(
        description="Validate occupation extraction from SocialCounterfactuals."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/scratch2/amishana/csci566/processed/occupation_validation"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...", flush=True)

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
    ).select_columns(COLUMNS)

    total = 0
    successful = 0

    occupations = Counter()
    occupations_by_segment = defaultdict(Counter)
    group_occupations = defaultdict(set)
    group_sizes = Counter()
    failures = []

    for row in dataset:
        if args.limit is not None and total >= args.limit:
            break

        total += 1

        occupation = extract_occupation(
            row["caption"],
            row["a1"],
            row["a2"],
        )

        segment = str(row["dataset_segment"])
        group = str(row["counterfactual_set"])

        group_sizes[group] += 1

        if occupation is None:
            failures.append({
                "caption": row["caption"],
                "a1": row["a1"],
                "a2": row["a2"],
                "dataset_segment": segment,
                "counterfactual_set": group,
            })
        else:
            successful += 1
            occupations[occupation] += 1
            occupations_by_segment[segment][occupation] += 1
            group_occupations[group].add(occupation)

        if total % 10000 == 0:
            print(f"Processed {total:,} rows", flush=True)

    inconsistent_groups = {
        group: sorted(values)
        for group, values in group_occupations.items()
        if len(values) > 1
    }

    failed_groups = {
        row["counterfactual_set"]
        for row in failures
    }

    report = {
        "dataset": DATASET_NAME,
        "total_rows": total,
        "successful_extractions": successful,
        "failed_extractions": len(failures),
        "extraction_success_rate": (
            successful / total if total else 0
        ),
        "unique_occupations": len(occupations),
        "occupation_counts": dict(occupations.most_common()),
        "occupations_by_segment": {
            segment: dict(counts.most_common())
            for segment, counts in occupations_by_segment.items()
        },
        "total_counterfactual_groups": len(group_sizes),
        "groups_with_extraction_failures": len(failed_groups),
        "inconsistent_groups": inconsistent_groups,
        "group_size_distribution": dict(
            Counter(group_sizes.values()).most_common()
        ),
        "extraction_failures": failures,
    }

    output_file = args.output / "occupation_validation.json"

    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\nVALIDATION COMPLETE")
    print(f"Total rows: {total:,}")
    print(f"Successful extractions: {successful:,}")
    print(f"Failed extractions: {len(failures):,}")
    print(f"Success rate: {successful / total:.2%}" if total else "")
    print(f"Unique occupations: {len(occupations)}")
    print(f"Counterfactual groups: {len(group_sizes):,}")
    print(f"Inconsistent groups: {len(inconsistent_groups):,}")
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
