import argparse
import csv
import hashlib
import io
import json
import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image


DATASET = "Intel/SocialCounterfactuals"
SPLITS = ("train", "val", "test")
VERIFY_KEYS = (
    "caption", "a1", "a2",
    "counterfactual_set", "dataset_segment",
    "a1_type", "a2_type",
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def process_image(image_id, image_bytes, image_src_path, out_path):
    """Worker: decode, convert, save as PNG (if needed), and hash.

    Runs in a separate process. image_bytes is None when the file
    already exists and only needs hashing.
    """
    out_path = Path(out_path)
    wrote = False

    if image_bytes is not None or image_src_path is not None:
        source = io.BytesIO(image_bytes) if image_bytes is not None else image_src_path
        temporary = out_path.with_name(out_path.name + ".tmp")
        with Image.open(source) as image:
            image.convert("RGB").save(temporary, format="PNG")
        os.replace(temporary, out_path)
        wrote = True

    return image_id, sha256_file(out_path), wrote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of worker processes for decoding/saving/hashing",
    )
    parser.add_argument(
        "--max-pending", type=int, default=None,
        help="Max images queued in memory (default: 4 x workers)",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    max_pending = args.max_pending or args.workers * 4

    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    images_dir.mkdir(exist_ok=True)

    metadata = read_csv(args.metadata)
    columns = list(metadata[0].keys())

    required = {
        "image_id", "source_row_index", "caption",
        "a1", "a2", "counterfactual_set", "occupation_raw",
        "image_path",
    }
    if not required.issubset(columns):
        raise ValueError("Missing required metadata columns")

    by_index = {}
    for row in metadata:
        idx = int(row["source_row_index"])
        if idx in by_index:
            raise ValueError(f"Duplicate source row index: {idx}")
        by_index[idx] = row

    # Record the exact source commit for reproducibility.
    revision = args.revision or HfApi().dataset_info(DATASET).sha
    print(f"Dataset revision: {revision}", flush=True)

    # The existing metadata was generated without a pinned revision.
    # We verify each row before exporting its image.
    dataset = load_dataset(
        DATASET,
        split="train",
        streaming=True,
        revision=revision,
    )
    # Keep raw bytes so decoding happens in the workers, and rows we
    # skip are never decoded at all.
    dataset = dataset.cast_column("image", HFImage(decode=False))

    selected = set(by_index)
    if args.limit is not None:
        selected = set(sorted(selected)[:args.limit])

    exported = 0
    skipped = 0
    completed = 0
    hashes = {}
    matched = set()
    pending = set()

    def collect(done_futures):
        nonlocal exported, skipped, completed
        for future in done_futures:
            image_id, digest, wrote = future.result()
            hashes[image_id] = digest
            if wrote:
                exported += 1
            else:
                skipped += 1
            completed += 1
            if completed % 1000 == 0:
                print(
                    f"Processed {completed:,}/{len(selected):,} "
                    f"selected images",
                    flush=True,
                )

    # "spawn" avoids fork-after-threads issues with the HF streaming client.
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=context
    ) as executor:
        for idx, source in enumerate(dataset):
            if len(matched) == len(selected):
                break

            if idx not in selected:
                continue

            row = by_index[idx]

            # Check the source record before associating its image.
            for key in VERIFY_KEYS:
                if str(source[key]) != str(row[key]):
                    raise ValueError(
                        f"Source mismatch at row {idx}, field {key}. "
                        "Dataset ordering or revision may have changed."
                    )

            matched.add(idx)

            filename = f"{row['image_id']}.png"
            path = images_dir / filename

            # Store paths relative to the output directory.
            row["image_path"] = f"images/{filename}"

            if path.exists() and path.stat().st_size > 0:
                image_bytes, image_src_path = None, None
            else:
                image_bytes = source["image"].get("bytes")
                image_src_path = source["image"].get("path")
                if image_bytes is None and image_src_path is None:
                    raise ValueError(f"No image data at row {idx}")
                if image_bytes is not None:
                    image_src_path = None

            # Bound memory: wait for some workers before queueing more.
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect(done)

            pending.add(executor.submit(
                process_image,
                row["image_id"], image_bytes, image_src_path, str(path),
            ))

        # Drain remaining work.
        done, _ = wait(pending)
        collect(done)

    if matched != selected:
        raise ValueError(
            f"Missing {len(selected - matched)} source records"
        )

    # Write a complete manifest only after successful export.
    if args.limit is None:
        output_rows = metadata
    else:
        output_rows = [
            row for row in metadata
            if int(row["source_row_index"]) in selected
        ]

    write_csv(
        args.output / "metadata_with_images.csv",
        output_rows,
        columns,
    )

    # Generate split manifests for fully exported datasets.
    if args.limit is None:
        for name in SPLITS:
            split_path = args.splits / f"{name}.csv"
            split_rows = read_csv(split_path)
            for row in split_rows:
                idx = int(row["source_row_index"])
                row["image_path"] = by_index[idx]["image_path"]

            write_csv(
                args.output / f"{name}.csv",
                split_rows,
                list(split_rows[0].keys()),
            )

    report = {
        "dataset": DATASET,
        "revision": revision,
        "selected_images": len(selected),
        "newly_exported": exported,
        "already_existing": skipped,
        "complete": len(matched) == len(selected),
        "workers": args.workers,
        "output_directory": str(args.output),
    }

    (args.output / "export_report.json").write_text(
        json.dumps(report, indent=2)
    )

    with (args.output / "image_hashes.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "sha256"])
        writer.writerows(sorted(hashes.items()))

    print("\nEXPORT COMPLETE")
    print(f"Images verified: {len(matched):,}")
    print(f"New images saved: {exported:,}")
    print(f"Existing images reused: {skipped:,}")
    print(f"Dataset revision: {revision}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()