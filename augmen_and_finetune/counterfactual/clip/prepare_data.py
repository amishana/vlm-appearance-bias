#!/usr/bin/env python3
"""Prepare complete, occupation-labelled SocialCounterfactuals groups.

The bounded ``hf_api`` source transfers metadata pages and selected images only.
The default does not fetch the 27 GB dataset. Relative ``image_path`` values in
the output manifests resolve against the manifest's parent directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image
import yaml


OFFICIAL_DATASET = "Intel/SocialCounterfactuals"
OFFICIAL_REVISION = "182f8d1739028dc3ab9e9f6be4e1cb52cf5d6fe1"
OFFICIAL_GROUP_SIZES = {"physical_gender": 10, "race_gender": 12, "physical_race": 30}
OFFICIAL_ATTRIBUTE_VALUES = {
    "physical": {"skinny", "obese", "young", "old", "tattooed"},
    "gender": {"male", "female"},
    "race": {"white", "black", "indian", "asian", "middle eastern", "latino"},
}
SPLIT_NAMES = ("train", "val", "test")
REQUIRED_METADATA = ("dataset_segment", "a1_type", "a2_type", "a1", "a2", "a1a2")


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def derive_occupation(row: dict) -> str:
    """Remove only the exact known attribute prefix; never use free-form guessing.

    Official rows have no occupation column. When a local manifest supplies an
    occupation, an available official caption is independently checked against it.
    """
    explicit = _text(row.get("occupation")).lower()
    caption = _text(row.get("caption"))
    attributes = _text(row.get("a1a2"))
    derived = None
    if caption:
        if not attributes:
            raise ValueError("caption requires a1a2 to derive an unambiguous occupation")
        # The source methodology uses four controlled caption templates.
        prefix = r"(?:(?:a photo of|a picture of|an image of) )?(?:a|an) "
        match = re.fullmatch(prefix + re.escape(attributes) + r" (.+)", caption, re.I)
        if not match:
            raise ValueError(f"Unsupported/ambiguous caption prefix: {caption!r}")
        derived = _text(match.group(1)).lower()
        if not re.fullmatch(r"[a-z][a-z0-9 &'/-]*", derived):
            raise ValueError(f"Ambiguous occupation suffix: {derived!r}")
    if not explicit and not derived:
        raise ValueError("Need an explicit occupation or an official caption and a1a2")
    if explicit and derived and explicit != derived:
        raise ValueError(f"Occupation/caption disagreement: {explicit!r} vs {derived!r}")
    occupation = explicit or derived
    group = _text(row.get("counterfactual_set") or row.get("group_id"))
    # Official IDs carry an occupation plus template index and generation seed.
    segment = _text(row.get("dataset_segment"))
    without_template = re.sub(r"^(?:a_photo_of_|a_picture_of_|an_image_of_)", "", group, flags=re.I)
    encoded_segment = next((known for known in OFFICIAL_GROUP_SIZES if without_template.startswith(known + "_")), None)
    if encoded_segment and encoded_segment != segment:
        raise ValueError(f"Group ID segment does not agree with dataset_segment: {group!r}")
    if without_template.startswith(segment + "_") and segment in OFFICIAL_GROUP_SIZES:
        encoded = re.sub(r"_\d+_\d+$", "", without_template[len(segment) + 1:])
        if encoded.replace("_", " ").lower() != occupation:
            raise ValueError(f"Group ID does not agree with occupation/caption: {group!r}")
    return occupation


def normalize_row(row: dict, index: int) -> dict:
    value = dict(row)
    value["occupation"] = derive_occupation(value)
    value["group_id"] = _text(value.get("counterfactual_set") or value.get("group_id"))
    if not value["group_id"]:
        raise ValueError(f"Row {index}: missing counterfactual_set/group_id")
    for field in REQUIRED_METADATA:
        value[field] = _text(value.get(field))
        if not value[field]:
            raise ValueError(f"Row {index}: missing {field}")
    if value["a1a2"].lower() != f"{value['a1']} {value['a2']}".lower():
        raise ValueError(f"Row {index}: a1a2 disagrees with a1/a2")
    if value["a1_type"] == value["a2_type"]:
        raise ValueError(f"Row {index}: counterfactual attribute types must differ")
    value["source_index"] = int(value.get("source_index", index))
    return value


def validate_groups(rows: list[dict], expected_group_size: int | None = None,
                    official: bool = False) -> dict[str, list[dict]]:
    """Reject mixed labels, missing variations, repeated cells and partial grids."""
    if not rows:
        raise ValueError("No usable rows selected")
    groups: dict[str, list[dict]] = defaultdict(list)
    segment_values: dict[str, tuple[set, set]] = {}
    for row in rows:
        group = row.get("group_id") or row.get("counterfactual_set")
        if not group:
            raise ValueError("Missing group_id/counterfactual_set")
        groups[group].append(row)
        segment = row["dataset_segment"]
        values = segment_values.setdefault(segment, (set(), set()))
        values[0].add(row["a1"])
        values[1].add(row["a2"])
    for group_id, group in groups.items():
        if len({r["occupation"] for r in group}) != 1:
            raise ValueError(f"Group {group_id}: mixed occupation labels")
        identity = {(r["dataset_segment"], r["a1_type"], r["a2_type"]) for r in group}
        if len(identity) != 1:
            raise ValueError(f"Group {group_id}: mixed attribute/segment definitions")
        segment = group[0]["dataset_segment"]
        a1, a2 = segment_values[segment]
        if len(a1) < 2 or len(a2) < 2:
            raise ValueError(f"Group {group_id}: missing counterfactual attribute variation")
        cells = [(r["a1"], r["a2"]) for r in group]
        if len(cells) != len(set(cells)):
            raise ValueError(f"Group {group_id}: duplicate counterfactual attribute cells")
        if set(cells) != set(itertools.product(a1, a2)):
            raise ValueError(f"Group {group_id}: incomplete counterfactual Cartesian grid")
        required = expected_group_size
        if official:
            if segment not in OFFICIAL_GROUP_SIZES:
                raise ValueError(f"Unsupported official dataset_segment: {segment}")
            required = OFFICIAL_GROUP_SIZES[segment]
            type1, type2 = segment.split("_")
            if group[0]["a1_type"] != type1 or group[0]["a2_type"] != type2:
                raise ValueError(f"Group {group_id}: official attribute types disagree with segment")
            if {value.lower() for value in a1} != OFFICIAL_ATTRIBUTE_VALUES[type1] or {value.lower() for value in a2} != OFFICIAL_ATTRIBUTE_VALUES[type2]:
                raise ValueError(f"Group {group_id}: unknown or missing official attribute values")
        if required is not None and len(group) != int(required):
            raise ValueError(f"Group {group_id}: has {len(group)} rows, expected {required}")
    return dict(groups)


def split_groups(rows: list[dict], seed: int = 42,
                 split_ratios: dict | list | None = None) -> dict[str, list[dict]]:
    """Deterministic occupation-stratified group allocation, all classes all splits."""
    ratios = split_ratios or {"train": .8, "val": .1, "test": .1}
    if isinstance(ratios, dict):
        ratios = [float(ratios[name]) for name in SPLIT_NAMES]
    else:
        ratios = list(map(float, ratios))
    if len(ratios) != 3 or any(not math.isfinite(x) or x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1., abs_tol=1e-8):
        raise ValueError("split_ratios must contain three positive values summing to 1")
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["occupation"]][row["group_id"]].append(row)
    seen_groups = {}
    for occupation, groups in grouped.items():
        for group in groups:
            if group in seen_groups and seen_groups[group] != occupation:
                raise ValueError(f"Group {group}: mixed occupation labels")
            seen_groups[group] = occupation
    result = {name: [] for name in SPLIT_NAMES}
    for occupation, groups in sorted(grouped.items()):
        n = len(groups)
        if n < 3:
            raise ValueError(f"Occupation {occupation!r} has {n} groups; at least 3 are needed for train/val/test")
        group_ids = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{occupation}:{group}".encode()).hexdigest())
        # Largest-remainder allocation with minimum one group in every split.
        quotas = [n * ratio for ratio in ratios]
        counts = [max(1, math.floor(quota)) for quota in quotas]
        while sum(counts) < n:
            index = max(range(3), key=lambda i: (quotas[i] - counts[i], -i))
            counts[index] += 1
        while sum(counts) > n:
            candidates = [i for i in range(3) if counts[i] > 1]
            index = max(candidates, key=lambda i: (counts[i] - quotas[i], -i))
            counts[index] -= 1
        offset = 0
        for split, count in zip(SPLIT_NAMES, counts):
            for group in group_ids[offset:offset + count]:
                result[split].extend(sorted(groups[group], key=lambda r: (r["a1"], r["a2"], r["source_index"])))
            offset += count
    audit_splits(result)
    return result


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    groups, images, pixels, samples = {}, {}, {}, {}
    for split, rows in splits.items():
        for row in rows:
            for key, seen in (("group_id", groups), ("image_sha256", images), ("pixel_sha256", pixels)):
                value = row.get(key)
                if value and value in seen and seen[value] != split:
                    raise ValueError(f"Leakage: {key} {value} occurs in {seen[value]} and {split}")
                if value:
                    seen[value] = split
            sample = row.get("sample_id")
            if sample and sample in samples:
                raise ValueError(f"Duplicate sample_id: {sample}")
            if sample:
                samples[sample] = split
    return {"group_leakage": False, "image_hash_leakage": False, "decoded_pixel_leakage": False,
            "unique_groups": len(groups), "unique_images": len(images), "unique_decoded_images": len(pixels)}


def _request(url: str, timeout: float = 45, max_bytes: int = 20 * 1024 * 1024) -> tuple[bytes, dict]:
    last_error = None
    for _ in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "counterfactual-finetuning/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError(f"Response exceeds {max_bytes} bytes")
                return data, dict(response.headers)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"Download failed for {urllib.parse.urlsplit(url)._replace(query='').geturl()}: {last_error}")


def _json_request(url: str, timeout: float = 45) -> tuple[dict, dict]:
    raw, headers = _request(url, timeout=timeout)
    return json.loads(raw), headers


def _api_page(data: dict, offset: int, length: int) -> tuple[dict, dict]:
    query = urllib.parse.urlencode({"dataset": data.get("dataset_name", OFFICIAL_DATASET),
                                   "config": data.get("hf_config", "default"),
                                   "split": data.get("hf_split", "train"),
                                   "offset": offset, "length": length})
    return _json_request("https://datasets-server.huggingface.co/rows?" + query,
                         float(data.get("request_timeout", 45)))


def load_hf_api(data: dict) -> tuple[list[dict], dict]:
    """Scan bounded metadata; take only intact official Cartesian groups.

    The viewer /rows API follows the current repository revision. Verify that
    every returned image asset and the repository SHA match the configured pin.
    Arbitrary older revisions require hf_streaming instead of pretending /rows
    supports revision pinning.
    """
    name = data.get("dataset_name", OFFICIAL_DATASET)
    if name != OFFICIAL_DATASET:
        raise ValueError("hf_api supports Intel/SocialCounterfactuals only; use local for another schema")
    pinned = data.get("revision") or OFFICIAL_REVISION
    api, _ = _json_request("https://huggingface.co/api/datasets/" + name)
    if api["sha"] != pinned:
        raise ValueError(f"Viewer is at revision {api['sha']}, configured {pinned}; use hf_streaming for a historical pin")
    occupations = [_text(x).lower() for x in data.get("occupations", [])]
    if not occupations or len(occupations) != len(set(occupations)):
        raise ValueError("hf_api requires an explicit, unique occupations list (the fixed label ontology)")
    target = int(data.get("groups_per_occupation", 5))
    if target < 3:
        raise ValueError("groups_per_occupation must be at least 3")
    max_rows = int(data.get("max_scan_rows", 5000))
    page_size = int(data.get("page_size", 100))
    if max_rows < 1 or not 1 <= page_size <= 100:
        raise ValueError("max_scan_rows must be positive; page_size must be in [1,100]")
    segments = set(data.get("segments") or OFFICIAL_GROUP_SIZES)
    if not segments <= set(OFFICIAL_GROUP_SIZES):
        raise ValueError("Unknown official segments")
    groups: dict[str, list[dict]] = defaultdict(list)
    total = None
    scanned = 0
    for offset in range(0, max_rows, page_size):
        page, _ = _api_page(data, offset, min(page_size, max_rows - offset))
        total = page["num_rows_total"]
        for entry in page["rows"]:
            scanned += 1
            raw = entry["row"]
            if entry.get("truncated_cells"):
                raise ValueError(f"Viewer truncated row {entry['row_idx']}")
            # Every asset path embeds the actual revision, unlike the API URL.
            asset = raw.get("image", {}).get("src", "")
            if f"/--/{pinned}/--/" not in asset:
                raise ValueError("Viewer image asset does not match pinned revision")
            if raw.get("dataset_segment") not in segments:
                continue
            row = normalize_row(raw, entry["row_idx"])
            if row["occupation"] in occupations:
                groups[row["group_id"]].append(row)
        complete = defaultdict(list)
        for group_id, group in groups.items():
            if len(group) == OFFICIAL_GROUP_SIZES[group[0]["dataset_segment"]]:
                complete[group[0]["occupation"]].append(group_id)
            elif len(group) > OFFICIAL_GROUP_SIZES[group[0]["dataset_segment"]]:
                raise ValueError(f"Group {group_id}: more variants than official schema")
        counts = {occupation: len(complete[occupation]) for occupation in occupations}
        print(f"Metadata scan {scanned}/{min(max_rows, total)}: complete groups {counts}", flush=True)
        if all(counts[occupation] >= target for occupation in occupations):
            chosen = {group_id for occupation in occupations for group_id in sorted(complete[occupation])[:target]}
            rows = [row for group_id in sorted(chosen) for row in groups[group_id]]
            validate_groups(rows, official=True)
            return rows, {"source": "hf_api", "dataset_name": name, "revision": pinned,
                          "source_split": data.get("hf_split", "train"), "source_total_rows": total,
                          "metadata_rows_scanned": scanned, "max_scan_rows": max_rows,
                          "selection": "first complete groups in bounded ordered metadata prefix, group IDs sorted",
                          "group_completeness": "official segment cardinality and unique full Cartesian attribute grid",
                          "dataset_card_url": f"https://huggingface.co/datasets/{name}/blob/{pinned}/README.md"}
        if not page["rows"] or offset + len(page["rows"]) >= total:
            break
    counts = Counter(group[0]["occupation"] for group in groups.values()
                     if len(group) == OFFICIAL_GROUP_SIZES[group[0]["dataset_segment"]])
    raise ValueError(f"Insufficient complete groups in bounded metadata scan: {dict(counts)}; need {target} per configured occupation. Increase max_scan_rows or revise occupations. No partial dataset was written.")


def load_local(data: dict, base_dir: Path) -> tuple[list[dict], dict]:
    path = Path(data["local_path"]).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if path.suffix.lower() == ".csv":
        import csv
        with path.open(newline="", encoding="utf-8-sig") as stream:
            raw_rows = list(csv.DictReader(stream))
    else:
        with path.open(encoding="utf-8") as stream:
            raw_rows = [json.loads(line) for line in stream if line.strip()]
    rows = []
    for index, raw in enumerate(raw_rows):
        row = normalize_row(raw, index)
        image = row.get("image_path") or row.get("image")
        if isinstance(image, dict):
            image = image.get("path")
        if not isinstance(image, str) or image.startswith(("http:", "https:")):
            raise ValueError("Local manifests require local image_path/image paths")
        image_path = Path(image).expanduser()
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
            if not image_path.is_relative_to(path.parent):
                raise ValueError(f"Local relative image path escapes manifest directory: {image}")
        if not image_path.is_file():
            raise ValueError(f"Missing local image: {image_path}")
        row["image"] = {"path": str(image_path)}
        rows.append(row)
    return rows, {"source": "local", "local_manifest": str(path),
                  "local_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "revision": data.get("revision"), "source_total_rows": len(rows)}


def load_hf_streaming(data: dict, image_dir: Path, processed_dir: Path) -> tuple[list[dict], dict]:
    from datasets import Image as HFImage, load_dataset
    name = data.get("dataset_name", OFFICIAL_DATASET)
    revision = data.get("revision") or OFFICIAL_REVISION
    source = load_dataset(name, name=data.get("hf_config", "default"),
                          split=data.get("hf_split", "train"), revision=revision, streaming=True)
    source = source.cast_column("image", HFImage(decode=False))
    # Exhaust the stream to collect all members even if groups are noncontiguous.
    # Materialize each selected image immediately: holding the embedded image
    # bytes for all 170k rows in memory would require tens of gigabytes.
    occupations = set(map(lambda x: _text(x).lower(), data.get("occupations", [])))
    segments = set(data.get("segments") or OFFICIAL_GROUP_SIZES)
    rows = []
    scanned = 0
    for index, raw in enumerate(source):
        scanned += 1
        row = normalize_row(raw, index)
        if occupations and row["occupation"] not in occupations or row["dataset_segment"] not in segments:
            continue
        prepared = _materialize(row, image_dir, processed_dir, float(data.get("request_timeout", 45)))
        row.pop("image", None)
        row["_prepared"] = prepared
        rows.append(row)
    return rows, {"source": "hf_streaming", "dataset_name": name, "revision": revision,
                  "source_split": data.get("hf_split", "train"), "source_total_rows": scanned,
                  "full_source_consumed": True}


def _materialize(row: dict, image_dir: Path, processed_dir: Path, timeout: float) -> dict:
    if "_prepared" in row:
        return dict(row["_prepared"])
    source = row.get("image") or row.get("image_path")
    if isinstance(source, dict) and source.get("src"):
        raw, _ = _request(source["src"], timeout=timeout)
    elif isinstance(source, dict) and source.get("bytes") is not None:
        raw = bytes(source["bytes"])
    else:
        path = source.get("path") if isinstance(source, dict) else source
        raw = Path(path).read_bytes()
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()  # Detect corrupt/truncated image data before any output.
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(f"{rgb.size}:RGB:".encode() + rgb.tobytes()).hexdigest()
            output = io.BytesIO()
            rgb.save(output, format="PNG")
            encoded = output.getvalue()
    except Exception as exc:
        raise ValueError(f"Corrupt image for group {row['group_id']} row {row['source_index']}: {exc}") from exc
    digest = hashlib.sha256(encoded).hexdigest()
    destination = image_dir / f"{digest}.png"
    if not destination.exists():
        # Atomic replacement also handles identical images in concurrent workers.
        with tempfile.NamedTemporaryFile(dir=image_dir, suffix=".tmp", delete=False) as temp:
            temp.write(encoded)
            temporary = Path(temp.name)
        os.replace(temporary, destination)
    record = {field: row[field] for field in ("occupation", "group_id", *REQUIRED_METADATA, "source_index")}
    record.update({"sample_id": "scf_" + hashlib.sha256(f"{row['group_id']}:{row['a1']}:{row['a2']}".encode()).hexdigest()[:24],
                   "image_path": os.path.relpath(destination, processed_dir), "image_sha256": digest,
                   "pixel_sha256": pixel_hash, "source_image_sha256": hashlib.sha256(raw).hexdigest()})
    return record


def prepare(config: dict, base_dir: str | Path | None = None) -> dict:
    """Prepare files and return the audited summary. Paths use config file's parent."""
    base_dir = Path(base_dir or config.get("_base_dir") or Path(__file__).parent).resolve()
    data = config.get("data", config)
    source = data.get("source", "hf_api")
    processed_dir = Path(data.get("processed_dir", data.get("output_dir", "data/processed")))
    image_dir = Path(data.get("image_dir", "data/images"))
    processed_dir = (base_dir / processed_dir).resolve()
    image_dir = (base_dir / image_dir).resolve()
    if processed_dir == image_dir or not image_dir.is_relative_to(base_dir) or not processed_dir.is_relative_to(base_dir):
        raise ValueError("processed_dir and image_dir must be distinct directories within the project/config directory")
    if any((processed_dir / f"{name}.jsonl").exists() for name in SPLIT_NAMES) and not data.get("overwrite", False):
        raise FileExistsError("Prepared manifests already exist; use a new processed_dir or --overwrite")
    if source == "hf_api":
        rows, provenance = load_hf_api(data)
    elif source == "hf_streaming":
        image_dir.mkdir(parents=True, exist_ok=True)
        rows, provenance = load_hf_streaming(data, image_dir, processed_dir)
    elif source == "local":
        rows, provenance = load_local(data, base_dir)
    else:
        raise ValueError(f"Unknown source: {source}")
    official = provenance.get("dataset_name") == OFFICIAL_DATASET or bool(data.get("official_schema", False))
    # Validate the complete source before applying local/full-input subset filters.
    validate_groups(rows, data.get("expected_group_size"), official=official)
    configured = [_text(x).lower() for x in data.get("occupations", [])]
    if len(configured) != len(set(configured)):
        raise ValueError("Duplicate configured occupation labels")
    if configured:
        rows = [r for r in rows if r["occupation"] in configured]
        absent = set(configured) - {r["occupation"] for r in rows}
        if absent:
            raise ValueError(f"Configured occupations absent from source: {sorted(absent)}")
    if data.get("segments"):
        rows = [r for r in rows if r["dataset_segment"] in set(data["segments"])]
    groups = validate_groups(rows, data.get("expected_group_size"), official=official)
    if source != "hf_api" and data.get("groups_per_occupation") is not None:
        target = int(data["groups_per_occupation"])
        if target < 3:
            raise ValueError("groups_per_occupation must be >=3")
        by_class = defaultdict(list)
        for group_id, group in groups.items():
            by_class[group[0]["occupation"]].append(group_id)
        for occupation, ids in by_class.items():
            if len(ids) < target:
                raise ValueError(f"Occupation {occupation}: only {len(ids)} complete groups; need {target}")
        chosen = {group for ids in by_class.values() for group in sorted(ids)[:target]}
        rows = [row for row in rows if row["group_id"] in chosen]
    absent = set(configured) - {row["occupation"] for row in rows}
    if absent:
        raise ValueError(f"Configured occupations absent after segment/group selection: {sorted(absent)}")
    labels = sorted(configured or {row["occupation"] for row in rows})
    preliminary = split_groups(rows, int(data.get("seed", config.get("seed", 42))),
                               data.get("split_ratios", data.get("splits")))
    processed_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=int(data.get("download_workers", 4))) as executor:
        prepared = list(executor.map(lambda row: _materialize(row, image_dir, processed_dir,
                                                              float(data.get("request_timeout", 45))), rows))
    label_to_id = {label: index for index, label in enumerate(labels)}
    by_group = {row["group_id"]: split for split, split_rows in preliminary.items() for row in split_rows}
    splits = {name: [] for name in SPLIT_NAMES}
    for row in prepared:
        row["label_id"] = label_to_id[row["occupation"]]
        splits[by_group[row["group_id"]]].append(row)
    audit = audit_splits(splits)
    summary = {"format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "provenance": provenance, "seed": int(data.get("seed", config.get("seed", 42))),
               "labels": labels, "label_ontology": "explicit configured subset" if configured else "all observed source occupations",
               "limited_label_space": bool(configured), "split_ratios_requested": data.get("split_ratios", {"train": .8, "val": .1, "test": .1}),
               "split_strategy": "occupation-stratified whole counterfactual groups; at least one group per class per split",
               "comparison_scope": "held-out counterfactual templates/seeds for known occupations; not the paper's occupation-heldout split",
               "image_path_base": "directory containing each JSONL manifest", "image_conversion": "RGB PNG; no crops, flips or fabricated attribute swaps",
               "augmentation": "real matched SocialCounterfactuals variants retained together; training sampler/objective handle grouped exposure",
               "audit": audit, "splits": {name: {"rows": len(values), "groups": len({r['group_id'] for r in values}),
                                                  "occupations": dict(Counter(r["occupation"] for r in values)),
                                                  "segments": dict(Counter(r["dataset_segment"] for r in values))}
                                         for name, values in splits.items()}}
    # Manifests are committed only after all images decode and all leakage checks pass.
    for split, values in splits.items():
        path = processed_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in values), encoding="utf-8")
    (processed_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (processed_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--source", choices=("hf_api", "hf_streaming", "local"))
    parser.add_argument("--local-path")
    parser.add_argument("--processed-dir")
    parser.add_argument("--image-dir")
    parser.add_argument("--occupations", nargs="+", help="Quote occupations containing spaces")
    parser.add_argument("--groups-per-occupation", type=int)
    parser.add_argument("--max-scan-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    data = config.setdefault("data", {})
    for key in ("source", "local_path", "processed_dir", "image_dir", "occupations", "groups_per_occupation", "max_scan_rows", "seed"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    if args.overwrite:
        data["overwrite"] = True
    prepare(config, args.config.resolve().parent)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        sys.exit(1)
