"""Shared CLIP backend, data checks, and the group-level training objective.

This is a closed-vocabulary occupation classifier. Generative VLM backends require
their own implementation; changing a model ID does not provide that support.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROMPT = "A photo of a person working as a {occupation}."
REQUIRED = {"sample_id", "image_path", "occupation", "label_id", "group_id",
            "dataset_segment", "a1_type", "a2_type", "a1", "a2", "a1a2",
            "image_sha256", "source_index"}


def read_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping.")
    cfg["_config_path"] = str(path)
    cfg["_project_root"] = str(path.parent)
    if cfg.get("model", {}).get("backend") != "clip":
        raise ValueError("Only model.backend=clip is implemented. Generative VLMs need a new backend.")
    if cfg.get("model", {}).get("prompt_template", PROMPT) != PROMPT:
        raise ValueError(f"This experiment fixes the neutral occupation prompt to {PROMPT!r}.")
    return cfg


def resolve_path(cfg: dict, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(cfg["_project_root"]) / path).resolve()


def processed_dir(cfg: dict) -> Path:
    data = cfg.get("data", {})
    return resolve_path(cfg, data.get("processed_dir", data.get("output_dir", "data/processed")))


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(requested)


def load_labels(folder: Path) -> list[str]:
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    if (not isinstance(labels, list) or not labels or
            not all(isinstance(x, str) and x.strip() for x in labels) or
            labels != sorted(set(labels))):
        raise ValueError("labels.json must be a nonempty sorted list of unique occupation strings.")
    return labels


def load_rows(folder: Path, split: str, labels: list[str]) -> list[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    rows = []
    with (folder / f"{split}.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = REQUIRED - row.keys()
            if missing:
                raise ValueError(f"{split}:{line_number} missing {sorted(missing)}")
            if not isinstance(row["label_id"], int) or isinstance(row["label_id"], bool):
                raise ValueError(f"{split}:{line_number} label_id must be an integer")
            if not 0 <= row["label_id"] < len(labels) or labels[row["label_id"]] != row["occupation"]:
                raise ValueError(f"{split}:{line_number} label order disagrees with labels.json")
            for key in ("sample_id", "group_id", "image_path"):
                if not isinstance(row[key], str) or not row[key]:
                    raise ValueError(f"{split}:{line_number} invalid {key}")
            digest = row["image_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{split}:{line_number} invalid image_sha256")
            rows.append(row)
    if not rows:
        raise ValueError(f"{split}.jsonl contains no examples.")
    return rows


def audit_splits(folder: Path, labels: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Read test metadata solely for isolation checks; never return it as training input."""
    splits = {name: load_rows(folder, name, labels) for name in ("train", "val", "test")}
    seen: dict[str, dict[str, str]] = {key: {} for key in ("sample_id", "group_id", "image_sha256")}
    if any("pixel_sha256" in row for rows in splits.values() for row in rows):
        seen["pixel_sha256"] = {}
    group_occupation = {}
    for split, rows in splits.items():
        sample_ids = set()
        for row in rows:
            if row["sample_id"] in sample_ids:
                raise ValueError(f"Duplicate sample_id within {split}: {row['sample_id']}")
            sample_ids.add(row["sample_id"])
            previous_occupation = group_occupation.setdefault(row["group_id"], row["occupation"])
            if previous_occupation != row["occupation"]:
                raise ValueError(f"Counterfactual group spans occupations: {row['group_id']}")
            for key in seen:
                if key == "pixel_sha256":
                    digest = row.get(key)
                    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError(f"Missing or invalid pixel_sha256 for {row['sample_id']}")
                previous = seen[key].setdefault(row[key], split)
                if previous != split:
                    raise ValueError(f"Split leakage: {key}={row[key]} in {previous} and {split}")
    report = {"passed": True, "checks": list(seen) + ["label_order", "group_occupation"],
              "counts": {key: {"images": len(rows), "groups": len(group_rows(rows))} for key, rows in splits.items()},
              "manifest_sha256": {f"{key}.jsonl": file_hash(folder / f"{key}.jsonl") for key in splits},
              "labels_sha256": file_hash(folder / "labels.json"),
              "test_use_during_training": "metadata isolation audit only; no images, predictions, or model selection"}
    return splits, report


def group_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    return dict(groups)


def image_path(cfg: dict, row: dict) -> Path:
    candidate = Path(row["image_path"]).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (processed_dir(cfg) / candidate).resolve()
    configured_root = cfg.get("data", {}).get("image_dir")
    image_root = resolve_path(cfg, configured_root) if configured_root else (processed_dir(cfg).parent / "images").resolve()
    if not path.is_relative_to(image_root):
        raise ValueError(f"Image path escapes configured data.image_dir: {row['sample_id']}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing image for sample {row['sample_id']}: {path}")
    return path


def pixel_batch(cfg: dict, processor: CLIPProcessor, rows: list[dict], device: torch.device,
                verify_hash: bool = False) -> torch.Tensor:
    images = []
    for row in rows:
        path = image_path(cfg, row)
        if verify_hash and file_hash(path) != row["image_sha256"]:
            raise ValueError(f"Image hash disagrees with manifest: {row['sample_id']}")
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    return processor(images=images, return_tensors="pt")["pixel_values"].to(device)


def model_kwargs(cfg: dict) -> dict:
    model = cfg["model"]
    return {"revision": model.get("revision", "main"),
            "local_files_only": bool(model.get("local_files_only", False))}


def base_model_id(cfg: dict) -> str:
    name = cfg["model"]["name"]
    candidate = resolve_path(cfg, name)
    return str(candidate) if candidate.exists() else name


def load_base(cfg: dict) -> tuple[CLIPModel, CLIPProcessor]:
    name = base_model_id(cfg)
    kwargs = model_kwargs(cfg)
    model = CLIPModel.from_pretrained(name, **kwargs)
    processor = CLIPProcessor.from_pretrained(name, **kwargs)
    if model.config.model_type != "clip":
        raise ValueError("This backend only supports Transformers CLIPModel.")
    return model, processor


def text_features(model, processor: CLIPProcessor, labels: list[str], device: torch.device) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    tokens = processor(text=[PROMPT.format(occupation=label) for label in labels],
                       return_tensors="pt", padding=True, truncation=True).to(device)
    base.eval()
    with torch.no_grad():
        return F.normalize(base.get_text_features(**tokens).float(), dim=-1).detach()


def occupation_logits(model, pixels: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    features = F.normalize(base.get_image_features(pixel_values=pixels).float(), dim=-1)
    return base.logit_scale.exp().clamp(max=100.0) * features @ classes.T


def group_objective(logits: torch.Tensor, targets: torch.Tensor, group_ids: list[str],
                    consistency_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each group gets equal weight. JS is differentiable and covers all classes."""
    if consistency_weight < 0:
        raise ValueError("consistency_weight must be nonnegative")
    if logits.ndim != 2 or len(group_ids) != len(logits) or len(targets) != len(logits):
        raise ValueError("Invalid logits, targets, or group IDs")
    ce_per_image = F.cross_entropy(logits, targets, reduction="none")
    logp = F.log_softmax(logits, dim=-1)
    probabilities = logp.exp()
    group_ce, group_js = [], []
    for group in dict.fromkeys(group_ids):
        indices = [i for i, item in enumerate(group_ids) if item == group]
        idx = torch.tensor(indices, device=logits.device)
        if targets[idx].unique().numel() != 1:
            raise ValueError("Every counterfactual group must have one occupation label")
        group_ce.append(ce_per_image[idx].mean())
        p = probabilities[idx]
        log_mean = p.mean(dim=0).clamp_min(torch.finfo(p.dtype).tiny).log()
        group_js.append((p * (logp[idx] - log_mean)).sum(dim=-1).mean())
    ce = torch.stack(group_ce).mean()
    js = torch.stack(group_js).mean()
    return ce + consistency_weight * js, ce, js


def package_versions() -> dict:
    output = {}
    for name in ("torch", "transformers", "peft", "datasets", "numpy", "Pillow", "PyYAML"):
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = None
    return output
