#!/usr/bin/env python3
"""Export CLIP occupation predictions for the team's separate evaluation code.

No fairness metrics or effectiveness conclusion are computed here. Without an
adapter, this exports the specified original CLIP model as an integration aid.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import CLIPProcessor

from cf_model import (PROMPT, audit_splits, base_model_id, choose_device, dump_json,
                      file_hash, load_base, load_labels, occupation_logits,
                      package_versions, pixel_batch, processed_dir, read_config,
                      seed_everything, text_features)


def predict(cfg: dict, split: str, output: Path, adapter: Path | None = None) -> Path:
    seed_everything(int(cfg.get("training", {}).get("seed", 42)))
    folder = processed_dir(cfg)
    labels = load_labels(folder)
    splits, audit = audit_splits(folder, labels)
    rows = splits[split]
    device = choose_device(cfg.get("prediction", {}).get("device", cfg.get("training", {}).get("device", "auto")))
    adapter_metadata = None
    if adapter:
        adapter = adapter.resolve()
        saved_labels = json.loads((adapter / "labels.json").read_text(encoding="utf-8"))
        if saved_labels != labels:
            raise ValueError("Adapter class order differs from processed labels.json")
        adapter_metadata = json.loads((adapter / "metadata.json").read_text(encoding="utf-8"))
        if adapter_metadata["base_model"] != base_model_id(cfg):
            local_base = Path(base_model_id(cfg))
            expected_files = adapter_metadata.get("local_base_files_sha256", {})
            # Local smoke artifacts may move with the package; require identical weights/config.
            identical_local_base = bool(expected_files) and local_base.is_dir() and all(
                (local_base / name).is_file() and file_hash(local_base / name) == digest
                for name, digest in expected_files.items())
            if not identical_local_base:
                raise ValueError("Configured base model differs from the adapter's recorded base model")
        saved_revision = adapter_metadata.get("resolved_base_commit") or adapter_metadata.get("requested_revision")
        cfg = dict(cfg, model=dict(cfg["model"], revision=saved_revision))
    model, processor = load_base(cfg)
    if adapter:
        processor = CLIPProcessor.from_pretrained(adapter / "processor", local_files_only=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    classes = text_features(model, processor, labels, device)
    batch_size = int(cfg.get("prediction", {}).get("batch_size", 16))
    if batch_size < 1:
        raise ValueError("prediction.batch_size must be positive")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Prediction file exists; choose a new --output: {output}")
    fields = ["sample_id", "group_id", "image_path", "occupation", "label_id", "dataset_segment",
              "a1_type", "a2_type", "a1", "a2", "a1a2", "image_sha256", "source_index", "split",
              "predicted_occupation", "predicted_label_id", "confidence", "class_order_json",
              "logits_json", "probabilities_json", "model_id", "adapter_path", "run_status"]
    status = cfg.get("experiment", {}).get("status", "PRETRAINED_CLIP_PREDICTIONS")
    if adapter_metadata:
        status = adapter_metadata["status"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        with torch.no_grad():
            for start in range(0, len(rows), batch_size):
                batch = rows[start:start + batch_size]
                logits = occupation_logits(model, pixel_batch(cfg, processor, batch, device, verify_hash=True), classes)
                probs = logits.softmax(dim=-1)
                if not torch.isfinite(logits).all() or not torch.isfinite(probs).all():
                    raise FloatingPointError("Nonfinite prediction scores")
                for row, scores, probabilities in zip(batch, logits.cpu().tolist(), probs.cpu().tolist()):
                    label_id = max(range(len(labels)), key=lambda index: scores[index])
                    record = {key: row[key] for key in fields if key in row}
                    record.update(split=split, predicted_occupation=labels[label_id], predicted_label_id=label_id,
                                  confidence=probabilities[label_id], class_order_json=json.dumps(labels),
                                  logits_json=json.dumps(scores), probabilities_json=json.dumps(probabilities),
                                  model_id=base_model_id(cfg), adapter_path=str(adapter) if adapter else "",
                                  run_status=status)
                    writer.writerow(record)
    manifest_match = None if not adapter_metadata else (
        audit["manifest_sha256"] == adapter_metadata["split_audit"]["manifest_sha256"] and
        audit["labels_sha256"] == adapter_metadata["split_audit"]["labels_sha256"])
    dump_json(output.with_suffix(".metadata.json"), {"status": status, "records": len(rows), "split": split,
              "labels": labels, "prompt_template": PROMPT, "base_model": base_model_id(cfg),
              "revision": cfg["model"].get("revision"), "adapter": str(adapter) if adapter else None,
              "adapter_metadata": adapter_metadata, "packages": package_versions(), "device": str(device),
              "split_audit": audit, "output_sha256": file_hash(output),
              "data_matches_training_manifests": manifest_match,
              "heldout_protocol": "same_saved_test_manifest" if manifest_match and split == "test" else "other_or_unconfirmed_input_manifest",
              "evaluation_scope": "prediction export only; shared evaluation and capability retention belong to the team"})
    print(f"Exported {len(rows)} {split} predictions to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter", help="Saved adapter directory; omit for original CLIP predictions")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", default="outputs/predictions.csv")
    parser.add_argument("--smoke-test", action="store_true", help="Use existing outputs/smoke/smoke_config.yaml")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.smoke_test:
        from cf_model import resolve_path
        cfg = read_config(resolve_path(cfg, "outputs/smoke/smoke_config.yaml"))
    predict(cfg, args.split, Path(args.output), Path(args.adapter) if args.adapter else None)


if __name__ == "__main__":
    main()
