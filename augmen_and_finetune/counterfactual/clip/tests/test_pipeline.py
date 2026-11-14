"""Independent, offline regression checks for the scoped counterfactual component.

These checks certify data and implementation behavior, not model effectiveness.
The tiny CLIP used below has random weights and is never a research result.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from cf_model import audit_splits, group_objective, load_labels, occupation_logits
from prepare_data import (derive_occupation, normalize_row, prepare,
                          split_groups, validate_groups)


RACES = ["White", "Black", "Indian", "Asian", "Middle Eastern", "Latino"]
PHYSICAL = ["skinny", "young", "obese", "tattooed", "old"]
GENDERS = ["male", "female"]


@pytest.mark.parametrize("prefix,group_prefix", [
    ("A photo of an", "a_photo_of_"),
    ("A picture of an", "a_picture_of_"),
    ("An image of an", "an_image_of_"),
    ("an", ""),
])
def test_official_occupation_parser_retains_multiword_labels_and_attributes(prefix, group_prefix):
    row = {"caption": f"{prefix} Middle Eastern female construction worker",
           "a1a2": "Middle Eastern female", "dataset_segment": "race_gender",
           "counterfactual_set": f"{group_prefix}race_gender_construction_worker_4_123"}
    assert derive_occupation(row) == "construction worker"


def test_parser_rejects_source_label_and_attribute_disagreements():
    row = {"caption": "A photo of a White male doctor", "a1a2": "White male",
           "dataset_segment": "race_gender", "counterfactual_set": "a_photo_of_race_gender_doctor_0_1"}
    with pytest.raises(ValueError, match="disagreement"):
        derive_occupation({**row, "occupation": "teacher"})
    with pytest.raises(ValueError, match="Group ID"):
        derive_occupation({**row, "counterfactual_set": "a_photo_of_race_gender_lawyer_0_1"})
    with pytest.raises(ValueError, match="a1a2"):
        normalize_row({**row, "a1": "Black", "a2": "male", "a1_type": "race", "a2_type": "gender"}, 0)


def _rows_for_groups(segment="race_gender", n_groups=3, occupations=("construction worker", "doctor")):
    axes = {
        "race_gender": ("race", "gender", RACES, GENDERS),
        "physical_gender": ("physical", "gender", PHYSICAL, GENDERS),
        "physical_race": ("physical", "race", PHYSICAL, RACES),
    }
    a1_type, a2_type, a1_values, a2_values = axes[segment]
    output = []
    for occupation in occupations:
        for group in range(n_groups):
            gid = f"a_photo_of_{segment}_{occupation.replace(' ', '_')}_{group}_123"
            for a1 in a1_values:
                for a2 in a2_values:
                    row = {"caption": f"A photo of a {a1} {a2} {occupation}",
                           "counterfactual_set": gid, "dataset_segment": segment,
                           "a1_type": a1_type, "a2_type": a2_type,
                           "a1": a1, "a2": a2, "a1a2": f"{a1} {a2}"}
                    output.append(normalize_row(row, len(output)))
    return output


@pytest.mark.parametrize("segment,size", [("race_gender", 12), ("physical_gender", 10), ("physical_race", 30)])
def test_official_groups_require_every_attribute_cell(segment, size):
    rows = _rows_for_groups(segment)
    groups = validate_groups(rows, official=True)
    assert len(groups) == 6
    assert {len(group) for group in groups.values()} == {size}
    with pytest.raises(ValueError, match="incomplete|expected"):
        validate_groups(rows[1:], official=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_groups(rows + [dict(rows[0])], official=True)


def test_group_label_corruption_is_rejected():
    rows = _rows_for_groups()
    rows[0]["occupation"] = "surgeon"
    with pytest.raises(ValueError, match="mixed occupation"):
        validate_groups(rows, official=True)
    with pytest.raises(ValueError, match="mixed occupation"):
        split_groups(rows)


def test_official_grid_rejects_unknown_attributes_even_with_correct_total_size():
    rows = _rows_for_groups(n_groups=1, occupations=("doctor",))
    for row in rows:
        if row["a1"] == "White":
            row["a1"] = "UnknownRace"
            row["a1a2"] = f"UnknownRace {row['a2']}"
    assert len(rows) == 12
    with pytest.raises(ValueError, match="vocab|attribute|values|axes"):
        validate_groups(rows, official=True)


def test_official_grid_rejects_swapped_attribute_type_names():
    rows = _rows_for_groups(n_groups=1, occupations=("doctor",))
    for row in rows:
        row["a1_type"], row["a2_type"] = row["a2_type"], row["a1_type"]
    with pytest.raises(ValueError, match="type|axes"):
        validate_groups(rows, official=True)


def test_split_is_deterministic_under_source_order_changes_and_preserves_groups():
    rows = _rows_for_groups(n_groups=9)
    first = split_groups(rows, seed=17)
    second = split_groups(list(reversed(rows)), seed=17)
    group_sets = {split: {row["group_id"] for row in values} for split, values in first.items()}
    assert group_sets == {split: {row["group_id"] for row in values} for split, values in second.items()}
    assert set(first) == {"train", "val", "test"}
    assert not group_sets["train"] & group_sets["val"]
    assert not group_sets["train"] & group_sets["test"]
    assert not group_sets["val"] & group_sets["test"]
    assert sum(map(len, first.values())) == len(rows)
    for values in first.values():
        assert {row["occupation"] for row in values} == {"construction worker", "doctor"}
        assert {len(group) for group in validate_groups(values, official=True).values()} == {12}
    assert group_sets != {split: {row["group_id"] for row in values}
                          for split, values in split_groups(rows, seed=99).items()}


def test_split_refuses_insufficient_groups_per_class():
    with pytest.raises(ValueError, match="at least 3"):
        split_groups(_rows_for_groups(n_groups=2))


def _local_manifest(folder):
    rows = _rows_for_groups(n_groups=3)
    images = folder / "source_images"
    images.mkdir()
    for index, row in enumerate(rows):
        image = Image.new("RGB", (12, 12), (index, (index * 31) % 256, (index * 79) % 256))
        path = images / f"{index}.png"
        image.save(path)
        row["image_path"] = str(path.relative_to(folder))
    path = folder / "source.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path, rows


def test_local_preparation_emits_audited_manifests_and_preserves_sources(tmp_path):
    source, original_rows = _local_manifest(tmp_path)
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {"data": {"source": "local", "local_path": source.name,
                       "processed_dir": "data/processed", "image_dir": "data/images",
                       "official_schema": True, "seed": 9}}
    summary = prepare(config, base_dir=tmp_path)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    processed = tmp_path / "data/processed"
    labels = load_labels(processed)
    splits, report = audit_splits(processed, labels)
    assert report["passed"]
    assert labels == ["construction worker", "doctor"]
    assert sum(len(rows) for rows in splits.values()) == len(original_rows)
    assert all(len(rows) == 24 for rows in splits.values())
    assert summary["comparison_scope"].startswith("held-out counterfactual")
    for rows in splits.values():
        for row in rows:
            image = (processed / row["image_path"]).resolve()
            assert image.is_file()
            assert hashlib.sha256(image.read_bytes()).hexdigest() == row["image_sha256"]
    with pytest.raises(FileExistsError):
        prepare(config, base_dir=tmp_path)


def test_incomplete_local_input_writes_no_manifests(tmp_path):
    source, rows = _local_manifest(tmp_path)
    source.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]), encoding="utf-8")
    config = {"data": {"source": "local", "local_path": source.name,
                       "processed_dir": "data/processed", "image_dir": "data/images", "official_schema": True}}
    with pytest.raises(ValueError, match="incomplete|expected"):
        prepare(config, base_dir=tmp_path)
    assert not (tmp_path / "data/processed/train.jsonl").exists()


def test_js_known_value_and_finite_gradients():
    """Opposing near-deterministic distributions approach JS=log(2)."""
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]], requires_grad=True)
    total, ce, js = group_objective(logits, torch.zeros(2, dtype=torch.long), ["g", "g"], 0.7)
    assert js.item() == pytest.approx(math.log(2.0), abs=1e-6)
    assert total.item() == pytest.approx((ce + 0.7 * js).item())
    total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_identical_group_predictions_have_zero_js_but_wrong_ce_is_not_zero():
    logits = torch.tensor([[2.0, -2.0], [2.0, -2.0], [2.0, -2.0]])
    _, ce, js = group_objective(logits, torch.ones(3, dtype=torch.long), ["g"] * 3, 1.0)
    assert abs(js.item()) < 1e-7
    assert ce.item() > 4.0  # Consistently wrong predictions cannot satisfy the task loss.


def test_objective_weights_groups_equally_and_is_permutation_invariant():
    a = torch.tensor([[3.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[0.0, 4.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    logits = torch.cat([a, b])
    targets = torch.tensor([0, 0, 1, 1, 1, 1])
    groups = ["a", "a", "b", "b", "b", "b"]
    actual = group_objective(logits, targets, groups, 0.25)
    individual_a = group_objective(a, targets[:2], groups[:2], 0.25)
    individual_b = group_objective(b, targets[2:], groups[2:], 0.25)
    for result, one, two in zip(actual, individual_a, individual_b):
        assert torch.allclose(result, (one + two) / 2, atol=1e-7)
    order = torch.tensor([4, 0, 3, 1, 5, 2])
    shuffled = group_objective(logits[order], targets[order], [groups[i] for i in order.tolist()], 0.25)
    for left, right in zip(actual, shuffled):
        assert torch.allclose(left, right, atol=1e-7)


def test_objective_rejects_mixed_occupation_groups_and_negative_weight():
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="one occupation"):
        group_objective(logits, torch.tensor([0, 1]), ["same", "same"], 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        group_objective(logits, torch.tensor([0, 0]), ["same", "same"], -0.1)


def _minimal_row(split, index=0):
    token = f"{split}_{index}"
    return {
        "sample_id": token,
        "image_path": f"images/{token}.png",
        "occupation": "construction worker",
        "label_id": 0,
        "group_id": f"group_{split}",
        "dataset_segment": "race_gender",
        "a1_type": "race",
        "a2_type": "gender",
        "a1": "Middle Eastern",
        "a2": "female",
        "a1a2": "Middle Eastern female",
        "image_sha256": hashlib.sha256(token.encode()).hexdigest(),
        "source_index": index,
    }


def _write_manifests(folder, mutate=None):
    folder.mkdir(parents=True, exist_ok=True)
    labels = ["construction worker", "doctor"]
    (folder / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    splits = {name: [_minimal_row(name)] for name in ("train", "val", "test")}
    if mutate:
        mutate(splits)
    for name, rows in splits.items():
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return labels


@pytest.mark.parametrize("field", ["sample_id", "group_id", "image_sha256"])
def test_manifest_audit_rejects_each_cross_split_leakage(tmp_path, field):
    def mutate(splits):
        splits["test"][0][field] = splits["train"][0][field]
    labels = _write_manifests(tmp_path, mutate)
    with pytest.raises(ValueError, match="Split leakage"):
        audit_splits(tmp_path, labels)


def test_manifest_audit_rejects_wrong_label_mapping(tmp_path):
    def mutate(splits):
        splits["test"][0]["label_id"] = 1
    labels = _write_manifests(tmp_path, mutate)
    with pytest.raises(ValueError, match="label order"):
        audit_splits(tmp_path, labels)


def test_labels_cannot_silently_reorder_classes(tmp_path):
    (tmp_path / "labels.json").write_text('["doctor", "construction worker"]', encoding="utf-8")
    with pytest.raises(ValueError, match="sorted"):
        load_labels(tmp_path)


def test_tiny_clip_vision_lora_updates_and_reload_equivalence(tmp_path):
    """Exercise real CLIP+PEFT mechanisms with no download or pretrained claims."""
    from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
    from peft import PeftModel
    from train_lora import configure_lora

    torch.manual_seed(41)
    text = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=16)
    vision = CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                              num_attention_heads=2, image_size=16, patch_size=8)
    base = CLIPModel(CLIPConfig.from_text_vision_configs(text, vision, projection_dim=8))
    pristine = copy.deepcopy(base)
    cfg = {"r": 2, "alpha": 4, "dropout": 0.0}
    model = configure_lora(base, cfg)
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert names, "At least one adapter parameter must be trainable."
    assert all("lora_" in name and ".vision_model." in name for name in names)
    frozen_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if not parameter.requires_grad}
    adapter_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                      if parameter.requires_grad}
    pixels = torch.randn(4, 3, 16, 16)
    classes = F.normalize(torch.randn(3, 8), dim=-1)
    targets = torch.tensor([0, 0, 1, 1])
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.05)
    model.train()
    loss, _, _ = group_objective(occupation_logits(model, pixels, classes), targets, ["a", "a", "b", "b"], 0.5)
    loss.backward()
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    optimizer.step()
    after = dict(model.named_parameters())
    assert any(not torch.equal(adapter_before[name], after[name].detach()) for name in adapter_before)
    assert all(torch.equal(value, after[name].detach()) for name, value in frozen_before.items())
    model.eval()
    with torch.no_grad():
        expected = occupation_logits(model, pixels, classes)
    model.save_pretrained(tmp_path / "adapter")
    reloaded = PeftModel.from_pretrained(pristine, tmp_path / "adapter").eval()
    with torch.no_grad():
        actual = occupation_logits(reloaded, pixels, classes)
    assert torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
