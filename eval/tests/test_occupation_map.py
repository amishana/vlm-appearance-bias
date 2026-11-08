from pathlib import Path

import pandas as pd
import pytest

from eval import occupation_map as om
from eval.score import normalize

REPO = Path(__file__).resolve().parents[2]
PREDICTIONS = REPO / "baseline" / "results" / "qwen2_5_vl" / "val_predictions.csv"


def _all_keys():
    yield from om.GOLD_TO_CANONICAL
    yield from om.EXCLUDED
    yield from om.ALIASES
    yield from om.CATEGORY_ONLY_ALIASES
    yield from om.GENDER_MARKED
    yield from om.CATEGORY


def test_keys_are_normalized_fixed_points():
    bad = [k for k in _all_keys() if normalize(k) != k]
    assert bad == []


def test_gold_and_excluded_are_disjoint():
    assert not set(om.GOLD_TO_CANONICAL) & set(om.EXCLUDED)


def test_every_canonical_class_has_exactly_one_category():
    seen = {}
    for cat, members in om.CATEGORY_MEMBERS.items():
        for cls in members:
            assert cls not in seen, f"{cls} in both {seen[cls]} and {cat}"
            seen[cls] = cat
    assert set(om.GOLD_TO_CANONICAL.values()) <= set(seen)


def test_alias_targets_are_canonical_classes():
    bad = {a: t for a, t in om.ALIASES.items() if t not in om.CANONICAL_CLASSES}
    assert bad == {}


def test_aliases_do_not_shadow_gold_labels():
    assert not set(om.ALIASES) & set(om.GOLD_TO_CANONICAL)
    assert not set(om.CATEGORY_ONLY_ALIASES) & set(om.GOLD_TO_CANONICAL)
    assert not set(om.CATEGORY_ONLY_ALIASES) & set(om.ALIASES)


def test_category_only_targets_are_known_categories():
    bad = {a: c for a, c in om.CATEGORY_ONLY_ALIASES.items()
           if c not in om.CATEGORY_MEMBERS}
    assert bad == {}


def test_gender_values():
    assert set(om.GENDER_MARKED.values()) <= {"male", "female"}


@pytest.mark.skipif(not PREDICTIONS.exists(), reason="baseline results not present")
def test_observed_gold_labels_are_covered():
    gold = set(pd.read_csv(PREDICTIONS, usecols=["occupation_raw"])["occupation_raw"])
    norm = {normalize(g) for g in gold}
    missing = norm - set(om.GOLD_TO_CANONICAL) - set(om.EXCLUDED)
    assert missing == set()
    assert len(gold) == 126


def test_lookup_cases():
    assert om.lookup("doctor") == ("doctor", "medical", None, False)
    assert om.lookup("physician") == ("doctor", "medical", None, False)
    assert om.lookup("businessman") == ("businessperson", "business", "male", False)
    assert om.lookup("artist") == (None, "visual arts", None, False)
    assert om.lookup("female doctor") == ("doctor", "medical", "female", False)
    assert om.lookup("male nurse") == ("nurse", "nursing", "male", False)
    assert om.lookup("student") == (None, None, None, True)
    # excluded gold labels are not valid prediction targets
    assert om.lookup("pensioner") == (None, None, None, True)
