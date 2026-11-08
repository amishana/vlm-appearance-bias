from pathlib import Path

import pandas as pd
import pytest

from eval.score import normalize, score_frame, strict_normalize, unmapped_report

REPO = Path(__file__).resolve().parents[2]
RESULT_FILES = sorted((REPO / "baseline" / "results").glob("*/val_predictions.csv"))


@pytest.mark.parametrize("raw, expected", [
    ("Chess player.", "chess player"),
    ("Real-Estate Developer", "real estate developer"),
    ("  Doctor  ", "doctor"),
    ("Professional/Office Worker", "professional office worker"),
    ("A nurse", "nurse"),
    ("ButButcher[[0, 5, 582, 883]]", "butbutcher"),
    ("Customer service representative.", "customer service representative"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def _row(gold, response):
    return {"occupation_raw": gold, "model_response": response}


def test_tiers_on_hand_built_rows():
    df = pd.DataFrame([
        _row("physician", "Doctor"),                 # synonym hit
        _row("nurse practitioner", "Nurse"),         # category-only hit
        _row("nurse practitioner", "Doctor"),        # category miss (nursing vs medical)
        _row("businessperson", "Businessman"),       # gendered synonym hit
        _row("painter", "Artist"),                   # category-only alias
        _row("pensioner", "Retiree"),                # excluded gold
        _row("marine biologist", "Diver"),           # unmapped
        _row("chess player", "Chess player."),       # strict miss, synonym hit
        _row("army", "Soldier"),                     # gold merge
    ])
    s = score_frame(df)

    assert s["correct_strict"].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert s["correct_synonym"].tolist() == [1, 0, 0, 1, 0, 0, 0, 1, 1]
    assert s["correct_category"].tolist() == [1, 1, 0, 1, 1, 0, 0, 1, 1]
    assert s["excluded"].tolist() == [0, 0, 0, 0, 0, 1, 0, 0, 0]
    assert s["pred_unmapped"].tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0]
    assert s["pred_gender_marking"].tolist() == ["", "", "", "male", "", "", "", "", ""]
    assert s.loc[4, "pred_canonical"] == "artist"
    assert s.loc[4, "pred_category"] == "visual arts"
    assert s.loc[8, "gold_canonical"] == "soldier"


def test_unknown_gold_label_raises():
    with pytest.raises(ValueError, match="missing from"):
        score_frame(pd.DataFrame([_row("wizard", "Wizard")]))


def test_unmapped_report_counts_and_sorts():
    s = score_frame(pd.DataFrame([
        _row("marine biologist", "Diver"),
        _row("marine biologist", "Diver"),
        _row("teacher", "Student"),
        _row("pensioner", "Retiree"),   # excluded rows do not appear
    ]))
    report = unmapped_report({"m": s})
    assert report["response_norm"].tolist() == ["diver", "student"]
    assert report["count"].tolist() == [2, 1]
    assert report.loc[0, "example_gold_labels"] == "marine biologist"


@pytest.mark.skipif(not RESULT_FILES, reason="baseline results not present")
@pytest.mark.parametrize("path", RESULT_FILES, ids=lambda p: p.parent.name)
def test_strict_tier_reproduces_existing_correct_column(path):
    df = pd.read_csv(path)
    s = score_frame(df)
    assert (s["correct_strict"] == df["correct"]).all()
    # original columns are untouched
    pd.testing.assert_frame_equal(s[df.columns], df)


def test_strict_normalize_matches_runner_rule():
    assert strict_normalize(" Doctor ") == "doctor"
    assert strict_normalize("Doctor.") == "doctor."
