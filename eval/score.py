"""Score baseline prediction CSVs with tiered label matching.

Usage:
    python -m eval.score baseline/results/*/val_predictions.csv --out eval/scored

For each input file ``<model>/<split>_predictions.csv`` this writes
``<out>/<model>/<split>_scored.csv`` with every original column plus:

    response_norm         normalized prediction text
    pred_canonical        canonical class, or response_norm if unmapped
    pred_category         category, or "" if unmapped
    pred_gender_marking   "male" / "female" / ""
    pred_unmapped         1 if the prediction matched neither a class nor a category
    gold_canonical        canonical class of occupation_raw
    gold_category         category of occupation_raw
    excluded              1 if occupation_raw is in EXCLUDED
    correct_strict        strip().casefold() exact match (reproduces `correct`)
    correct_synonym       pred_canonical == gold_canonical
    correct_category      pred_category == gold_category

It also writes ``<out>/unmapped_report.csv`` and prints a per-model summary.
Excluded rows are kept in the scored files but skipped in the summary.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from eval.occupation_map import CATEGORY, EXCLUDED, GOLD_TO_CANONICAL, lookup

_BBOX = re.compile(r"\[\[.*?\]\]")
_ARTICLE = re.compile(r"^(a|an|the)\s+")
_WS = re.compile(r"\s+")

SCORE_COLUMNS = [
    "response_norm",
    "pred_canonical",
    "pred_category",
    "pred_gender_marking",
    "pred_unmapped",
    "gold_canonical",
    "gold_category",
    "excluded",
    "correct_strict",
    "correct_synonym",
    "correct_category",
]


def strict_normalize(text):
    """The rule the baseline runners use for `correct`."""
    return str(text).strip().casefold()


def normalize(text):
    """Deterministic cleanup applied to predictions and gold labels."""
    text = str(text).casefold().strip()
    text = _BBOX.sub("", text)
    text = text.replace("-", " ").replace("/", " ")
    text = _WS.sub(" ", text).strip()
    text = text.rstrip(".!,").strip()
    text = _ARTICLE.sub("", text)
    return text


def _gold_info(gold_norm):
    if gold_norm in EXCLUDED:
        return "", "", 1
    canonical = GOLD_TO_CANONICAL[gold_norm]
    return canonical, CATEGORY[canonical], 0


def score_frame(df):
    """Return a copy of ``df`` with the SCORE_COLUMNS appended."""
    out = df.copy()

    out["response_norm"] = out["model_response"].map(normalize)
    gold_norm = out["occupation_raw"].map(normalize)

    unknown = sorted(set(gold_norm) - set(GOLD_TO_CANONICAL) - set(EXCLUDED))
    if unknown:
        raise ValueError(
            f"gold labels missing from eval/occupation_map.py: {unknown}"
        )

    gold = gold_norm.map(_gold_info)
    out["gold_canonical"] = gold.str[0]
    out["gold_category"] = gold.str[1]
    out["excluded"] = gold.str[2]

    pred = out["response_norm"].map(lookup)
    canonical = pred.str[0]
    out["pred_canonical"] = canonical.where(canonical.notna(), out["response_norm"])
    out["pred_category"] = pred.str[1].fillna("")
    out["pred_gender_marking"] = pred.str[2].fillna("")
    out["pred_unmapped"] = pred.str[3].astype(int)

    out["correct_strict"] = (
        out["model_response"].map(strict_normalize)
        == out["occupation_raw"].map(strict_normalize)
    ).astype(int)
    out["correct_synonym"] = (
        canonical.notna() & (canonical == out["gold_canonical"])
    ).astype(int)
    out["correct_category"] = (
        (out["pred_category"] != "")
        & (out["pred_category"] == out["gold_category"])
    ).astype(int)

    return out


def summarize(scored):
    kept = scored[scored["excluded"] == 0]
    return {
        "rows": len(scored),
        "excluded_rows": int(scored["excluded"].sum()),
        "unmapped_rate": round(kept["pred_unmapped"].mean(), 4),
        "acc_strict": round(kept["correct_strict"].mean(), 4),
        "acc_synonym": round(kept["correct_synonym"].mean(), 4),
        "acc_category": round(kept["correct_category"].mean(), 4),
    }


def unmapped_report(scored_by_model):
    frames = []
    for model, scored in scored_by_model.items():
        rows = scored[(scored["pred_unmapped"] == 1) & (scored["excluded"] == 0)]
        if rows.empty:
            continue
        grouped = rows.groupby("response_norm")
        frames.append(pd.DataFrame({
            "response_norm": grouped.size().index,
            "model": model,
            "count": grouped.size().values,
            "example_gold_labels": grouped["occupation_raw"].agg(
                lambda s: "; ".join(s.value_counts().head(3).index)
            ).values,
        }))
    if not frames:
        return pd.DataFrame(
            columns=["response_norm", "model", "count", "example_gold_labels"]
        )
    report = pd.concat(frames, ignore_index=True)
    return report.sort_values(
        ["count", "response_norm", "model"], ascending=[False, True, True]
    ).reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path("eval/scored"))
    args = parser.parse_args(argv)

    scored_by_model = {}
    summaries = {}
    for path in args.predictions:
        model = path.parent.name
        split = path.stem.replace("_predictions", "")
        scored = score_frame(pd.read_csv(path))

        mismatch = int((scored["correct_strict"] != scored["correct"]).sum())
        if mismatch:
            raise AssertionError(
                f"{path}: correct_strict disagrees with `correct` on {mismatch} rows"
            )

        out_path = args.out / model / f"{split}_scored.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_path, index=False)

        scored_by_model[model] = scored
        summaries[model] = summarize(scored)

    report = unmapped_report(scored_by_model)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "unmapped_report.csv", index=False)

    summary = pd.DataFrame(summaries).T
    print(summary.to_string())
    print(f"\nunmapped report: {args.out / 'unmapped_report.csv'} "
          f"({len(report)} distinct response/model pairs)")
    print("top unmapped responses:")
    top = (report.groupby("response_norm")["count"].sum()
           .sort_values(ascending=False).head(15))
    print(top.to_string())


if __name__ == "__main__":
    main()
