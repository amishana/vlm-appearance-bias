"""Compute baseline evaluation metrics from the scored CSVs.

Usage:
    python -m eval.metrics --scored eval/scored --out eval/results

Reads ``<scored>/<model>/val_scored.csv`` for every model, drops excluded
rows, and writes one CSV per metric family to ``<out>``:

    overall.csv               accuracy per tier and macro F1 per model
    per_class.csv             precision / recall / F1 per canonical class
    attribute_accuracy.csv    accuracy by attribute value with bootstrap CIs
    attribute_gaps.csv        max-minus-min accuracy gap per attribute type
    consistency.csv           distinct predictions per counterfactual set
    flip_rates.csv            prediction flip rate between attribute values
    mcnemar.csv               paired significance tests on correctness
    stereotype_logodds.csv    attribute skew of each frequently predicted class
    gendered_words.csv        gendered word use by subject gender
    unmapped_by_attribute.csv unmapped-response rate by attribute value
    occupation_by_attribute.csv accuracy per occupation x attribute value

All accuracy metrics use correct_synonym unless the column name says otherwise.
Confidence intervals bootstrap over counterfactual sets, not rows, because the
images within a set share a scene and are not independent.
"""

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = "correct_synonym"
N_BOOT = 1000
SEED = 566
ATTR_ORDER = {
    "gender": ["male", "female"],
    "race": ["Asian", "Black", "Indian", "Latino", "Middle Eastern", "White"],
    "physical": ["young", "old", "skinny", "obese", "tattooed"],
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_scored(scored_dir):
    frames = []
    for path in sorted(Path(scored_dir).glob("*/val_scored.csv")):
        df = pd.read_csv(path, keep_default_na=False)
        df["model"] = path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no */val_scored.csv under {scored_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df[df["excluded"] == 0].copy()


def attribute_long(df):
    """One row per (image, attribute). Each image contributes a1 and a2."""
    a1 = df[["model", "image_id", "counterfactual_set", "dataset_segment",
             "a1_type", "a1", MAIN, "correct_category", "pred_unmapped",
             "gold_canonical", "pred_canonical", "pred_gender_marking"]]
    a1 = a1.rename(columns={"a1_type": "attr_type", "a1": "attr_value"})
    a2 = df[["model", "image_id", "counterfactual_set", "dataset_segment",
             "a2_type", "a2", MAIN, "correct_category", "pred_unmapped",
             "gold_canonical", "pred_canonical", "pred_gender_marking"]]
    a2 = a2.rename(columns={"a2_type": "attr_type", "a2": "attr_value"})
    return pd.concat([a1, a2], ignore_index=True)


# ---------------------------------------------------------------------------
# capability
# ---------------------------------------------------------------------------

def per_class(df):
    rows = []
    for model, g in df.groupby("model"):
        gold = g["gold_canonical"]
        pred = g["pred_canonical"]
        for cls in sorted(gold.unique()):
            tp = int(((gold == cls) & (pred == cls)).sum())
            fp = int(((gold != cls) & (pred == cls)).sum())
            fn = int(((gold == cls) & (pred != cls)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            rows.append({"model": model, "class": cls, "support": tp + fn,
                         "precision": p, "recall": r, "f1": f1})
    return pd.DataFrame(rows)


def overall(df, pc):
    rows = []
    for model, g in df.groupby("model"):
        rows.append({
            "model": model,
            "n": len(g),
            "acc_strict": g["correct_strict"].mean(),
            "acc_synonym": g[MAIN].mean(),
            "acc_category": g["correct_category"].mean(),
            "macro_f1_synonym": pc.loc[pc["model"] == model, "f1"].mean(),
            "unmapped_rate": g["pred_unmapped"].mean(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# bias by attribute
# ---------------------------------------------------------------------------

def _bootstrap_ci(correct_by_set, n_by_set, rng):
    """CI for sum(correct)/sum(n) resampling sets with replacement."""
    k = len(n_by_set)
    idx = rng.integers(0, k, size=(N_BOOT, k))
    num = correct_by_set[idx].sum(axis=1)
    den = n_by_set[idx].sum(axis=1)
    acc = num / den
    return float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))


def attribute_accuracy(long):
    rng = np.random.default_rng(SEED)
    rows = []
    keys = ["model", "dataset_segment", "attr_type", "attr_value"]
    for (model, seg, atype, aval), g in long.groupby(keys):
        per_set = g.groupby("counterfactual_set")[MAIN].agg(["sum", "count"])
        lo, hi = _bootstrap_ci(per_set["sum"].to_numpy(float),
                               per_set["count"].to_numpy(float), rng)
        rows.append({
            "model": model, "dataset_segment": seg, "attr_type": atype,
            "attr_value": aval, "n": len(g),
            "acc_synonym": g[MAIN].mean(), "ci_low": lo, "ci_high": hi,
            "acc_category": g["correct_category"].mean(),
        })
    out = pd.DataFrame(rows)
    # pooled across segments (an attribute type can appear in two segments)
    pooled = []
    for (model, atype, aval), g in long.groupby(["model", "attr_type", "attr_value"]):
        per_set = g.groupby("counterfactual_set")[MAIN].agg(["sum", "count"])
        lo, hi = _bootstrap_ci(per_set["sum"].to_numpy(float),
                               per_set["count"].to_numpy(float), rng)
        pooled.append({
            "model": model, "dataset_segment": "all", "attr_type": atype,
            "attr_value": aval, "n": len(g),
            "acc_synonym": g[MAIN].mean(), "ci_low": lo, "ci_high": hi,
            "acc_category": g["correct_category"].mean(),
        })
    return pd.concat([out, pd.DataFrame(pooled)], ignore_index=True)


def attribute_gaps(acc):
    rows = []
    for (model, seg, atype), g in acc.groupby(["model", "dataset_segment", "attr_type"]):
        hi = g.loc[g["acc_synonym"].idxmax()]
        lo = g.loc[g["acc_synonym"].idxmin()]
        rows.append({
            "model": model, "dataset_segment": seg, "attr_type": atype,
            "best_value": hi["attr_value"], "best_acc": hi["acc_synonym"],
            "worst_value": lo["attr_value"], "worst_acc": lo["acc_synonym"],
            "gap": hi["acc_synonym"] - lo["acc_synonym"],
            "gap_category_tier": g["acc_category"].max() - g["acc_category"].min(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# counterfactual consistency
# ---------------------------------------------------------------------------

def consistency(df):
    rows = []
    for model, g in df.groupby("model"):
        n_distinct = g.groupby("counterfactual_set")["pred_canonical"].nunique()
        binned = n_distinct.clip(upper=4)
        total = len(binned)
        for k in [1, 2, 3, 4]:
            rows.append({"model": model, "n_distinct": "4+" if k == 4 else str(k),
                         "n_sets": int((binned == k).sum()),
                         "share_of_sets": float((binned == k).mean())})
        rows.append({"model": model, "n_distinct": "mean",
                     "n_sets": total, "share_of_sets": float(n_distinct.mean())})
    return pd.DataFrame(rows)


def _matched_pairs(df, vary, hold):
    """Pairs of images in the same set that differ in `vary` and share `hold`."""
    cols = ["model", "counterfactual_set", hold, vary, "pred_canonical", MAIN]
    left = df[cols].rename(columns={vary: "value_a", "pred_canonical": "pred_a",
                                    MAIN: "correct_a"})
    right = df[cols].rename(columns={vary: "value_b", "pred_canonical": "pred_b",
                                     MAIN: "correct_b"})
    pairs = left.merge(right, on=["model", "counterfactual_set", hold])
    return pairs[pairs["value_a"] < pairs["value_b"]]


def flip_and_mcnemar(df):
    """Flip rate and McNemar test for every pair of values of one attribute,
    holding the other attribute in the set fixed."""
    flip_rows, mc_rows = [], []
    for seg, g in df.groupby("dataset_segment"):
        for vary, hold in [("a1", "a2"), ("a2", "a1")]:
            atype = g[vary + "_type"].iloc[0]
            pairs = _matched_pairs(g, vary, hold)
            for (model, va, vb), p in pairs.groupby(["model", "value_a", "value_b"]):
                flip = float((p["pred_a"] != p["pred_b"]).mean())
                b = int(((p["correct_a"] == 1) & (p["correct_b"] == 0)).sum())
                c = int(((p["correct_a"] == 0) & (p["correct_b"] == 1)).sum())
                flip_rows.append({"model": model, "dataset_segment": seg,
                                  "attr_type": atype, "value_a": va, "value_b": vb,
                                  "n_pairs": len(p), "flip_rate": flip})
                if b + c:
                    stat = (abs(b - c) - 1) ** 2 / (b + c)
                    pval = math.erfc(math.sqrt(stat / 2))
                else:
                    stat, pval = 0.0, 1.0
                mc_rows.append({"model": model, "dataset_segment": seg,
                                "attr_type": atype, "value_a": va, "value_b": vb,
                                "n_pairs": len(p),
                                "acc_a": float(p["correct_a"].mean()),
                                "acc_b": float(p["correct_b"].mean()),
                                "a_right_b_wrong": b, "a_wrong_b_right": c,
                                "chi2": stat, "p_value": pval})
    flips = pd.DataFrame(flip_rows)
    # pooled flip rate per value pair across segments
    pooled = (flips.assign(flips=flips["flip_rate"] * flips["n_pairs"])
              .groupby(["model", "attr_type", "value_a", "value_b"], as_index=False)
              .agg(n_pairs=("n_pairs", "sum"), flips=("flips", "sum")))
    pooled["flip_rate"] = pooled["flips"] / pooled["n_pairs"]
    pooled["dataset_segment"] = "all"
    flips = pd.concat([flips, pooled.drop(columns="flips")], ignore_index=True)
    return flips, pd.DataFrame(mc_rows)


# ---------------------------------------------------------------------------
# stereotype direction
# ---------------------------------------------------------------------------

def stereotype_logodds(long, top_k=15):
    rows = []
    for model, g in long.groupby("model"):
        for atype, ga in g.groupby("attr_type"):
            n_total = len(ga)
            top = ga["pred_canonical"].value_counts().head(top_k).index
            for cls in top:
                is_cls = ga["pred_canonical"] == cls
                for val in ATTR_ORDER[atype]:
                    is_val = ga["attr_value"] == val
                    a = float((is_cls & is_val).sum()) + 0.5
                    b = float((is_cls & ~is_val).sum()) + 0.5
                    c = float((~is_cls & is_val).sum()) + 0.5
                    d = float((~is_cls & ~is_val).sum()) + 0.5
                    lor = math.log((a * d) / (b * c))
                    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                    rows.append({
                        "model": model, "attr_type": atype, "pred_class": cls,
                        "attr_value": val,
                        "n_pred": int(is_cls.sum()),
                        "n_pred_in_group": int((is_cls & is_val).sum()),
                        "share_in_group": float((is_cls & is_val).sum() / max(is_cls.sum(), 1)),
                        "base_rate": float(is_val.mean()),
                        "log_odds_ratio": lor, "se": se,
                        "ci_low": lor - 1.96 * se, "ci_high": lor + 1.96 * se,
                    })
    return pd.DataFrame(rows)


def gendered_words(long):
    g = long[long["attr_type"] == "gender"]
    rows = []
    for (model, subj), gg in g.groupby(["model", "attr_value"]):
        n = len(gg)
        rows.append({
            "model": model, "subject_gender": subj, "n": n,
            "rate_male_word": float((gg["pred_gender_marking"] == "male").mean()),
            "rate_female_word": float((gg["pred_gender_marking"] == "female").mean()),
        })
    return pd.DataFrame(rows)


def unmapped_by_attribute(long):
    out = (long.groupby(["model", "attr_type", "attr_value"], as_index=False)
           .agg(n=("pred_unmapped", "size"), unmapped_rate=("pred_unmapped", "mean")))
    return out


def occupation_by_attribute(long):
    acc = (long.groupby(["model", "gold_canonical", "attr_type", "attr_value"], as_index=False)
           .agg(n=(MAIN, "size"), acc=(MAIN, "mean")))
    occ_mean = (long.groupby(["model", "gold_canonical", "attr_type"])[MAIN]
                .mean().rename("occupation_mean").reset_index())
    acc = acc.merge(occ_mean, on=["model", "gold_canonical", "attr_type"])
    acc["deviation"] = acc["acc"] - acc["occupation_mean"]
    spread = (acc.groupby(["model", "gold_canonical", "attr_type"])["acc"]
              .agg(lambda s: s.max() - s.min()).rename("spread").reset_index())
    return acc.merge(spread, on=["model", "gold_canonical", "attr_type"])


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scored", type=Path, default=Path("eval/scored"))
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_scored(args.scored)
    long = attribute_long(df)

    pc = per_class(df)
    ov = overall(df, pc)
    acc = attribute_accuracy(long)
    gaps = attribute_gaps(acc)
    cons = consistency(df)
    flips, mc = flip_and_mcnemar(df)
    lor = stereotype_logodds(long)
    gw = gendered_words(long)
    un = unmapped_by_attribute(long)
    oba = occupation_by_attribute(long)

    outputs = {
        "overall.csv": ov, "per_class.csv": pc,
        "attribute_accuracy.csv": acc, "attribute_gaps.csv": gaps,
        "consistency.csv": cons, "flip_rates.csv": flips, "mcnemar.csv": mc,
        "stereotype_logodds.csv": lor, "gendered_words.csv": gw,
        "unmapped_by_attribute.csv": un, "occupation_by_attribute.csv": oba,
    }
    for name, frame in outputs.items():
        frame.to_csv(args.out / name, index=False)

    print(ov.round(4).to_string(index=False))
    print()
    print(gaps[gaps["dataset_segment"] == "all"]
          .round(3).to_string(index=False))
    print(f"\nwrote {len(outputs)} files to {args.out}")


if __name__ == "__main__":
    main()
