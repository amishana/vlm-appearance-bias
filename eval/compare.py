"""Paired comparison of mitigation methods against a baseline.

Usage:
    python -m eval.compare --scored projection/scored --baseline qwen2_5_vl_local \
        --methods qwen2_5_vl_leace_all --out projection/eval_results/compare

Reads ``<scored>/<model>/val_scored.csv`` (from eval.score) for the baseline
and each method, keeps non-excluded images present in both, and compares them
on the same images:

    accuracy, accuracy_category   share correct (synonym / category tier)
    macro_f1                      synonym-tier macro F1 (no CI)
    attribute_gap                 max minus min accuracy across values of an
                                  attribute type (pooled over segments)
    flip_rate                     share of matched pairs (same set, same other
                                  attribute, different value of this one) whose
                                  predictions differ
    consistent_sets               share of counterfactual sets with one
                                  distinct prediction
    mean_distinct                 mean number of distinct predictions per set
    changed_predictions           share of images whose prediction changed
                                  (method vs baseline; no baseline value)

Each delta (method minus baseline) gets a 95% paired bootstrap CI that
resamples counterfactual sets, the same sets for both models, because images
in a set share a scene. Accuracy also gets an exact McNemar test on the
paired correctness. Writes compare.csv and compare.md to --out.
"""

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import ATTR_ORDER, MAIN, per_class

N_BOOT = 2000
SEED = 566


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load(scored, model):
    path = Path(scored) / model / "val_scored.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} (run eval.score on this model first)")
    df = pd.read_csv(path, keep_default_na=False)
    df = df[df["excluded"] == 0].copy()
    df["model"] = model
    return df.set_index("image_id")


def paired(base, meth):
    common = base.index.intersection(meth.index)
    missing = len(base) - len(common)
    if missing:
        print(f"note: {missing:,} baseline images have no prediction from "
              f"{meth['model'].iloc[0]}; comparing on {len(common):,} shared images")
    return base.loc[common].sort_index(), meth.loc[common].sort_index()


# ---------------------------------------------------------------------------
# set-level sums: every metric below is a ratio of sums over sets, so the
# bootstrap is a weighted sum with resampled set counts
# ---------------------------------------------------------------------------

def set_index(df):
    sets = np.array(sorted(df["counterfactual_set"].unique()))
    return sets, pd.Index(sets).get_indexer(df["counterfactual_set"])


def sums_accuracy(df, s_idx, n_sets, col):
    num = np.bincount(s_idx, weights=df[col].to_numpy(float), minlength=n_sets)
    den = np.bincount(s_idx, minlength=n_sets).astype(float)
    return num, den


def sums_by_value(df, s_idx, n_sets, atype):
    """[sets x values] correct counts and image counts for one attribute type."""
    values = ATTR_ORDER[atype]
    num = np.zeros((n_sets, len(values)))
    den = np.zeros((n_sets, len(values)))
    corr = df[MAIN].to_numpy(float)
    for slot in ("a1", "a2"):
        is_type = (df[f"{slot}_type"] == atype).to_numpy()
        vi = pd.Index(values).get_indexer(df[slot])
        ok = is_type & (vi >= 0)
        np.add.at(num, (s_idx[ok], vi[ok]), corr[ok])
        np.add.at(den, (s_idx[ok], vi[ok]), 1)
    return num, den


def sums_flips(df, sets, atype):
    """Per set: number of matched pairs varying `atype`, and how many flip."""
    pos = {s: i for i, s in enumerate(sets)}
    pairs = np.zeros(len(sets))
    flips = np.zeros(len(sets))
    cols = ["counterfactual_set", "a1_type", "a2_type", "a1", "a2", "pred_canonical"]
    for (s, t1, t2), g in df[cols].groupby(["counterfactual_set", "a1_type", "a2_type"]):
        for vary, hold, vtype in (("a1", "a2", t1), ("a2", "a1", t2)):
            if vtype != atype:
                continue
            for _, h in g.groupby(hold):
                rows = list(zip(h[vary], h["pred_canonical"]))
                for (va, pa), (vb, pb) in combinations(rows, 2):
                    if va != vb:
                        pairs[pos[s]] += 1
                        flips[pos[s]] += pa != pb
    return flips, pairs


def sums_consistency(df, s_idx, n_sets, sets):
    n_distinct = df.groupby("counterfactual_set")["pred_canonical"].nunique().reindex(sets).to_numpy(float)
    return (n_distinct == 1).astype(float), n_distinct, np.ones(n_sets)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_weights(n_sets, rng):
    idx = rng.integers(0, n_sets, size=(N_BOOT, n_sets))
    return np.stack([np.bincount(r, minlength=n_sets) for r in idx]).astype(float)


def ratio(W, num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W @ num) / (W @ den)


def gap(W, num, den):
    acc = ratio(W, num, den)                        # [B, values] or [values]
    return np.nanmax(acc, axis=-1) - np.nanmin(acc, axis=-1)


def ci(samples):
    s = samples[np.isfinite(samples)]
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value (binomial on discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# ---------------------------------------------------------------------------

def compare_pair(base, meth, rng):
    sets, s_idx = set_index(base)
    n = len(sets)
    W = boot_weights(n, rng)
    ones = np.ones((1, n))
    rows = []

    def add(metric, attr, fb, fm, args_b, args_m, extra=None):
        pb, pm = float(fb(ones, *args_b)[0]), float(fm(ones, *args_m)[0])
        lo, hi = ci(fm(W, *args_m) - fb(W, *args_b))
        rows.append({"metric": metric, "attr_type": attr, "baseline": pb, "method": pm,
                     "delta": pm - pb, "ci_low": lo, "ci_high": hi, **(extra or {})})

    for col, name in ((MAIN, "accuracy"), ("correct_category", "accuracy_category")):
        nb, db = sums_accuracy(base, s_idx, n, col)
        nm, dm = sums_accuracy(meth, s_idx, n, col)
        extra = {}
        if col == MAIN:
            b = int(((base[MAIN] == 1) & (meth[MAIN] == 0)).sum())
            c = int(((base[MAIN] == 0) & (meth[MAIN] == 1)).sum())
            extra = {"n_base_right_method_wrong": b, "n_base_wrong_method_right": c,
                     "mcnemar_p": mcnemar_exact(b, c)}
        add(name, "", ratio, ratio, (nb, db), (nm, dm), extra)

    f1b = per_class(base.reset_index())["f1"].mean()
    f1m = per_class(meth.reset_index())["f1"].mean()
    rows.append({"metric": "macro_f1", "attr_type": "", "baseline": f1b, "method": f1m,
                 "delta": f1m - f1b})

    for atype in ATTR_ORDER:
        nb, db = sums_by_value(base, s_idx, n, atype)
        nm, dm = sums_by_value(meth, s_idx, n, atype)
        if db.sum() == 0:
            continue
        add("attribute_gap", atype, gap, gap, (nb, db), (nm, dm))
        fb_, pb_ = sums_flips(base, sets, atype)
        fm_, pm_ = sums_flips(meth, sets, atype)
        add("flip_rate", atype, ratio, ratio, (fb_, pb_), (fm_, pm_),
            {"n_pairs": int(pb_.sum())})

    cb, mb, ob = sums_consistency(base, s_idx, n, sets)
    cm, mm, om = sums_consistency(meth, s_idx, n, sets)
    add("consistent_sets", "", ratio, ratio, (cb, ob), (cm, om))
    add("mean_distinct", "", ratio, ratio, (mb, ob), (mm, om))

    changed = (base["pred_canonical"] != meth["pred_canonical"]).to_numpy(float)
    nc = np.bincount(s_idx, weights=changed, minlength=n)
    dc = np.bincount(s_idx, minlength=n).astype(float)
    lo, hi = ci(ratio(W, nc, dc))
    rows.append({"metric": "changed_predictions", "attr_type": "", "method": changed.mean(),
                 "delta": changed.mean(), "ci_low": lo, "ci_high": hi})
    return rows


def to_markdown(table, baseline):
    out = [f"# Mitigation vs `{baseline}`\n",
           "Delta = method minus baseline on the same images; 95% CI from a paired bootstrap "
           "over counterfactual sets. Lower is better for gap, flip rate and mean distinct; "
           "higher is better for accuracy, macro F1 and consistent sets.\n"]
    for method, g in table.groupby("method_name", sort=False):
        out += [f"\n## {method}\n",
                "| metric | attribute | baseline | method | delta | 95% CI |",
                "|---|---|---|---|---|---|"]
        for _, r in g.iterrows():
            f = (lambda x: f"{x:.3f}") if r["metric"] in ("mean_distinct",) else (lambda x: f"{100 * x:.1f}%")
            d = (lambda x: f"{x:+.3f}") if r["metric"] in ("mean_distinct",) else (lambda x: f"{100 * x:+.1f} pp")
            base_v = "" if pd.isna(r["baseline"]) else f(r["baseline"])
            ci_s = "" if pd.isna(r.get("ci_low")) else f"[{d(r['ci_low'])}, {d(r['ci_high'])}]"
            out.append(f"| {r['metric']} | {r['attr_type']} | {base_v} | {f(r['method'])} | "
                       f"{d(r['delta'])} | {ci_s} |")
        acc = g[g["metric"] == "accuracy"].iloc[0]
        out.append(f"\nMcNemar (accuracy): baseline right / method wrong = "
                   f"{int(acc['n_base_right_method_wrong']):,}, baseline wrong / method right = "
                   f"{int(acc['n_base_wrong_method_right']):,}, exact p = {acc['mcnemar_p']:.3g}.")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--baseline", required=True, help="model folder name of the baseline")
    parser.add_argument("--methods", nargs="+", required=True, help="model folder names to compare")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(SEED)
    base_all = load(args.scored, args.baseline)
    tables = []
    for m in args.methods:
        base, meth = paired(base_all, load(args.scored, m))
        rows = compare_pair(base, meth, rng)
        t = pd.DataFrame(rows)
        t.insert(0, "method_name", m)
        t.insert(1, "baseline_name", args.baseline)
        t.insert(2, "n_images", len(base))
        tables.append(t)
        print(f"{m}: compared on {len(base):,} images")

    table = pd.concat(tables, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "compare.csv", index=False)
    (args.out / "compare.md").write_text(to_markdown(table, args.baseline))
    show = table[["method_name", "metric", "attr_type", "baseline", "method", "delta", "ci_low", "ci_high"]]
    print(show.round(4).to_string(index=False))
    print(f"\nwrote {args.out / 'compare.csv'} and compare.md")


if __name__ == "__main__":
    main()
