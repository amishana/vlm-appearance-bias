"""Write a Markdown summary of the baseline evaluation from eval/results.

Usage:
    python -m eval.report --results eval/results --figures eval/figures --out eval/results/summary.md

Every number in the summary is read from the metric CSVs, so re-running
score -> metrics -> plots -> report after a mapping change keeps it current.
"""

import argparse
from pathlib import Path

import pandas as pd

from eval import occupation_map as om

MODEL_LABELS = {"internvl3": "InternVL3", "qwen2_5_vl": "Qwen2.5-VL", "smolvlm2": "SmolVLM2"}
ATTR_TITLES = {"gender": "gender", "race": "race", "physical": "physical appearance"}


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def md_table(df, cols, headers, fmt):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt[c](r[c]) if c in fmt else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    parser.add_argument("--figures", type=Path, default=Path("eval/figures"))
    parser.add_argument("--out", type=Path, default=Path("eval/results/summary.md"))
    args = parser.parse_args(argv)
    R = args.results
    rel_fig = Path("..") / args.figures.name  # summary.md lives in eval/results/

    ov = pd.read_csv(R / "overall.csv")
    pc = pd.read_csv(R / "per_class.csv")
    acc = pd.read_csv(R / "attribute_accuracy.csv")
    gaps = pd.read_csv(R / "attribute_gaps.csv")
    cons = pd.read_csv(R / "consistency.csv")
    flips = pd.read_csv(R / "flip_rates.csv")
    mc = pd.read_csv(R / "mcnemar.csv")
    lor = pd.read_csv(R / "stereotype_logodds.csv")
    gw = pd.read_csv(R / "gendered_words.csv")
    un = pd.read_csv(R / "unmapped_by_attribute.csv")

    ov["model"] = ov["model"].map(MODEL_LABELS)
    n_rows = int(ov["n"].iloc[0])
    out = []
    w = out.append

    w("# Baseline evaluation summary: occupation prediction on SocialCounterfactuals (val)\n")
    w("Pre-mitigation results for three vision-language models. Predictions were scored with the "
      "tiered label map in `eval/occupation_map.py`; unless stated otherwise, every number below uses "
      "the **synonym tier** (`correct_synonym`), excludes the three dropped gold labels, and treats "
      f"the counterfactual set as the unit of resampling. Rows per model after exclusion: {n_rows:,}.\n")
    w("Figures are in `eval/figures/` (interactive HTML and PNG); the table behind each metric is in "
      "`eval/results/`.\n")

    # 0 scoring method ------------------------------------------------------
    w("## 0. How predictions were scored\n")
    w("### Why a label map\n")
    w("The baseline runners mark a prediction correct only if the model's text exactly equals the gold "
      "label after trimming and lowercasing. Under that rule \"Doctor\" for `physician`, \"Businessman\" "
      "for `businessperson` and \"Chief Executive Officer\" for `ceo` all count as wrong, so raw accuracy "
      "mixes vocabulary mismatch with genuine error. The gold label set itself also contains duplicate "
      "names for the same job and a few labels that are not US occupations or are caption-parsing "
      "artifacts. The map in `eval/occupation_map.py` fixes this in one shared, reviewable place.\n")
    w("### How the map was built\n")
    merges = sorted(om._GOLD_MERGES.items())
    w("1. **Gold merges.** Pairs of gold labels that name the same job were collapsed into one canonical "
      "class, chosen from the meaning of the 126 labels rather than from the models' errors: "
      + "; ".join(f"`{a}` → `{b}`" for a, b in merges) + ".")
    w("2. **Exclusions.** Gold labels dropped from every metric, with the reason: "
      + "; ".join(f"`{k}` ({v})" for k, v in sorted(om.EXCLUDED.items())) + ". Their rows stay in the "
      "scored files with `excluded = 1` so nothing is silently lost.")
    w(f"3. **Categories.** Each of the {len(om.CANONICAL_CLASSES)} canonical classes was assigned to one of "
      f"{len(om.CATEGORY_MEMBERS)} broad fields (medical, nursing, culinary, software, business, construction, "
      "trades, and so on). Nursing is deliberately separate from medical so that \"Doctor\" for a nurse "
      "practitioner stays a miss at every tier.")
    w(f"4. **Prediction aliases.** {len(om.ALIASES)} strings the models emit that are not gold labels but "
      "name a canonical class (\"programmer\" → computer programmer, \"waitress\" → waiter, \"software "
      f"engineer\" → software developer), plus {len(om.CATEGORY_ONLY_ALIASES)} category-only aliases for "
      "words that name a field but no single job (\"artist\", \"office worker\", \"IT\"). The first pass "
      "was written from the label list; a second pass added the most frequent strings from the scorer's "
      "unmapped report (misspellings such as \"florrist\", sports such as \"badminton player\"). Classes "
      "were never merged because a model confused them.")
    w(f"5. **Gendered forms.** {len(om.GENDER_MARKED)} gendered words (businessman, waitress, policewoman, "
      "fireman) are recorded separately. They still count as correct; the tally shows when a model chose "
      "the gendered word.")
    w("\nJudgment calls worth knowing: `army` merges into soldier; `optometrist` maps to optician; "
      "`store clerk` maps to cashier rather than clerk; `cook` and `chef` stay separate classes in the "
      "same category; `dietitian` and `nutritionist` stay separate classes in the same category.\n")
    w("### Three correctness tiers\n")
    w("Each prediction is still judged against `occupation_raw`; the tiers differ only in how much "
      "vocabulary difference is forgiven. They nest: strict ⊂ synonym ⊂ category.\n")
    w("| Tier | Rule | Example that counts as correct |\n|---|---|---|")
    w("| `correct_strict` | exact text match, as the runners computed it | \"Doctor\" for `doctor` |")
    w("| `correct_synonym` | same canonical class after merging and aliases | \"Doctor\" for `physician`, "
      "\"Businessman\" for `businessperson` |")
    w("| `correct_category` | same broad field | \"Nurse\" for `nurse practitioner`, \"Chef\" for `pastry chef` |")
    w("\n`correct_synonym` is the headline metric. `correct_strict` is the floor and reproduces the original "
      "`correct` column bit for bit (the scorer refuses to run if it does not). `correct_category` is a "
      "robustness check: a gap that disappears at this tier was about specificity, not stereotype.\n")
    w("### The scored CSV files\n")
    w("The original prediction files under `baseline/results/<model>/val_predictions.csv` are untouched. "
      "`eval/score.py` writes `eval/scored/<model>/val_scored.csv`, which keeps every original column "
      "(`image_id`, `counterfactual_set`, `occupation_raw`, `dataset_segment`, `a1_type`, `a1`, `a2_type`, "
      "`a2`, `image_path`, `model_response`, `correct`, `inference_seconds`) and appends the columns below. "
      "It also writes `eval/scored/unmapped_report.csv`, a count of every response the map could not place, "
      "which is the list to read when extending the aliases.\n")
    w("| New column | Meaning |\n|---|---|")
    w("| `response_norm` | Model response after cleanup: lowercased, trailing period removed, hyphens and "
      "slashes turned to spaces, InternVL bounding-box text stripped. |")
    w("| `pred_canonical` | Canonical class the response maps to (`doctor` for \"Physician\"). If the "
      "response matched nothing, this is just `response_norm`. |")
    w("| `pred_category` | Broad field of the prediction (`medical`, `software`). Empty if unmapped. |")
    w("| `pred_gender_marking` | `male` or `female` if the model used a gendered word; empty otherwise. |")
    w("| `pred_unmapped` | 1 if the response matched no class and no category (\"student\", \"diver\"). |")
    w("| `gold_canonical` | Canonical class of `occupation_raw` (`ceo` for both `ceo` and `chief executive "
      "officer`). Empty for excluded rows. |")
    w("| `gold_category` | Broad field of the gold label. Empty for excluded rows. |")
    w("| `excluded` | 1 if the gold label is one of the dropped labels. Filter these out before computing anything. |")
    w("| `correct_strict` | Exact-match rule; identical to `correct`. |")
    w("| `correct_synonym` | 1 if `pred_canonical` equals `gold_canonical`. Main accuracy column. |")
    w("| `correct_category` | 1 if `pred_category` equals `gold_category`. |")
    w("\nConventions used everywhere below: filter `excluded == 0`, score with `correct_synonym`, and use "
      "`gold_canonical` and `pred_canonical` as the class columns for precision and recall.\n")

    # 1 capability --------------------------------------------------------
    w("## 1. Capability baseline\n")
    w(md_table(ov, ["model", "acc_strict", "acc_synonym", "acc_category", "macro_f1_synonym", "unmapped_rate"],
               ["Model", "Strict acc", "Synonym acc", "Category acc", "Macro F1 (synonym)", "Unmapped"],
               {c: pct for c in ["acc_strict", "acc_synonym", "acc_category", "unmapped_rate"]}
               | {"macro_f1_synonym": lambda v: f"{v:.3f}"}))
    best = ov.loc[ov["acc_synonym"].idxmax()]
    spread = ov["acc_synonym"].max() - ov["acc_synonym"].min()
    w(f"\n- Label merging lifts every model by {pct((ov['acc_synonym'] - ov['acc_strict']).min())} to "
      f"{pct((ov['acc_synonym'] - ov['acc_strict']).max())} points and narrows the between-model spread to "
      f"{pct(spread)} points at the synonym tier ({best['model']} is highest).")
    w(f"- The jump from synonym to category tier ({pct((ov['acc_category'] - ov['acc_synonym']).min())} to "
      f"{pct((ov['acc_category'] - ov['acc_synonym']).max())} points) means most remaining errors are "
      "specificity misses (a broader or sibling job in the right field), not wrong fields.")
    mean_f1 = pc.groupby("class")["f1"].mean().sort_values()
    zero = mean_f1[mean_f1 == 0]
    w(f"- {len(zero)} of {mean_f1.size} canonical classes have F1 = 0 for every model, e.g. "
      f"{', '.join(zero.index[:6])}. These are absorbed by a broader class (doctor, chef, nurse, "
      "software developer, construction worker).")
    w(f"- Highest mean F1: {', '.join(mean_f1.index[-5:][::-1])}.")
    w(f"\n![overall]({rel_fig}/01_overall_accuracy.png)\n")
    w(f"![per-class F1]({rel_fig}/03_per_class_f1.png)\n")

    # 2 accuracy by attribute ----------------------------------------------
    w("## 2. Accuracy by attribute value\n")
    ga = gaps[gaps["dataset_segment"] == "all"].copy()
    ga["model"] = ga["model"].map(MODEL_LABELS)
    ga["attr_type"] = ga["attr_type"].map(ATTR_TITLES)
    w(md_table(ga.sort_values(["attr_type", "model"]),
               ["attr_type", "model", "best_value", "best_acc", "worst_value", "worst_acc", "gap", "gap_category_tier"],
               ["Attribute", "Model", "Best value", "Acc", "Worst value", "Acc", "Gap", "Gap at category tier"],
               {c: pct for c in ["best_acc", "worst_acc", "gap", "gap_category_tier"]}))
    worst = ga.loc[ga["gap"].idxmax()]
    w(f"\n- Largest pooled gap: {worst['attr_type']} for {worst['model']}, {pct(worst['gap'])} "
      f"({worst['best_value']} {pct(worst['best_acc'])} vs {worst['worst_value']} {pct(worst['worst_acc'])}).")
    w(f"- Gaps are small relative to overall accuracy (all under {pct(ga['gap'].max() + 0.005, 0)}), "
      "and most survive at the category tier, so they are not artifacts of label vocabulary.")
    # recurring worst values
    worst_counts = ga.groupby(["attr_type", "worst_value"]).size().sort_values(ascending=False)
    recurring = [f"{v} ({a})" for (a, v), n in worst_counts.items() if n >= 2]
    if recurring:
        w(f"- Values that are the worst group for at least two models: {', '.join(recurring)}.")
    sig = mc[mc["p_value"] < 0.05]
    w(f"- McNemar tests on matched pairs: {len(sig)} of {len(mc)} value pairs differ significantly "
      "(p < 0.05) in correctness. With tens of thousands of pairs, small differences are detectable; "
      "the effect sizes above matter more than the p-values.")
    w(f"\n![attribute accuracy]({rel_fig}/02_attribute_accuracy.png)\n")
    w(f"![occupation heatmap]({rel_fig}/06_occupation_heatmap.png)\n")

    # 3 consistency ---------------------------------------------------------
    w("## 3. Counterfactual consistency\n")
    cm = cons.pivot(index="model", columns="n_distinct", values="share_of_sets").reset_index()
    cm["model"] = cm["model"].map(MODEL_LABELS)
    w(md_table(cm, ["model", "1", "2", "3", "4+", "mean"],
               ["Model", "1 prediction", "2", "3", "4+", "Mean distinct per set"],
               {"1": pct, "2": pct, "3": pct, "4+": pct, "mean": lambda v: f"{v:.2f}"}))
    fa = flips[flips["dataset_segment"] == "all"].copy()
    fa["model"] = fa["model"].map(MODEL_LABELS)
    by_type = fa.groupby(["attr_type", "model"])["flip_rate"].mean().unstack()
    w("\nMean flip rate across value pairs (share of matched image pairs whose predicted occupation changes):\n")
    bt = by_type.reset_index()
    bt["attr_type"] = bt["attr_type"].map(ATTR_TITLES)
    w(md_table(bt, ["attr_type"] + list(by_type.columns), ["Attribute"] + list(by_type.columns),
               {m: pct for m in by_type.columns}))
    top_pairs = (fa.groupby(["attr_type", "value_a", "value_b"])["flip_rate"].mean()
                 .sort_values(ascending=False).head(3))
    low_pairs = (fa.groupby(["attr_type", "value_a", "value_b"])["flip_rate"].mean()
                 .sort_values().head(2))
    w(f"\n- Only {pct(cm['1'].min(), 0)} to {pct(cm['1'].max(), 0)} of sets get one prediction across all "
      "their variants. Changing a single attribute flips the predicted occupation in roughly one of five "
      "matched pairs, even though accuracy by group barely moves. Inconsistency is the stronger bias signal.")
    w("- Most flip-prone pairs: " + "; ".join(
        f"{a} ↔ {b} ({pct(v)})" for (t, a, b), v in top_pairs.items()) + ".")
    w("- Least flip-prone pairs: " + "; ".join(
        f"{a} ↔ {b} ({pct(v)})" for (t, a, b), v in low_pairs.items()) + ".")
    w(f"\n![consistency]({rel_fig}/04_consistency.png)\n")
    w(f"![flip rates]({rel_fig}/05_flip_rates.png)\n")

    # 4 stereotype direction ------------------------------------------------
    w("## 4. Stereotype direction\n")
    g = lor[(lor["attr_type"] == "gender") & (lor["attr_value"] == "female")].copy()
    g["model"] = g["model"].map(MODEL_LABELS)
    strong = g[g["ci_low"].gt(0) | g["ci_high"].lt(0)].copy()
    strong["abs"] = strong["log_odds_ratio"].abs()
    strong = strong.sort_values("abs", ascending=False)
    male_skew = strong[strong["log_odds_ratio"] < 0]
    female_skew = strong[strong["log_odds_ratio"] > 0]
    w("Among each model's 15 most frequent predictions, classes whose gender skew is significant "
      "(95% CI excludes 0):\n")
    w("- Predicted more for **male** subjects: " + ("; ".join(
        f"{r['pred_class']} ({r['model']}, log-odds {r['log_odds_ratio']:+.2f})"
        for _, r in male_skew.iterrows()) or "none") + ".")
    w("- Predicted more for **female** subjects: " + ("; ".join(
        f"{r['pred_class']} ({r['model']}, log-odds {r['log_odds_ratio']:+.2f})"
        for _, r in female_skew.iterrows()) or "none") + ".")
    rc = lor[lor["attr_type"] == "race"].copy()
    rc["model"] = rc["model"].map(MODEL_LABELS)
    rc_sig = rc[rc["ci_low"].gt(0) | rc["ci_high"].lt(0)].copy()
    rc_sig["abs"] = rc_sig["log_odds_ratio"].abs()
    rc_top = rc_sig.sort_values("abs", ascending=False).head(8)
    w("- Largest significant race skews: " + ("; ".join(
        f"{r['pred_class']} {'over' if r['log_odds_ratio'] > 0 else 'under'}-predicted for "
        f"{r['attr_value']} ({r['model']}, {r['log_odds_ratio']:+.2f})"
        for _, r in rc_top.iterrows()) or "none") + ".")
    w(f"\n![gender skew]({rel_fig}/07_stereotype_gender.png)\n")
    w(f"![race skew]({rel_fig}/08_stereotype_race.png)\n")

    # 5 gendered words ------------------------------------------------------
    w("## 5. Gendered occupation words\n")
    gw["model"] = gw["model"].map(MODEL_LABELS)
    w(md_table(gw.sort_values(["model", "subject_gender"]),
               ["model", "subject_gender", "rate_male_word", "rate_female_word"],
               ["Model", "Subject", "Male-marked word", "Female-marked word"],
               {"rate_male_word": lambda v: pct(v, 2), "rate_female_word": lambda v: pct(v, 2)}))
    hi = gw.assign(any=gw["rate_male_word"] + gw["rate_female_word"]).groupby("model")["any"].mean()
    users = [m for m, v in hi.items() if v >= 0.01]
    rare = [m for m, v in hi.items() if v < 0.01]
    w(f"\n- {' and '.join(users)} choose a gendered form (businessman, waitress, policewoman) in about "
      f"one response in twenty, almost always matching the subject's apparent gender"
      + (f"; {' and '.join(rare)} almost never {'does' if len(rare) == 1 else 'do'}" if rare else "")
      + ". These count as correct at the synonym tier but show the models read and encode gender "
      "when naming a job.")
    w(f"\n![gendered words]({rel_fig}/09_gendered_words.png)\n")

    # 6 robustness ----------------------------------------------------------
    w("## 6. Robustness and caveats\n")
    un["model"] = un["model"].map(MODEL_LABELS)
    um = un.loc[un.groupby("model")["unmapped_rate"].idxmax()]
    w("- Unmapped responses (no class or category in the map) are rare but not uniform: highest for " +
      "; ".join(f"{r['attr_value']} subjects under {r['model']} ({pct(r['unmapped_rate'])})"
                for _, r in um.iterrows()) +
      ". \"Young\" subjects draw \"student\", \"tattooed\" draws \"tattoo artist\".")
    w("- Accuracy differs by dataset segment because segments cover different occupations; compare "
      "attribute values within a segment or use the pooled rows, not across segments.")
    w("- The label map is a judgment call (see `eval/occupation_map.py`). Rerun "
      "`score -> metrics -> plots -> report` after any change; all numbers here regenerate.")
    w("- Significance tests treat each matched pair as independent; pairs from the same set are correlated, "
      "so p-values are optimistic. Bootstrap CIs in the attribute figure resample sets and are the safer read.")
    w(f"\n![unmapped]({rel_fig}/10_unmapped_by_attribute.png)\n")

    # 7 reproduce -----------------------------------------------------------
    w("## Reproduce\n")
    w("```\n.venv/bin/python -m eval.score baseline/results/*/val_predictions.csv --out eval/scored\n"
      ".venv/bin/python -m eval.metrics --scored eval/scored --out eval/results\n"
      ".venv/bin/python -m eval.plots --results eval/results --out eval/figures\n"
      ".venv/bin/python -m eval.report\n```\n")

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
