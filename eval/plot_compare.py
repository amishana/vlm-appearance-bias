"""Plot method-minus-baseline deltas from eval.compare.

Usage:
    python -m eval.plot_compare --compare projection/eval_results/compare \
        --out projection/eval_figures/compare

For every <compare>/<backend>/compare.csv, writes <out>/<backend>_compare.html
(and .png when kaleido can render): one dot per method with its 95% paired
bootstrap CI, in two panels because the good direction differs.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from eval.plots import (AXIS, GRID, INK, INK_2, MODEL_COLORS, MODEL_LABELS, SURFACE,
                        base_layout, configure_models, save)

PANELS = [
    ("Bias: lower is better", [
        ("flip_rate", "gender", "Flip rate · gender"),
        ("flip_rate", "race", "Flip rate · race"),
        ("flip_rate", "physical", "Flip rate · physical"),
        ("attribute_gap", "gender", "Accuracy gap · gender"),
        ("attribute_gap", "race", "Accuracy gap · race"),
        ("attribute_gap", "physical", "Accuracy gap · physical"),
    ]),
    ("Consistency and capability: higher is better", [
        ("consistent_sets", "", "Consistent sets"),
        ("accuracy", "", "Accuracy (synonym)"),
        ("accuracy_category", "", "Accuracy (category)"),
    ]),
]


def fig_compare(table, baseline, methods):
    configure_models([baseline] + methods)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.2,
                        subplot_titles=[title for title, _ in PANELS])
    spread = 0.5 / max(len(methods), 1)
    for col, (_, rows) in enumerate(PANELS, start=1):
        labels = [label for *_, label in rows][::-1]          # first row at the top
        for j, m in enumerate(methods):
            xs, ys, lo, hi, base, meth = [], [], [], [], [], []
            for i, (metric, attr, label) in enumerate(rows[::-1]):
                r = table[(table["method_name"] == m) & (table["metric"] == metric)
                          & (table["attr_type"].fillna("") == attr)]
                if r.empty:
                    continue
                r = r.iloc[0]
                xs.append(100 * r["delta"])
                ys.append(i + (j - (len(methods) - 1) / 2) * spread)
                lo.append(100 * (r["delta"] - r["ci_low"]))
                hi.append(100 * (r["ci_high"] - r["delta"]))
                base.append(100 * r["baseline"])
                meth.append(100 * r["method"])
            fig.add_trace(go.Scatter(
                name=MODEL_LABELS[m], x=xs, y=ys, mode="markers",
                marker={"color": MODEL_COLORS[m], "size": 10, "line": {"color": SURFACE, "width": 2}},
                error_x={"type": "data", "array": hi, "arrayminus": lo,
                         "color": MODEL_COLORS[m], "thickness": 1.5, "width": 0},
                showlegend=(col == 1),
                customdata=list(zip(base, meth)),
                hovertemplate=("Δ %{x:+.1f} pp<br>baseline %{customdata[0]:.1f}% → "
                               "%{customdata[1]:.1f}%<extra>" + MODEL_LABELS[m] + "</extra>"),
            ), row=1, col=col)
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(labels))), ticktext=labels,
                         range=[-0.6, len(labels) - 0.4], showgrid=False,
                         tickfont={"color": INK_2, "size": 12}, row=1, col=col)
        fig.add_vline(x=0, line_color=AXIS, line_width=1, row=1, col=col)
        fig.update_xaxes(title="Method minus baseline (percentage points)", showgrid=True,
                         gridcolor=GRID, zeroline=False, row=1, col=col)
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(
        fig, f"Change from {MODEL_LABELS[baseline]}",
        "Same images for both models. Bars are 95% paired bootstrap CIs over counterfactual sets; "
        "a change is reliable when its bar does not cross 0.",
        height=520, facets=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--compare", type=Path, required=True,
                        help="folder with <backend>/compare.csv from eval.compare")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    files = sorted(args.compare.glob("*/compare.csv"))
    if not files:
        raise SystemExit(f"no */compare.csv under {args.compare}")
    for path in files:
        table = pd.read_csv(path)
        baseline = table["baseline_name"].iloc[0]
        methods = list(dict.fromkeys(table["method_name"]))
        fig = fig_compare(table, baseline, methods)
        name = f"{path.parent.name}_compare"
        save(fig, args.out, name, png=not args.no_png)
        print(f"  wrote {args.out / name}.html")


if __name__ == "__main__":
    main()
