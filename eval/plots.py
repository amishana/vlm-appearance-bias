"""Render the baseline evaluation figures with plotly.

Usage:
    python -m eval.plots --results eval/results --out eval/figures

Each figure is written as interactive HTML and, when kaleido can render, as a
PNG. ``index.html`` links them all. Model colors are fixed by model name so a
model keeps its color in every figure.
"""

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# --- palette (eval/figures follow the project's data-viz palette) -----------
MODEL_COLORS = {            # categorical slots 1-3, validated all-pairs
    "internvl3": "#2a78d6",
    "qwen2_5_vl": "#eb6834",
    "smolvlm2": "#1baf7a",
}
MODEL_LABELS = {
    "internvl3": "InternVL3",
    "qwen2_5_vl": "Qwen2.5-VL",
    "smolvlm2": "SmolVLM2",
}
PAIR_COLORS = {"female": "#6250d6", "male": "#eda100"}   # validated pair
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DIVERGING = [[0.0, "#1c5cab"], [0.5, "#f0efec"], [1.0, "#c0392b"]]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

ATTR_ORDER = {
    "gender": ["male", "female"],
    "race": ["Asian", "Black", "Indian", "Latino", "Middle Eastern", "White"],
    "physical": ["young", "old", "skinny", "obese", "tattooed"],
}
ATTR_TITLES = {"gender": "Gender", "race": "Race", "physical": "Physical appearance"}
TIER_LABELS = {"acc_strict": "Strict", "acc_synonym": "Synonym", "acc_category": "Category"}


def _wrap(text, width=88):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


def base_layout(fig, title, subtitle=None, height=480, width=1000, facets=False):
    text = f"<b>{title}</b>"
    n_sub_lines = 0
    if subtitle:
        wrapped = _wrap(subtitle)
        n_sub_lines = wrapped.count("<br>") + 1
        text += f"<br><span style='font-size:13px;color:{INK_2}'>{wrapped}</span>"
    # top band = title block (title + subtitle lines) + legend row + subplot-title row
    title_block = 36 + 24 * n_sub_lines
    legend_row = 34
    subplot_row = 36 if facets else 0
    top = title_block + legend_row + subplot_row
    plot_h = height - top - 60
    # legend bottom sits just above the subplot titles (or the plot area)
    legend_y = 1.0 + (subplot_row + 2) / plot_h
    fig.update_layout(
        title={"text": text, "x": 0.02, "xanchor": "left", "font": {"size": 17, "color": INK}},
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font={"family": FONT, "color": INK_2, "size": 12},
        height=height, width=width,
        margin={"l": 60, "r": 30, "t": top, "b": 60},
        legend={"orientation": "h", "yanchor": "bottom", "y": legend_y, "xanchor": "right",
                "x": 1.0, "font": {"color": INK_2}, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "#ffffff", "font": {"family": FONT, "color": INK}},
        bargap=0.3, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS,
                     tickfont={"color": MUTED}, title_font={"color": INK_2})
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=AXIS,
                     showline=False, tickfont={"color": MUTED}, title_font={"color": INK_2})
    return fig


def bar(model, x, y, **kw):
    return go.Bar(
        name=MODEL_LABELS[model], x=x, y=y, marker_color=MODEL_COLORS[model],
        marker_cornerradius=4, marker_line_width=0, **kw,
    )


def save(fig, out_dir, name, png=True):
    fig.write_html(out_dir / f"{name}.html", include_plotlyjs="cdn", full_html=True)
    if png:
        try:
            fig.write_image(out_dir / f"{name}.png", scale=2)
        except Exception as exc:  # kaleido/chrome missing
            print(f"  png skipped for {name}: {type(exc).__name__}")
            return False
    return True


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_overall(ov):
    fig = go.Figure()
    tiers = list(TIER_LABELS)
    for model in MODEL_COLORS:
        row = ov[ov["model"] == model].iloc[0]
        ys = [row[t] for t in tiers]
        fig.add_trace(bar(model, [TIER_LABELS[t] for t in tiers], ys,
                          text=[f"{v:.1%}" for v in ys], textposition="outside",
                          textfont={"color": INK_2, "size": 11},
                          hovertemplate="%{x} tier: %{y:.1%}<extra>" + MODEL_LABELS[model] + "</extra>"))
    fig.update_yaxes(title="Accuracy", range=[0, 1], tickformat=".0%")
    fig.update_xaxes(title="Label-matching tier")
    return base_layout(fig, "Overall occupation accuracy by matching tier",
                       "Strict = exact string match. Synonym = same job, merged labels. Category = same broad field.",
                       height=440)


def fig_attribute_accuracy(acc):
    acc = acc[acc["dataset_segment"] == "all"]
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        column_widths=[0.18, 0.47, 0.35],
                        subplot_titles=[ATTR_TITLES[a] for a in ATTR_ORDER],
                        horizontal_spacing=0.03)
    for col, (atype, order) in enumerate(ATTR_ORDER.items(), start=1):
        for model in MODEL_COLORS:
            g = acc[(acc["attr_type"] == atype) & (acc["model"] == model)]
            g = g.set_index("attr_value").reindex(order)
            fig.add_trace(bar(
                model, order, g["acc_synonym"],
                error_y={"type": "data", "array": g["ci_high"] - g["acc_synonym"],
                         "arrayminus": g["acc_synonym"] - g["ci_low"],
                         "color": INK_2, "thickness": 1, "width": 3},
                showlegend=(col == 1),
                customdata=g[["ci_low", "ci_high", "n"]].to_numpy(),
                hovertemplate=("%{x}: %{y:.1%}<br>95% CI %{customdata[0]:.1%} to "
                               "%{customdata[1]:.1%}<br>n = %{customdata[2]:,}"
                               "<extra>" + MODEL_LABELS[model] + "</extra>"),
            ), row=1, col=col)
    fig.update_yaxes(range=[0.4, 0.7], tickformat=".0%")
    fig.update_yaxes(title="Synonym-tier accuracy", row=1, col=1)
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, "Accuracy by attribute value",
                       "Pooled across segments. Error bars are 95% bootstrap CIs over counterfactual sets.",
                       height=500, facets=True)


def fig_per_class_f1(pc, k=15):
    mean_f1 = pc.groupby("class")["f1"].mean().sort_values()
    all_zero = mean_f1[mean_f1 == 0]
    nonzero = mean_f1[mean_f1 > 0]
    bottom = list(nonzero.index[:k])
    top = list(mean_f1.index[-k:])
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.24,
                        subplot_titles=[f"Highest {k} classes (mean F1)",
                                        f"Lowest {k} classes with any correct prediction"])
    for col, classes in [(1, top), (2, bottom)]:
        for model in MODEL_COLORS:
            g = pc[pc["model"] == model].set_index("class").reindex(classes)
            fig.add_trace(go.Scatter(
                name=MODEL_LABELS[model], x=g["f1"], y=classes, mode="markers",
                marker={"color": MODEL_COLORS[model], "size": 10,
                        "line": {"color": SURFACE, "width": 2}},
                showlegend=(col == 1),
                customdata=g[["precision", "recall", "support"]].to_numpy(),
                hovertemplate=("%{y}<br>F1 %{x:.2f} · P %{customdata[0]:.2f} · "
                               "R %{customdata[1]:.2f} · n = %{customdata[2]}"
                               "<extra>" + MODEL_LABELS[model] + "</extra>"),
            ), row=1, col=col)
    fig.update_xaxes(range=[0, 1.02], title="F1 (synonym tier)", gridcolor=GRID, showgrid=True)
    fig.update_yaxes(showgrid=False, tickfont={"color": INK_2, "size": 11})
    fig.update_annotations(font={"color": INK, "size": 13})
    zero_note = (f" {len(all_zero)} further classes score F1 = 0 for every model "
                 f"(e.g. {', '.join(all_zero.index[:4])}): they are always absorbed by a "
                 f"broader class such as doctor or chef.")
    return base_layout(fig, "Per-occupation F1",
                       "Dots are one model each; ranking uses the mean across models." + zero_note,
                       height=580, facets=True)


def fig_consistency(cons):
    cons = cons[cons["n_distinct"] != "mean"]
    order = ["1", "2", "3", "4+"]
    fig = go.Figure()
    for model in MODEL_COLORS:
        g = cons[cons["model"] == model].set_index("n_distinct").reindex(order)
        fig.add_trace(bar(model, order, g["share_of_sets"],
                          text=[f"{v:.0%}" for v in g["share_of_sets"]], textposition="outside",
                          textfont={"color": INK_2, "size": 11},
                          customdata=g[["n_sets"]].to_numpy(),
                          hovertemplate="%{x} distinct: %{y:.1%} of sets (%{customdata[0]:,})<extra>"
                                        + MODEL_LABELS[model] + "</extra>"))
    fig.update_xaxes(title="Distinct predicted occupations within one counterfactual set")
    fig.update_yaxes(title="Share of sets", tickformat=".0%", range=[0, 0.7])
    return base_layout(fig, "Counterfactual consistency",
                       "A set holds the same occupation and scene with only two attributes changed. "
                       "A consistent model predicts one occupation per set.", height=440)


def fig_flip_rates(flips):
    flips = flips[flips["dataset_segment"] == "all"].copy()
    flips["pair"] = flips["value_a"] + " ↔ " + flips["value_b"]
    counts = {a: flips[flips["attr_type"] == a]["pair"].nunique() for a in ATTR_ORDER}
    total = sum(counts.values())
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[max(counts[a], 2) / (total + 3) for a in ATTR_ORDER],
                        subplot_titles=[ATTR_TITLES[a] for a in ATTR_ORDER],
                        vertical_spacing=0.06)
    for row, atype in enumerate(ATTR_ORDER, start=1):
        sub = flips[flips["attr_type"] == atype]
        order = (sub.groupby("pair")["flip_rate"].mean().sort_values().index.tolist())
        for model in MODEL_COLORS:
            g = sub[sub["model"] == model].set_index("pair").reindex(order)
            fig.add_trace(go.Scatter(
                name=MODEL_LABELS[model], x=g["flip_rate"], y=order, mode="markers",
                marker={"color": MODEL_COLORS[model], "size": 10,
                        "line": {"color": SURFACE, "width": 2}},
                showlegend=(row == 1),
                customdata=g[["n_pairs"]].to_numpy(),
                hovertemplate="%{y}<br>flip rate %{x:.1%} · %{customdata[0]:,} pairs<extra>"
                              + MODEL_LABELS[model] + "</extra>",
            ), row=row, col=1)
    fig.update_xaxes(range=[0, 0.32], tickformat=".0%", gridcolor=GRID, showgrid=True)
    fig.update_xaxes(title="Share of matched pairs where the predicted occupation changes", row=3, col=1)
    fig.update_yaxes(showgrid=False, tickfont={"color": INK_2, "size": 11})
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, "Prediction flip rate between attribute values",
                       "Each pair is two images from the same set that differ only in this attribute.",
                       height=780, facets=True)


def fig_occupation_heatmap(oba, k=30):
    # rank occupations by mean spread across models and attribute types
    rank = (oba.groupby("gold_canonical")["spread"].mean().sort_values(ascending=False))
    occs = list(rank.index[:k])[::-1]
    cols = [(a, v) for a in ATTR_ORDER for v in ATTR_ORDER[a]]
    col_labels = [v for _, v in cols]
    lim = float(min(0.5, oba[oba["gold_canonical"].isin(occs)]["deviation"].abs().quantile(0.98)))

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[MODEL_LABELS[m] for m in MODEL_COLORS])
    for c, model in enumerate(MODEL_COLORS, start=1):
        g = oba[oba["model"] == model]
        z, text = [], []
        for occ in occs:
            zr, tr = [], []
            for atype, val in cols:
                r = g[(g["gold_canonical"] == occ) & (g["attr_type"] == atype) & (g["attr_value"] == val)]
                if r.empty:
                    zr.append(None); tr.append("no images")
                else:
                    r = r.iloc[0]
                    zr.append(r["deviation"])
                    tr.append(f"acc {r['acc']:.0%} vs occupation mean {r['occupation_mean']:.0%} (n={int(r['n'])})")
            z.append(zr); text.append(tr)
        fig.add_trace(go.Heatmap(
            z=z, x=col_labels, y=occs, text=text, zmin=-lim, zmax=lim, zmid=0,
            colorscale=DIVERGING, xgap=2, ygap=2,
            showscale=(c == 3),
            colorbar={"title": "Δ accuracy", "tickformat": "+.0%", "thickness": 12, "len": 0.6},
            hovertemplate="%{y} · %{x}<br>%{text}<extra>" + MODEL_LABELS[model] + "</extra>",
        ), row=1, col=c)
        fig.update_xaxes(tickangle=-45, tickmode="array", tickvals=col_labels,
                         ticktext=col_labels, row=1, col=c)
    fig.update_yaxes(tickfont={"color": INK_2, "size": 11})
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, f"Occupation × attribute accuracy deviation (top {k} by spread)",
                       "Cell = accuracy for that attribute value minus the occupation's mean accuracy within "
                       "that attribute type. Blue = below mean, red = above. Blank = occupation not in that segment.",
                       height=920, facets=True)


def fig_stereotype_gender(lor):
    g = lor[(lor["attr_type"] == "gender") & (lor["attr_value"] == "female")]
    classes = (g.groupby("pred_class")["log_odds_ratio"].mean().sort_values().index.tolist())
    fig = go.Figure()
    for model in MODEL_COLORS:
        gm = g[g["model"] == model].set_index("pred_class").reindex(classes)
        fig.add_trace(go.Bar(
            name=MODEL_LABELS[model], y=classes, x=gm["log_odds_ratio"], orientation="h",
            marker_color=MODEL_COLORS[model], marker_cornerradius=4, marker_line_width=0,
            error_x={"type": "data", "array": gm["ci_high"] - gm["log_odds_ratio"],
                     "arrayminus": gm["log_odds_ratio"] - gm["ci_low"],
                     "color": INK_2, "thickness": 1, "width": 2},
            customdata=gm[["share_in_group", "n_pred"]].to_numpy(),
            hovertemplate=("%{y}<br>log-odds(female) %{x:+.2f}<br>%{customdata[0]:.0%} of "
                           "%{customdata[1]:,} predictions were female subjects"
                           "<extra>" + MODEL_LABELS[model] + "</extra>"),
        ))
    fig.add_vline(x=0, line_color=AXIS, line_width=1)
    fig.update_xaxes(title="Log-odds of a female subject receiving this prediction "
                           "(← more for men · more for women →)",
                     gridcolor=GRID, showgrid=True, zeroline=False)
    fig.update_yaxes(showgrid=False, tickfont={"color": INK_2, "size": 11})
    fig.update_layout(barmode="group")
    return base_layout(fig, "Gender skew of frequently predicted occupations",
                       "Top 15 predicted classes per model. 0 = predicted equally for male and female subjects; "
                       "a missing bar means the class is not in that model's top 15.", height=620)


def fig_stereotype_race(lor):
    g = lor[lor["attr_type"] == "race"]
    classes = (g.groupby("pred_class")["log_odds_ratio"].apply(lambda s: s.abs().mean())
               .sort_values().index.tolist())
    order = ATTR_ORDER["race"]
    lim = float(min(1.5, g["log_odds_ratio"].abs().quantile(0.98)))
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=[MODEL_LABELS[m] for m in MODEL_COLORS])
    for c, model in enumerate(MODEL_COLORS, start=1):
        gm = g[g["model"] == model]
        z, text = [], []
        for cls in classes:
            zr, tr = [], []
            for val in order:
                r = gm[(gm["pred_class"] == cls) & (gm["attr_value"] == val)]
                if r.empty:
                    zr.append(None); tr.append("not in top 15")
                else:
                    r = r.iloc[0]
                    zr.append(r["log_odds_ratio"])
                    tr.append(f"{r['share_in_group']:.0%} of {int(r['n_pred'])} predictions "
                              f"(base rate {r['base_rate']:.0%})")
            z.append(zr); text.append(tr)
        fig.add_trace(go.Heatmap(
            z=z, x=order, y=classes, text=text, zmin=-lim, zmax=lim, zmid=0,
            colorscale=DIVERGING, xgap=2, ygap=2, showscale=(c == 3),
            colorbar={"title": "log-odds", "tickformat": "+.1f", "thickness": 12, "len": 0.6},
            hovertemplate="%{y} · %{x}<br>log-odds %{z:+.2f}<br>%{text}<extra>" + MODEL_LABELS[model] + "</extra>",
        ), row=1, col=c)
        fig.update_xaxes(tickangle=-45, row=1, col=c)
    fig.update_yaxes(tickfont={"color": INK_2, "size": 11})
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, "Race skew of frequently predicted occupations",
                       "Log-odds ratio of each race receiving the prediction, relative to its base rate. "
                       "Blue = under-predicted for that group, red = over-predicted. "
                       "Blank row = class not in that model's top 15.", height=640, facets=True)


def fig_gendered_words(gw):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[MODEL_LABELS[m] for m in MODEL_COLORS])
    for c, model in enumerate(MODEL_COLORS, start=1):
        g = gw[gw["model"] == model].set_index("subject_gender").reindex(["male", "female"])
        for word, col in [("male", "rate_male_word"), ("female", "rate_female_word")]:
            fig.add_trace(go.Bar(
                name=f"{word}-marked word",
                x=["male subject", "female subject"], y=g[col],
                marker_color=PAIR_COLORS[word], marker_cornerradius=4, marker_line_width=0,
                text=[f"{v:.1%}" for v in g[col]], textposition="outside",
                textfont={"color": INK_2, "size": 11},
                showlegend=(c == 1),
                hovertemplate="%{x}: %{y:.2%} of responses<extra>" + f"{word}-marked word" + "</extra>",
            ), row=1, col=c)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.08])
    fig.update_yaxes(title="Share of responses", row=1, col=1)
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, "Use of gendered occupation words by subject gender",
                       "Male-marked: businessman, policeman, fireman. Female-marked: waitress, businesswoman, "
                       "policewoman. Both still count as correct at the synonym tier.",
                       height=460, facets=True)


def fig_unmapped(un):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                        column_widths=[0.18, 0.47, 0.35],
                        subplot_titles=[ATTR_TITLES[a] for a in ATTR_ORDER],
                        horizontal_spacing=0.03)
    for col, (atype, order) in enumerate(ATTR_ORDER.items(), start=1):
        for model in MODEL_COLORS:
            g = un[(un["attr_type"] == atype) & (un["model"] == model)]
            g = g.set_index("attr_value").reindex(order)
            fig.add_trace(bar(model, order, g["unmapped_rate"], showlegend=(col == 1),
                              hovertemplate="%{x}: %{y:.1%} unmapped<extra>" + MODEL_LABELS[model] + "</extra>"),
                          row=1, col=col)
    fig.update_yaxes(tickformat=".0%", range=[0, 0.08])
    fig.update_yaxes(title="Unmapped share of responses", row=1, col=1)
    fig.update_annotations(font={"color": INK, "size": 13})
    return base_layout(fig, "Unmapped responses by attribute value",
                       "Responses like “student” or “diver” that match no class or category in the label map.",
                       height=460, facets=True)


FIGURES = [
    ("01_overall_accuracy", "Overall accuracy by matching tier", fig_overall, "overall.csv"),
    ("02_attribute_accuracy", "Accuracy by attribute value", fig_attribute_accuracy, "attribute_accuracy.csv"),
    ("03_per_class_f1", "Per-occupation F1", fig_per_class_f1, "per_class.csv"),
    ("04_consistency", "Counterfactual consistency", fig_consistency, "consistency.csv"),
    ("05_flip_rates", "Prediction flip rates", fig_flip_rates, "flip_rates.csv"),
    ("06_occupation_heatmap", "Occupation × attribute deviation", fig_occupation_heatmap, "occupation_by_attribute.csv"),
    ("07_stereotype_gender", "Gender skew of predictions", fig_stereotype_gender, "stereotype_logodds.csv"),
    ("08_stereotype_race", "Race skew of predictions", fig_stereotype_race, "stereotype_logodds.csv"),
    ("09_gendered_words", "Gendered word use", fig_gendered_words, "gendered_words.csv"),
    ("10_unmapped_by_attribute", "Unmapped responses by attribute", fig_unmapped, "unmapped_by_attribute.csv"),
]


def write_index(out_dir, entries):
    items = "\n".join(
        f'<li><a href="{name}.html">{title}</a>'
        + (f' · <a href="{name}.png">png</a>' if png else "")
        + f' · <span class="src">{src}</span></li>'
        for name, title, png, src in entries
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Baseline evaluation figures</title>
<style>
body{{font-family:{FONT};background:{SURFACE};color:{INK};max-width:720px;margin:40px auto;padding:0 16px}}
h1{{font-size:20px}} li{{margin:8px 0;line-height:1.5}} a{{color:#2a78d6}} .src{{color:{MUTED};font-size:12px}}
p{{color:{INK_2}}}
</style></head><body>
<h1>Baseline evaluation figures</h1>
<p>Pre-mitigation occupation-prediction results for three VLMs on SocialCounterfactuals (val split).
Interactive HTML, with the source table from <code>eval/results/</code> named beside each.</p>
<ol>{items}</ol></body></html>"""
    (out_dir / "index.html").write_text(html)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=Path("eval/results"))
    parser.add_argument("--out", type=Path, default=Path("eval/figures"))
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pio.templates.default = "plotly_white"

    entries = []
    for name, title, builder, src in FIGURES:
        frame = pd.read_csv(args.results / src, keep_default_na=False)
        fig = builder(frame)
        png = save(fig, args.out, name, png=not args.no_png)
        entries.append((name, title, png, src))
        print(f"  wrote {name}")
    write_index(args.out, entries)
    print(f"index: {args.out / 'index.html'}")


if __name__ == "__main__":
    main()
