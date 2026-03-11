"""Build a local project page for the alignment-method study.

This script works entirely from saved analysis artifacts. It does not rerun
probing or model inference.
"""

from __future__ import annotations

import ast
import html
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "alignment"
PLOTS_DIR = RESULTS_DIR / "plots"
CROSS_METHOD_DIR = PLOTS_DIR / "cross_method"
PAGE_PATH = RESULTS_DIR / "index.html"

COHORT_ORDER = ["base_plus_instruct", "tulu_sft_stack"]
COHORT_LABELS = {
    "base_plus_instruct": "Base + Instruct",
    "tulu_sft_stack": "Tulu SFT/DPO/RLVR",
}
MODEL_TO_COHORT = {
    "llama-3.1-base": "base_plus_instruct",
    "llama-3.1-instruct": "base_plus_instruct",
    "tulu-3-sft": "tulu_sft_stack",
    "tulu-3-dpo": "tulu_sft_stack",
    "tulu-3-rlvr": "tulu_sft_stack",
}
MODEL_ORDER = [
    "llama-3.1-base",
    "llama-3.1-instruct",
    "tulu-3-sft",
    "tulu-3-dpo",
    "tulu-3-rlvr",
]
MODEL_LABELS = {
    "llama-3.1-base": "Base",
    "llama-3.1-instruct": "Instruct",
    "tulu-3-sft": "SFT",
    "tulu-3-dpo": "DPO",
    "tulu-3-rlvr": "RLVR",
}
REPRESENTATIVE_SLICES = {
    "HHH": ("politics", 10),
    "personalprotective": ("philosophy", 10),
}


def _parse_obj(value):
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return value
    return ast.literal_eval(str(value))


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "all_results.csv")
    for col in ["ranking_t0", "ranking_t1", "per_value_delta", "flip_direction", "per_pair_flip_rate"]:
        df[col] = df[col].apply(_parse_obj)
    df["cohort"] = df["model"].map(MODEL_TO_COHORT)
    df["t0_scale"] = df["ranking_t0"].apply(
        lambda d: max(float(np.std(list(d.values()))), 1e-8)
    )
    return df


def build_cohort_summary(df: pd.DataFrame) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for value_set, vs_df in df.groupby("value_set"):
        values = sorted(vs_df.iloc[0]["per_value_delta"].keys())
        summary[value_set] = {}
        for cohort, cohort_df in vs_df.groupby("cohort"):
            cohort_stats = {}
            for value_name in values:
                raw = np.array([row[value_name] for row in cohort_df["per_value_delta"]], dtype=float)
                norm = np.array(
                    [
                        row[value_name] / scale
                        for row, scale in zip(cohort_df["per_value_delta"], cohort_df["t0_scale"])
                    ],
                    dtype=float,
                )
                pos_frac = float(np.mean(norm > 0))
                neg_frac = float(np.mean(norm < 0))
                cohort_stats[value_name] = {
                    "mean_raw": float(np.mean(raw)),
                    "mean_norm": float(np.mean(norm)),
                    "positive_fraction": pos_frac,
                    "negative_fraction": neg_frac,
                    "sign_consistency": float(max(pos_frac, neg_frac)),
                    "n": int(len(norm)),
                }
            summary[value_set][cohort] = cohort_stats
    return summary


def plot_cohort_attractors(summary: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    matplotlib.use("Agg")

    for value_set, cohort_data in summary.items():
        fig, axes = plt.subplots(1, len(COHORT_ORDER), figsize=(14, 7), sharey=True)
        if len(COHORT_ORDER) == 1:
            axes = [axes]

        max_abs = 0.0
        for cohort in COHORT_ORDER:
            values = cohort_data[cohort]
            max_abs = max(max_abs, max(abs(stats["mean_norm"]) for stats in values.values()))
        max_abs = max(0.6, max_abs * 1.18)

        # Use one shared value order across cohorts so labels align with bars.
        value_names = sorted(
            cohort_data[COHORT_ORDER[0]].keys(),
            key=lambda value_name: max(
                abs(cohort_data[cohort][value_name]["mean_norm"]) for cohort in COHORT_ORDER
            ),
            reverse=True,
        )

        for ax, cohort in zip(axes, COHORT_ORDER):
            values = cohort_data[cohort]
            means = [values[value_name]["mean_norm"] for value_name in value_names]
            consistencies = [values[value_name]["sign_consistency"] for value_name in value_names]

            colors = []
            for mean in means:
                if mean > 0.03:
                    colors.append("#1f8f6a")
                elif mean < -0.03:
                    colors.append("#b4493e")
                else:
                    colors.append("#868686")

            ypos = np.arange(len(value_names))
            ax.barh(ypos, means, color=colors, alpha=0.92)
            ax.axvline(0, color="#222222", linewidth=1)
            ax.set_yticks(ypos)
            ax.set_yticklabels(value_names, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlim(-max_abs, max_abs)
            ax.set_title(COHORT_LABELS[cohort], fontsize=13, fontweight="bold")
            ax.set_xlabel("Signed normalized mean drift")

            for y, mean, consistency in zip(ypos, means, consistencies):
                side = "left" if mean >= 0 else "right"
                x = mean + (0.03 * max_abs if mean >= 0 else -0.03 * max_abs)
                ax.text(
                    x,
                    y,
                    f"{mean:+.2f} | cons {consistency:.0%}",
                    va="center",
                    ha=side,
                    fontsize=9,
                    color="#111111",
                )

        fig.suptitle(
            f"Directional Drift by Cohort — {value_set}\n"
            "Green = toward value, red = away from value, grey = near zero mean",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        plt.savefig(CROSS_METHOD_DIR / f"cohort_attractors_{value_set}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def _radar_vectors(row: pd.Series) -> tuple[list[str], list[float], list[float]]:
    values = sorted(row["ranking_t0"].keys())
    t0 = [row["ranking_t0"][value_name] for value_name in values]
    t1 = [row["ranking_t1"][value_name] for value_name in values]
    return values, t0, t1


def _ability_range(df: pd.DataFrame, value_set: str) -> tuple[float, float]:
    subset = df[df["value_set"] == value_set]
    all_values = []
    for _, row in subset.iterrows():
        all_values.extend(row["ranking_t0"].values())
        all_values.extend(row["ranking_t1"].values())
    return float(min(all_values)), float(max(all_values))


def plot_radar_panels(df: pd.DataFrame) -> None:
    matplotlib.use("Agg")

    for value_set, (domain, num_turns) in REPRESENTATIVE_SLICES.items():
        subset = df[
            (df["value_set"] == value_set)
            & (df["domain"] == domain)
            & (df["num_turns"] == num_turns)
        ].copy()
        subset["model_order"] = subset["model"].map({model: idx for idx, model in enumerate(MODEL_ORDER)})
        subset = subset.sort_values("model_order")
        global_min, global_max = _ability_range(df, value_set)

        for mode in ["shared", "local"]:
            fig, axes = plt.subplots(
                2,
                3,
                figsize=(15, 10),
                subplot_kw=dict(polar=True),
            )
            axes = axes.flatten()

            for ax in axes[len(subset) :]:
                ax.axis("off")

            for ax, (_, row) in zip(axes, subset.iterrows()):
                values, t0, t1 = _radar_vectors(row)
                angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
                angles += angles[:1]

                if mode == "shared":
                    low, high = global_min, global_max
                else:
                    low = min(t0 + t1)
                    high = max(t0 + t1)

                spread = max(high - low, 0.25)
                padding = spread * 0.12
                shift = -low + padding
                radial_max = spread + (2 * padding)

                t0_plot = [score + shift for score in t0]
                t1_plot = [score + shift for score in t1]
                t0_plot += t0_plot[:1]
                t1_plot += t1_plot[:1]

                ax.plot(angles, t0_plot, color="#1f4e79", linewidth=2.1, label="T0")
                ax.plot(angles, t1_plot, color="#b45309", linewidth=2.1, linestyle="--", label="T1")
                ax.fill(angles, t0_plot, color="#1f4e79", alpha=0.08)
                ax.fill(angles, t1_plot, color="#b45309", alpha=0.08)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(values, fontsize=9)
                ax.set_ylim(0, radial_max)
                ax.set_yticklabels([])
                ax.set_title(MODEL_LABELS[row["model"]], fontsize=11, pad=16)
                ax.grid(color="#d7dbdf", alpha=0.9)

            axes[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 1.22), frameon=False)
            mode_label = "shared-scale" if mode == "shared" else "within-plot scale"
            fig.suptitle(
                f"{value_set} radar comparison ({domain}, {num_turns} turns, {mode_label})\n"
                "Radial values are shifted BT abilities to keep all radii positive.",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )
            plt.tight_layout()
            plt.savefig(
                CROSS_METHOD_DIR / f"radar_panel_{value_set}_{mode}.png",
                dpi=170,
                bbox_inches="tight",
            )
            plt.close(fig)


def _top_values(summary: dict[str, dict[str, float]], count: int = 3) -> list[str]:
    ordered = sorted(summary.items(), key=lambda item: abs(item[1]["mean_norm"]), reverse=True)
    out = []
    for value_name, stats in ordered[:count]:
        out.append(
            f"{value_name} {stats['mean_norm']:+.2f} "
            f"(consistency {stats['sign_consistency']:.0%})"
        )
    return out


def _method_metric(df: pd.DataFrame, value_set: str, method: str, metric: str) -> float | None:
    subset = df[(df["value_set"] == value_set) & (df["method"] == method)]
    if subset.empty:
        return None
    return float(subset[metric].mean())


def render_page(df: pd.DataFrame, summary: dict[str, dict[str, dict[str, dict[str, float]]]]) -> None:
    counts = {
        "conditions": int(len(df)),
        "models": int(df["model"].nunique()),
        "value_sets": int(df["value_set"].nunique()),
        "domains": int(df["domain"].nunique()),
        "turn_counts": sorted(df["num_turns"].unique().tolist()),
    }

    hhh_base = summary["HHH"]["base_plus_instruct"]
    hhh_tulu = summary["HHH"]["tulu_sft_stack"]
    pp_base = summary["personalprotective"]["base_plus_instruct"]
    pp_tulu = summary["personalprotective"]["tulu_sft_stack"]

    hhh_dpo_l2 = _method_metric(df, "HHH", "DPO", "l2_distance")
    pp_rlvr_l2 = _method_metric(df, "personalprotective", "RLVR", "l2_distance")
    pp_dpo_l2 = _method_metric(df, "personalprotective", "DPO", "l2_distance")

    def img_card(path: str, title: str, caption: str) -> str:
        return (
            '<figure class="card">'
            f'<a href="{html.escape(path)}" target="_blank" rel="noopener noreferrer">'
            f'<img src="{html.escape(path)}" alt="{html.escape(title)}">'
            "</a>"
            f"<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>"
            "</figure>"
        )

    def metric_table(value_set: str) -> str:
        rows = []
        for cohort in COHORT_ORDER:
            cohort_label = COHORT_LABELS[cohort]
            for value_name, stats in sorted(
                summary[value_set][cohort].items(),
                key=lambda item: abs(item[1]["mean_norm"]),
                reverse=True,
            ):
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(cohort_label)}</td>"
                    f"<td>{html.escape(value_name)}</td>"
                    f"<td>{stats['mean_norm']:+.3f}</td>"
                    f"<td>{stats['sign_consistency']:.0%}</td>"
                    "</tr>"
                )
        return (
            '<table class="metrics"><thead><tr><th>Cohort</th><th>Value</th><th>Mean Norm Drift</th>'
            "<th>Sign Consistency</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
        )

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Persona Drifting Across Alignment Methods</title>
  <style>
    :root {{
      --bg: #f5f3ee;
      --panel: #fcfbf8;
      --ink: #1c2026;
      --muted: #59616b;
      --line: #d7dce2;
      --accent: #163756;
      --title: "Libre Franklin", "Helvetica Neue", Helvetica, Arial, sans-serif;
      --body: "Source Serif 4", Georgia, "Times New Roman", serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(180deg, #faf8f3 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: var(--body);
      line-height: 1.58;
    }}
    .page {{
      width: min(1220px, calc(100vw - 36px));
      margin: 24px auto 72px;
    }}
    .masthead, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 10px 24px rgba(18, 29, 43, 0.05);
    }}
    .masthead {{
      padding: 34px 36px 28px;
    }}
    .section {{
      margin-top: 16px;
      padding: 28px 30px;
    }}
    .eyebrow {{
      display: inline-block;
      margin-bottom: 12px;
      font-family: var(--title);
      font-size: 0.8rem;
      letter-spacing: 0.17em;
      text-transform: uppercase;
      color: #697789;
    }}
    h1, h2, h3 {{
      margin: 0 0 10px;
      font-family: var(--title);
      letter-spacing: 0.01em;
    }}
    h1 {{
      font-size: clamp(2.3rem, 4.4vw, 3.8rem);
      line-height: 1.03;
      color: var(--accent);
      max-width: 12ch;
    }}
    h2 {{
      font-size: 1.45rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--accent);
    }}
    h3 {{
      font-size: 0.98rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #627387;
    }}
    p {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 1.01rem;
    }}
    strong {{
      color: var(--ink);
    }}
    code {{
      font-size: 0.95em;
      background: #eef2f5;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    .abstract {{
      max-width: 76ch;
      color: var(--ink);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      margin-top: 22px;
      border-top: 1px solid var(--line);
      border-left: 1px solid var(--line);
    }}
    .meta div {{
      padding: 16px 14px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .meta strong {{
      display: block;
      margin-bottom: 4px;
      font-family: var(--title);
      font-size: 1.35rem;
      color: var(--accent);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: minmax(0, 1.12fr) minmax(0, 0.88fr);
      gap: 26px;
      align-items: start;
    }}
    .metrics-explain {{
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }}
    .metric-box, .finding, .note-box {{
      border: 1px solid var(--line);
      background: #ffffff;
      padding: 16px;
    }}
    .metric-box code {{
      display: inline-block;
      margin-top: 6px;
    }}
    .finding-grid, .note-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }}
    .finding .value {{
      display: block;
      margin-top: 8px;
      font-family: var(--title);
      font-size: 1.14rem;
      color: var(--accent);
    }}
    .figure-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 18px;
    }}
    figure.card {{
      margin: 0;
      border: 1px solid var(--line);
      background: #ffffff;
      padding: 10px;
    }}
    figure.card a {{
      display: block;
      text-decoration: none;
    }}
    figure.card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #fff;
    }}
    figcaption {{
      padding: 10px 4px 2px;
      font-size: 0.95rem;
      color: var(--muted);
    }}
    figcaption strong {{
      display: block;
      margin-bottom: 4px;
      font-family: var(--title);
      font-size: 0.92rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--ink);
    }}
    .metrics {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 18px;
      font-size: 0.95rem;
      background: #ffffff;
      border: 1px solid var(--line);
    }}
    .metrics th, .metrics td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
    }}
    .metrics th {{
      font-family: var(--title);
      font-size: 0.89rem;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--accent);
      background: #f2f5f8;
    }}
    .footnote {{
      margin-top: 14px;
      font-size: 0.93rem;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .page {{ width: min(100vw - 18px, 1220px); }}
      .masthead {{ padding: 26px 22px 22px; }}
      .section {{ padding: 22px; }}
      .two-col, .figure-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="masthead">
      <span class="eyebrow">Research Summary</span>
      <h1>Persona Drifting Across Alignment Methods</h1>
      <p class="abstract">
        This page summarizes saved results from the alignment-method comparison in <code>pipeline/results/alignment/</code>.
        The main empirical result is a cohort split between <strong>Base + Instruct</strong> and the <strong>Tulu SFT/DPO/RLVR</strong> stack,
        especially in <code>HHH</code>. Figures are linked to full-resolution images.
      </p>
      <div class="meta">
        <div><strong>{counts['conditions']}</strong><span>Conditions</span></div>
        <div><strong>{counts['models']}</strong><span>Models</span></div>
        <div><strong>{counts['value_sets']}</strong><span>Value Sets</span></div>
        <div><strong>{counts['domains']}</strong><span>Domains</span></div>
        <div><strong>{", ".join(str(x) + "t" for x in counts['turn_counts'])}</strong><span>Conversation Lengths</span></div>
      </div>
    </section>

    <section class="section">
      <h2>Experimental Scope And Metrics</h2>
      <div class="two-col">
        <div>
          <p><strong>Models:</strong> <code>llama-3.1-base</code>, <code>llama-3.1-instruct</code>, <code>tulu-3-sft</code>, <code>tulu-3-dpo</code>, and <code>tulu-3-rlvr</code>.</p>
          <p><strong>Value sets:</strong> <code>HHH</code> and <code>personalprotective</code>.</p>
          <p><strong>Conditions:</strong> politics and philosophy conversations at 5 and 10 turns, with T0 and T1 Bradley-Terry value rankings saved for each model-condition pair.</p>
        </div>
        <div>
          <p><strong>Interpretation note on sign consistency:</strong> this is <em>not</em> “how often the model preferred the value” across pairwise questions. It is the fraction of conditions in which the value’s signed drift had the same sign after conversation.</p>
          <p>Multiple values can therefore have 100% sign consistency simultaneously if one consistently rises and another consistently falls. Bradley-Terry abilities are relative and mean-centered, so consistent opposite movements are expected.</p>
        </div>
      </div>
      <div class="metrics-explain">
        <div class="metric-box">
          <h3>Per-Value Delta</h3>
          <p>Change in Bradley-Terry ability for one value between T0 and T1.</p>
          <code>delta[v] = ability_T1[v] - ability_T0[v]</code>
        </div>
        <div class="metric-box">
          <h3>Signed Normalized Mean Drift</h3>
          <p>Per-condition delta divided by the T0 standard deviation of the ability vector, then averaged across conditions. Positive means drift toward the value; negative means drift away.</p>
          <code>normalized_delta[v] = delta[v] / std(T0 abilities)</code>
        </div>
        <div class="metric-box">
          <h3>Sign Consistency</h3>
          <p>Fraction of conditions sharing the same drift sign for that value.</p>
          <code>max(frac(normalized_delta &gt; 0), frac(normalized_delta &lt; 0))</code>
        </div>
        <div class="metric-box">
          <h3>L2 Drift And Rank Correlation</h3>
          <p>L2 drift measures total movement in the value vector. Rank correlation measures whether the ordering of values changed.</p>
          <code>L2 = sqrt(sum_v delta[v]^2); rho = Spearman(rank_T0, rank_T1)</code>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>Principal Findings</h2>
      <div class="finding-grid">
        <div class="finding">
          <h3>Finding 1</h3>
          <p><code>HHH</code> separates cleanly by cohort.</p>
          <span class="value">Base + Instruct: harmlessness +1.67, honesty -1.52</span>
          <p class="footnote">Top values: {html.escape(", ".join(_top_values(hhh_base)))}</p>
        </div>
        <div class="finding">
          <h3>Finding 2</h3>
          <p>The Tulu stack trends toward honesty in <code>HHH</code> with lower magnitude than Base + Instruct.</p>
          <span class="value">Tulu stack: honesty +0.22, helpfulness -0.14</span>
          <p class="footnote">Top values: {html.escape(", ".join(_top_values(hhh_tulu)))}</p>
        </div>
        <div class="finding">
          <h3>Finding 3</h3>
          <p><code>personalprotective</code> has a shared authenticity increase, but the rest of the directional pattern remains cohort-dependent.</p>
          <span class="value">Base + Instruct: authenticity +1.24, compliance -1.29</span>
          <p class="footnote">Top values: {html.escape(", ".join(_top_values(pp_base)))}</p>
        </div>
        <div class="finding">
          <h3>Finding 4</h3>
          <p>DPO is the stability anchor in <code>HHH</code>; RLVR and DPO are the lowest-drift aligned methods in <code>personalprotective</code>.</p>
          <span class="value">HHH DPO L2 {hhh_dpo_l2:.3f} | PP RLVR {pp_rlvr_l2:.3f} | PP DPO {pp_dpo_l2:.3f}</span>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>HHH Results</h2>
      <p>
        The <code>HHH</code> results do not support a single aligned-model attractor. Cohort-separated signed drift and per-model heatmaps both indicate
        that Base + Instruct move toward <code>harmlessness</code>, whereas the Tulu stack trends toward <code>honesty</code>.
      </p>
      <div class="figure-grid">
        {img_card("plots/cross_method/cohort_attractors_HHH.png", "Figure H1. Cohort-Specific Signed Drift", "Primary directional evidence for HHH. Signed normalized mean drift is separated by cohort rather than pooled across all aligned models.")}
        {img_card("plots/cross_method/delta_heatmap_HHH.png", "Figure H2. Per-Model Mean Delta", "Raw mean per-value deltas by model. The cohort split in Figure H1 is also visible at the model level.")}
        {img_card("plots/cross_method/radar_panel_HHH_shared.png", "Figure H3. Shared-Scale Radar Comparison", "Representative HHH radar panel for politics, 10 turns, rendered on a shared radial scale for cross-model comparison.")}
        {img_card("plots/cross_method/radar_panel_HHH_local.png", "Figure H4. Within-Plot Radar Comparison", "The same representative HHH slice rendered with local scaling to emphasize within-model shape changes.")}
      </div>
      {metric_table("HHH")}
    </section>

    <section class="section">
      <h2>PersonalProtective Results</h2>
      <p>
        In <code>personalprotective</code>, both cohorts move toward <code>authenticity</code>, but they diverge on <code>harmlessness</code>,
        <code>compliance</code>, <code>privacy</code>, and <code>responsibility</code>. Cohort-specific signed drift is therefore more informative than a pooled attractor view.
      </p>
      <div class="figure-grid">
        {img_card("plots/cross_method/cohort_attractors_personalprotective.png", "Figure P1. Cohort-Specific Signed Drift", "Primary directional evidence for personalprotective. Shared movement toward authenticity remains visible after cohort separation.")}
        {img_card("plots/cross_method/delta_heatmap_personalprotective.png", "Figure P2. Per-Model Mean Delta", "Raw mean per-value deltas by model. Similarities within cohorts are visible, but models are not identical.")}
        {img_card("plots/cross_method/radar_panel_personalprotective_shared.png", "Figure P3. Shared-Scale Radar Comparison", "Representative personalprotective radar panel for philosophy, 10 turns, rendered on a shared radial scale for cross-model comparison.")}
        {img_card("plots/cross_method/radar_panel_personalprotective_local.png", "Figure P4. Within-Plot Radar Comparison", "The same representative personalprotective slice rendered with local scaling to highlight within-model directional structure.")}
      </div>
      {metric_table("personalprotective")}
    </section>

    <section class="section">
      <h2>Notes On Reading The Figures</h2>
      <div class="note-grid">
        <div class="note-box">
          <h3>Sign Consistency</h3>
          <p>A 100% sign consistency for <code>harmlessness</code> and <code>honesty</code> in <code>HHH</code> means those values moved in opposite directions in every included condition. The remaining value, <code>helpfulness</code>, absorbed the residual variation and therefore has lower consistency.</p>
        </div>
        <div class="note-box">
          <h3>Shared vs Local Radar Scale</h3>
          <p>The shared-scale radar panels support cross-model comparison. The locally scaled panels are intended to highlight within-model directional trends that can be visually compressed on a shared scale.</p>
        </div>
      </div>
    </section>
  </main>
</body>
</html>
"""

    PAGE_PATH.write_text(html_text, encoding="utf-8")


def main() -> None:
    df = load_results()
    summary = build_cohort_summary(df)
    plot_cohort_attractors(summary)
    plot_radar_panels(df)
    render_page(df, summary)


if __name__ == "__main__":
    main()
