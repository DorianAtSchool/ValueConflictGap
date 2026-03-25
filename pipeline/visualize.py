"""Visualization: radar charts, trajectory plots, heatmaps, flip rate charts."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from pathlib import Path

from config import RESULTS_DIR, PERSONAS, VALUE_SETS


def _load_all_results() -> pd.DataFrame:
    """Load all saved result JSONs into a single DataFrame."""
    records = []
    results_dir = RESULTS_DIR / "runs"
    if not results_dir.exists():
        return pd.DataFrame()
    for path in results_dir.glob("*.json"):
        with open(path) as f:
            records.append(json.load(f))
    return pd.DataFrame(records)


def plot_radar_t0_t1(
    ranking_t0: pd.DataFrame,
    ranking_t1: pd.DataFrame,
    title: str,
    output_path: Path,
):
    """Spider/radar chart comparing T0 vs T1 BT scores."""
    values = ranking_t0.sort_values("value")["value"].tolist()
    t0_scores = ranking_t0.set_index("value").loc[values, "ability"].tolist()
    t1_scores = ranking_t1.set_index("value").loc[values, "ability"].tolist()

    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    t0_scores += t0_scores[:1]
    t1_scores += t1_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, t0_scores, "o-", label="T0 (baseline)", linewidth=2)
    ax.plot(angles, t1_scores, "s--", label="T1 (post-conversation)", linewidth=2)
    ax.fill(angles, t0_scores, alpha=0.1)
    ax.fill(angles, t1_scores, alpha=0.1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(values, size=10)
    ax.set_title(title, size=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_trajectories_pca(results_df: pd.DataFrame, output_path: Path):
    """PCA trajectory plot per value set: arrows from T0 -> T1, colored by persona."""
    if results_df.empty:
        return

    for vs, group in results_df.groupby("value_set"):
        t0_vectors = []
        t1_vectors = []
        personas = []
        domains = []

        for _, row in group.iterrows():
            t0 = row["ranking_t0"]
            t1 = row["ranking_t1"]
            values_sorted = sorted(t0.keys())
            t0_vectors.append([t0[v] for v in values_sorted])
            t1_vectors.append([t1[v] for v in values_sorted])
            personas.append(row["persona"])
            domains.append(row["domain"])

        all_vectors = np.array(t0_vectors + t1_vectors)
        if all_vectors.shape[0] < 3:
            continue
        n_components = min(2, all_vectors.shape[1])
        pca = PCA(n_components=n_components)
        projected = pca.fit_transform(all_vectors)
        n = len(t0_vectors)
        t0_proj = projected[:n]
        t1_proj = projected[n:]

        fig, ax = plt.subplots(figsize=(12, 10))
        unique_personas = sorted(set(personas))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_personas), 1)))
        persona_color = {p: colors[i] for i, p in enumerate(unique_personas)}

        for i in range(n):
            c = persona_color[personas[i]]
            ax.annotate(
                "",
                xy=t1_proj[i],
                xytext=t0_proj[i],
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5),
            )
            ax.scatter(*t0_proj[i], c=[c], marker="o", s=40, zorder=5)
            ax.scatter(*t1_proj[i], c=[c], marker="x", s=40, zorder=5)

        for p in unique_personas:
            ax.scatter([], [], c=[persona_color[p]], label=p)
        ax.legend(title="Persona")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        if n_components > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"Value Ranking Trajectories — {vs}")

        out = output_path.parent / f"{output_path.stem}_{vs}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_drift_heatmap(results_df: pd.DataFrame, output_path: Path):
    """Heatmap of L2 drift magnitude: persona x domain, aggregated over turn lengths."""
    if results_df.empty:
        return

    for vs, group in results_df.groupby("value_set"):
        pivot = group.pivot_table(
            values="l2_distance", index="persona", columns="domain", aggfunc="mean"
        )
        # Shorten domain names for display
        pivot.columns = [c.replace("value_aligned_", "VA:") for c in pivot.columns]
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.8)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
        ax.set_title(f"Mean L2 Drift — {vs}")
        out = output_path.parent / f"{output_path.stem}_{vs}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_flip_rate_bars(results_df: pd.DataFrame, output_path: Path):
    """Bar chart of overall answer flip rate per persona."""
    if results_df.empty:
        return

    flip_rates = results_df.groupby("persona")["overall_flip_rate"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    flip_rates.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean Answer Flip Rate")
    ax.set_title("Answer Flip Rate by Persona")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_attractor_analysis(results_df: pd.DataFrame, output_path: Path):
    """Cluster T1 vectors per value set to check for attractor convergence."""
    if results_df.empty or len(results_df) < 4:
        return

    for vs, group in results_df.groupby("value_set"):
        t1_vectors = []
        labels_list = []
        for _, row in group.iterrows():
            t1 = row["ranking_t1"]
            values_sorted = sorted(t1.keys())
            t1_vectors.append([t1[v] for v in values_sorted])
            labels_list.append(row["persona"])

        t1_arr = np.array(t1_vectors)
        if len(t1_arr) < 4:
            continue
        n_components = min(2, t1_arr.shape[1])
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(t1_arr)

        n_clusters = min(4, len(proj))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = km.fit_predict(proj)

        fig, ax = plt.subplots(figsize=(10, 8))
        unique_personas = sorted(set(labels_list))
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_personas), 1)))
        persona_color = {p: colors[i] for i, p in enumerate(unique_personas)}

        for i in range(len(proj)):
            ax.scatter(proj[i, 0], proj[i, 1] if n_components > 1 else 0,
                       c=[persona_color[labels_list[i]]], s=60, zorder=5)

        ax.scatter(km.cluster_centers_[:, 0],
                   km.cluster_centers_[:, 1] if n_components > 1 else np.zeros(n_clusters),
                   marker="*", s=200, c="red", label="Centroids", zorder=10)

        for p in unique_personas:
            ax.scatter([], [], c=[persona_color[p]], label=p)
        ax.legend(title="Persona")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        if n_components > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"T1 Attractor Analysis — {vs}")

        out = output_path.parent / f"{output_path.stem}_{vs}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_drift_by_turns(results_df: pd.DataFrame, output_path: Path):
    """Line plot of L2 drift vs turn count, per persona, faceted by value set."""
    if results_df.empty:
        return

    for vs, group in results_df.groupby("value_set"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for persona, pgroup in group.groupby("persona"):
            by_turns = pgroup.groupby("num_turns")["l2_distance"].mean()
            ax.plot(by_turns.index, by_turns.values, "o-", label=persona, linewidth=2)
        ax.set_xlabel("Number of Turns")
        ax.set_ylabel("Mean L2 Drift")
        ax.set_title(f"Drift vs Conversation Length — {vs}")
        ax.legend(title="Persona")
        ax.set_xticks(sorted(group["num_turns"].unique()))

        out = output_path.parent / f"{output_path.stem}_{vs}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def generate_all_plots(results_df: pd.DataFrame):
    """Generate all visualization plots from aggregated results."""
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_trajectories_pca(results_df, plots_dir / "trajectories_pca.png")
    plot_drift_heatmap(results_df, plots_dir / "drift_heatmap.png")
    plot_flip_rate_bars(results_df, plots_dir / "flip_rate_bars.png")
    plot_attractor_analysis(results_df, plots_dir / "attractor_analysis.png")
    plot_drift_by_turns(results_df, plots_dir / "drift_by_turns.png")


# ---------------------------------------------------------------------------
# Scenario conversation experiment plots
# ---------------------------------------------------------------------------

def plot_pair_consistency_heatmap(result: dict, output_path: Path):
    """Heatmap of per-pair flip rate and directional consistency.

    Rows = value pairs (with scenario count), columns: flip_rate and
    directional_consistency.  A text column on the right shows the dominant
    value (i.e. the value toward which flips went).

    flip_rate        — fraction of scenarios that changed answer T0→T1
    dir. consistency — 1.0 = all flips same direction; 0.5 = random
    dominant         — the value that received more flips in this pair
    n                — number of scenarios in this pair (sample size)
    """
    pc = result.get("pair_consistency", {})
    if not pc:
        return

    pairs = sorted(pc.keys())
    flip_rates = [pc[p]["flip_rate"] for p in pairs]
    consistencies = [
        pc[p]["directional_consistency"] if not np.isnan(pc[p]["directional_consistency"]) else 0.0
        for p in pairs
    ]
    dominant = [pc[p]["dominant_value"] for p in pairs]
    # Include n_scenarios in row labels so sample size is always visible
    short = [
        f"{p.replace('_vs_', ' vs ')}  (n={pc[p]['n_scenarios']})"
        for p in pairs
    ]

    data = pd.DataFrame({
        "flip rate": flip_rates,
        "dir. consistency": consistencies,
    }, index=short)

    row_height = 0.6
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(pairs) * row_height + 1.5)))

    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, ax=ax, cbar=False,
    )
    stance = result.get("stance", "neutral")
    mode   = result.get("mode", "mcq")
    ax.set_title(
        f"Pair Consistency — {result['model']} · {result['value_set']} · "
        f"{stance} · {result['num_turns']} turns"
        + (f" · {mode}" if mode != "mcq" else ""),
        fontsize=10, pad=10,
    )
    ax.set_ylabel("")

    fig.canvas.draw()
    ax_pos = ax.get_position()
    n = len(pairs)
    for i, dom in enumerate(dominant):
        row_frac = (i + 0.5) / n
        fig_y = ax_pos.y1 - row_frac * ax_pos.height
        fig.text(
            ax_pos.x1 + 0.02, fig_y, dom,
            va="center", ha="left", fontsize=7, color="navy",
            transform=fig.transFigure,
        )

    fig.subplots_adjust(right=0.68)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _flip_bars(ax, values, pvfs, title):
    """Draw a flip-toward / flip-away grouped bar chart on *ax*."""
    def _s(x):
        return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

    toward = [_s(pvfs[v]["flip_rate_toward"]) for v in values]
    away   = [_s(pvfs[v]["flip_rate_away"])   for v in values]
    ns     = [pvfs[v].get("total_appearances", 0) for v in values]

    x = np.arange(len(values))
    w = 0.35
    b_t = ax.bar(x - w / 2, toward, w, label="Flip toward", color="steelblue", alpha=0.85)
    b_a = ax.bar(x + w / 2, away,   w, label="Flip away",   color="tomato",    alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{v}\n(n={n})" for v, n in zip(values, ns)],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_ylabel("Rate")
    peak = float(np.nanmax(toward + away)) if any(v > 0 for v in toward + away) else 0.1
    ax.set_ylim(0, max(peak * 1.3, 0.05))
    ax.bar_label(b_t, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(b_a, fmt="%.2f", fontsize=7, padding=2)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_per_value_flip_stats(result: dict, output_path: Path):
    """Grouped bar chart: flip-toward-rate and flip-away-rate per value.

    For non-neutral stances, shows two panels side by side:
      Left  — Overall: all pairs, all appearances (unfiltered)
      Right — Role-filtered: only appearances where the value was in the
              favoured role (v1 for pro_v1, v2 for pro_v2)

    The n= count under each label is the number of scenarios that back the bar.
    """
    pvfs_filtered = result.get("per_value_flip_stats", {})
    pvfs_overall  = result.get("per_value_flip_stats_overall", {})
    if not pvfs_filtered:
        return

    stance = result.get("stance", "neutral")
    mode   = result.get("mode", "mcq")
    model  = result["model"]
    vs     = result["value_set"]
    nt     = result["num_turns"]
    base_title = f"{model} · {vs} · {stance} · {nt} turns" + (f" · {mode}" if mode != "mcq" else "")

    values = sorted(pvfs_filtered.keys())

    if stance == "neutral" or not pvfs_overall or pvfs_overall == pvfs_filtered:
        # Single panel
        fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.9), 5))
        _flip_bars(ax, values, pvfs_filtered, f"Flip Rates — {base_title}")
    else:
        # Dual panel: overall | role-filtered
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(max(14, len(values) * 1.8), 5),
                                          sharey=True)
        _flip_bars(ax_l, values, pvfs_overall,
                   f"Overall (all pairs) — {base_title}")
        _flip_bars(ax_r, values, pvfs_filtered,
                   f"Role-filtered (favoured role only) — {base_title}")
        fig.suptitle(f"Flip Rates — {base_title}", fontsize=10, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scenario_radar(result: dict, output_path: Path):
    """Radar (spider) chart comparing T0 vs T1 BT scores for a single result."""
    ranking_t0 = pd.DataFrame(result["ranking_t0"])
    ranking_t1 = pd.DataFrame(result["ranking_t1"])
    if ranking_t0.empty or ranking_t1.empty:
        return
    title = f"{result['model']} — {result['value_set']} — {result['num_turns']}t"
    plot_radar_t0_t1(ranking_t0, ranking_t1, title=title, output_path=output_path)


def _drift_bar_ax(ax, delta: dict, title: str):
    """Draw a horizontal BT-delta bar chart on *ax*."""
    values = sorted(delta.keys(), key=lambda v: delta[v])
    deltas = [delta[v] for v in values]
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]
    bars = ax.barh(values, deltas, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("BT score delta (T1 − T0)")
    ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=3)
    ax.set_title(title, fontsize=9)


def plot_scenario_drift_bars(result: dict, output_path: Path):
    """Horizontal bar chart of per-value BT score delta (T1 − T0).

    For non-neutral stances, shows two panels side by side:
      Left  — Overall: global BT delta (all scenarios contribute)
      Right — Role-filtered: BT delta computed only from scenarios where
              each value was in the favoured role (v1/v2 matching stance)

    A positive delta means the value's BT ability rose after the conversation.
    """
    stance   = result.get("stance", "neutral")
    mode     = result.get("mode", "mcq")
    base     = f"{result['model']} · {result['value_set']} · {stance} · {result['num_turns']} turns" \
               + (f" · {mode}" if mode != "mcq" else "")
    delta_overall  = result.get("drift", {}).get("per_value_delta", {})
    delta_filtered = result.get("role_filtered_drift", {}).get("per_value_delta", {})

    if not delta_overall:
        return

    if stance == "neutral" or not delta_filtered:
        fig, ax = plt.subplots(figsize=(7, max(3, len(delta_overall) * 0.45 + 1)))
        _drift_bar_ax(ax, delta_overall, f"Value Drift (BT delta) — {base}")
    else:
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(13, max(3, len(delta_overall) * 0.45 + 1))
        )
        _drift_bar_ax(ax_l, delta_overall,
                      f"Overall (all pairs) — {base}")
        _drift_bar_ax(ax_r, delta_filtered,
                      f"Role-filtered (favoured role only) — {base}")
        fig.suptitle(f"Value Drift (BT delta) — {base}", fontsize=10, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranking_heatmap_single(result: dict, output_path: Path):
    """Per-condition ranking heatmap: T0 vs T1 ranks as a 2-row matrix.

    Rows = [T0, T1]; Columns = values; Cell = ordinal rank (1=highest priority).
    Same colour scheme as the cross-condition ranking heatmap.
    """
    t0_list = result.get("ranking_t0", [])
    t1_list = result.get("ranking_t1", [])
    if not t0_list or not t1_list:
        return

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "rank_cmap", ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    )
    t0d = _ranking_list_to_dict(t0_list)
    t1d = _ranking_list_to_dict(t1_list)
    values = sorted(set(t0d) | set(t1d))
    n = len(values)

    t0_ranks = _ability_to_rank_scenario(t0d)
    t1_ranks = _ability_to_rank_scenario(t1d)
    matrix = np.array([[t0_ranks.get(v, np.nan) for v in values],
                       [t1_ranks.get(v, np.nan) for v in values]])

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), 2.5))
    ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=1, vmax=n)
    for ri, row in enumerate(matrix):
        for ci, val in enumerate(row):
            if not np.isnan(val):
                tc = "white" if val > n * 0.6 else "black"
                ax.text(ci, ri, f"{int(val)}", ha="center", va="center",
                        fontsize=9, color=tc, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(values, rotation=40, ha="right", fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["T0 (baseline)", "T1 (post-conv)"], fontsize=9)
    stance = result.get("stance", "neutral")
    mode = result.get("mode", "mcq")
    ax.set_title(
        f"Value Rankings — {result['model']} / {result['value_set']} / "
        f"{result['num_turns']}t / {stance} / {mode}\n"
        "(rank 1 = highest BT ability; lighter = higher priority)",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rank_shift_heatmap_single(result: dict, output_path: Path):
    """Per-condition rank-shift heatmap: single row showing T1 rank − T0 rank per value.

    Blue = value rose in priority (rank decreased), red = dropped.
    """
    t0_list = result.get("ranking_t0", [])
    t1_list = result.get("ranking_t1", [])
    if not t0_list or not t1_list:
        return

    t0d = _ranking_list_to_dict(t0_list)
    t1d = _ranking_list_to_dict(t1_list)
    values = sorted(set(t0d) | set(t1d))
    n = len(values)

    t0_ranks = _ability_to_rank_scenario(t0d)
    t1_ranks = _ability_to_rank_scenario(t1d)
    deltas = np.array([[t1_ranks.get(v, np.nan) - t0_ranks.get(v, np.nan) for v in values]])

    max_abs = max(np.nanmax(np.abs(deltas)) if not np.all(np.isnan(deltas)) else 1, 1)

    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), 1.8))
    im = ax.imshow(deltas, cmap="RdBu", aspect="auto", vmin=-max_abs, vmax=max_abs)
    for ci, val in enumerate(deltas[0]):
        if not np.isnan(val):
            sign = "+" if val > 0 else ""
            ax.text(ci, 0, f"{sign}{int(val)}", ha="center", va="center",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(values, rotation=40, ha="right", fontsize=8)
    ax.set_yticks([])

    stance = result.get("stance", "neutral")
    mode = result.get("mode", "mcq")
    ax.set_title(
        f"Rank Shift (T1−T0) — {result['model']} / {result['value_set']} / "
        f"{result['num_turns']}t / {stance} / {mode}\n"
        "(blue = rose in priority, red = dropped)",
        fontsize=9,
    )

    # Colorbar: dedicated axis to the right so it never overlaps the matrix
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.87, 0.25, 0.025, 0.5])
    fig.colorbar(im, cax=cax, label="Rank change")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_scenario_experiment_plots(
    all_results: list[dict], results_dir: Path, run_id: str | None = None
):
    """Generate all plots for the scenario conversation experiment.

    Folder layout::

        plots/{run_id}/
          per_condition/{model}/{value_set}/{stance}/{N}t[_{mode}]/
              radar.png               BT spider chart T0 vs T1
              bt_ranking_bars.png     absolute BT abilities ± 95 % CI
              drift_bars.png          per-value BT delta; dual-panel for non-neutral
              flip_rates.png          flip-toward/away; dual-panel for non-neutral
              pair_consistency.png    per-pair flip rate & directional consistency
              ranking_heatmap.png     ordinal ranks T0 vs T1
              rank_shift_heatmap.png  rank change from T0 baseline
          cross_condition/
            ranking/                  ranking & radar panels across turn counts
            drift/                    L2 drift curves & heatmaps
            aggregated/               turn-count-averaged drift & flip rates
            comparison/               stance, mode, and model comparisons
    """
    if not all_results:
        return

    if run_id is None:
        from datetime import datetime
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    plots_dir = results_dir / "plots" / run_id
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_cond_root = plots_dir / "per_condition"
    cross_root    = plots_dir / "cross_condition"
    ranking_dir   = cross_root / "ranking"
    drift_dir     = cross_root / "drift"
    agg_dir       = cross_root / "aggregated"
    cmp_dir       = cross_root / "comparison"

    # --- Per-condition plots ---
    for r in all_results:
        stance   = r.get("stance", "neutral")
        mode     = r.get("mode", "mcq")
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        cond_dir = per_cond_root / r["model"] / r["value_set"] / stance / f"{r['num_turns']}t{mode_tag}"
        cond_dir.mkdir(parents=True, exist_ok=True)

        plot_pair_consistency_heatmap(r, cond_dir / "pair_consistency.png")
        plot_per_value_flip_stats(r, cond_dir / "flip_rates.png")
        plot_scenario_radar(r, cond_dir / "radar.png")
        plot_scenario_drift_bars(r, cond_dir / "drift_bars.png")
        plot_bt_ranking_bars(r, cond_dir / "bt_ranking_bars.png")
        plot_ranking_heatmap_single(r, cond_dir / "ranking_heatmap.png")
        plot_rank_shift_heatmap_single(r, cond_dir / "rank_shift_heatmap.png")

    # --- Cross-condition: ranking & radar ---
    plot_ranking_heatmap_scenario(all_results, ranking_dir / "ranking_heatmap.png")
    plot_rank_shift_heatmap_scenario(all_results, ranking_dir / "rank_shift_heatmap.png")
    plot_radar_panel_scenario(all_results, ranking_dir / "radar_panel.png")

    # --- Cross-condition: drift curves ---
    plot_drift_by_turns_scenario(all_results, drift_dir / "drift_by_turns.png")
    plot_l2_heatmap_scenario(all_results, drift_dir / "l2_heatmap.png")

    # --- Cross-condition: aggregated (averaged over turn counts) ---
    plot_aggregated_drift_bars(all_results, agg_dir / "drift_bars.png")
    plot_aggregated_per_value_flip_rate(all_results, agg_dir / "flip_rates.png")

    # --- Cross-condition: comparisons ---
    stances_present = {r.get("stance", "neutral") for r in all_results}
    if len(stances_present) > 1:
        plot_stance_comparison(all_results, cmp_dir / "stance_comparison.png")

    modes_present = {r.get("mode", "mcq") for r in all_results}
    if len(modes_present) > 1:
        plot_mode_comparison(all_results, cmp_dir / "mode_comparison.png")

    models_present = {r["model"] for r in all_results}
    if len(models_present) > 1:
        plot_model_comparison(all_results, cmp_dir / "model_comparison.png")


# ---------------------------------------------------------------------------
# Per-condition helpers
# ---------------------------------------------------------------------------

def _ranking_list_to_dict(ranking_list: list[dict]) -> dict[str, float]:
    """Convert [{value, ability, ...}, ...] to {value: ability}."""
    return {r["value"]: r["ability"] for r in ranking_list}


def _ability_to_rank_scenario(ability_dict: dict[str, float]) -> dict[str, int]:
    """Convert {value: ability} to {value: rank} (1 = highest ability)."""
    sorted_vals = sorted(ability_dict, key=lambda v: ability_dict[v], reverse=True)
    return {v: i + 1 for i, v in enumerate(sorted_vals)}


def plot_bt_ranking_bars(result: dict, output_path: Path):
    """Side-by-side horizontal bar chart of absolute BT abilities at T0 and T1 with 95% CIs.

    This is the most informative single-condition plot: it shows the actual ranking
    order, confidence intervals, and which values moved significantly.

    A BT ability difference of D between two values means the higher-ranked value
    wins with probability σ(D) = e^D / (1 + e^D). So a delta of 0.15 ≈ +4 pp win
    rate against an average-ability opponent; delta 0.5 ≈ +12 pp; delta 1.0 ≈ +23 pp.
    """
    t0_list = result.get("ranking_t0", [])
    t1_list = result.get("ranking_t1", [])
    if not t0_list or not t1_list:
        return

    t0 = pd.DataFrame(t0_list).set_index("value")
    t1 = pd.DataFrame(t1_list).set_index("value")
    common = sorted(set(t0.index) & set(t1.index), key=lambda v: t0.loc[v, "ability"])

    y = np.arange(len(common))
    height = 0.35

    fig, ax = plt.subplots(figsize=(8, max(4, len(common) * 0.55)))

    t0_ab = [t0.loc[v, "ability"] for v in common]
    t1_ab = [t1.loc[v, "ability"] for v in common]

    ax.barh(y - height / 2, t0_ab, height, label="T0 (no context)",
            color="steelblue", alpha=0.8)
    ax.barh(y + height / 2, t1_ab, height, label="T1 (post-conv)",
            color="darkorange", alpha=0.8)

    # 95% CIs
    for i, v in enumerate(common):
        if "ci_lower" in t0.columns and "ci_upper" in t0.columns:
            ax.errorbar(
                t0.loc[v, "ability"], y[i] - height / 2,
                xerr=[[t0.loc[v, "ability"] - t0.loc[v, "ci_lower"]],
                      [t0.loc[v, "ci_upper"] - t0.loc[v, "ability"]]],
                fmt="none", color="navy", capsize=3, linewidth=1.2,
            )
        if "ci_lower" in t1.columns and "ci_upper" in t1.columns:
            ax.errorbar(
                t1.loc[v, "ability"], y[i] + height / 2,
                xerr=[[t1.loc[v, "ability"] - t1.loc[v, "ci_lower"]],
                      [t1.loc[v, "ci_upper"] - t1.loc[v, "ability"]]],
                fmt="none", color="darkred", capsize=3, linewidth=1.2,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(common, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel(
        "BT ability (log-odds scale; Δ0.15 ≈ +4 pp win rate against avg opponent)",
        fontsize=8,
    )
    stance = result.get("stance", "neutral")
    mode   = result.get("mode", "mcq")
    ax.set_title(
        f"BT Abilities T0 vs T1 — {result['model']} · {result['value_set']} · "
        f"{stance} · {result['num_turns']} turns"
        + (f" · {mode}" if mode != "mcq" else ""),
        fontsize=9,
    )
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Cross-condition plots
# ---------------------------------------------------------------------------

def plot_ranking_heatmap_scenario(all_results: list[dict], output_path: Path):
    """Value rank heatmap across context lengths — analogue of ranking_heatmap in alignment exp.

    For each (value_set, mode) combination:
      - Panels: T0 baseline + one panel per num_turns in results
      - Rows: stances (or just the model name if only one stance)
      - Columns: values
      - Cell colour: rank (1=highest priority=lightest, N=lowest=darkest)
      - Cell text: rank number

    Mirrors the alignment experiment's Figure-4-style ranking_heatmap.
    """
    if not all_results:
        return

    from collections import defaultdict
    from matplotlib.colors import LinearSegmentedColormap
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["value_set"], r.get("mode", "mcq"))].append(r)

    cmap = LinearSegmentedColormap.from_list(
        "rank_cmap", ["#fff5f0", "#fcbba1", "#fb6a4a", "#cb181d", "#67000d"]
    )

    for (model, vs, mode), records in groups.items():
        turn_counts = sorted({r["num_turns"] for r in records})
        stances = sorted({r.get("stance", "neutral") for r in records},
                         key=lambda s: ["neutral", "pro_v1", "pro_v2"].index(s)
                         if s in ["neutral", "pro_v1", "pro_v2"] else 99)

        # Collect all values present
        values = sorted({v for r in records
                         for v in _ranking_list_to_dict(r["ranking_t0"]).keys()})
        n_values = len(values)
        if not values:
            continue

        panels = ["T0 (baseline)"] + [f"T1 @ {t}t" for t in turn_counts]
        n_panels = len(panels)

        fig, axes = plt.subplots(
            1, n_panels,
            figsize=(n_values * 1.3 * n_panels, len(stances) * 0.75 + 2),
            sharey=True,
        )
        if n_panels == 1:
            axes = [axes]

        for panel_idx, (ax, label) in enumerate(zip(axes, panels)):
            rank_matrix = np.full((len(stances), n_values), np.nan)
            for si, stance in enumerate(stances):
                if panel_idx == 0:
                    # T0 — same for all turns; pick any record with this stance
                    rec = next((r for r in records if r.get("stance", "neutral") == stance), None)
                    if rec is None:
                        continue
                    abilities = _ranking_list_to_dict(rec["ranking_t0"])
                else:
                    nt = turn_counts[panel_idx - 1]
                    rec = next(
                        (r for r in records
                         if r.get("stance", "neutral") == stance and r["num_turns"] == nt),
                        None,
                    )
                    if rec is None:
                        continue
                    abilities = _ranking_list_to_dict(rec["ranking_t1"])

                ranks = _ability_to_rank_scenario(abilities)
                for j, v in enumerate(values):
                    rank_matrix[si, j] = ranks.get(v, np.nan)

            ax.imshow(rank_matrix, cmap=cmap, aspect="auto", vmin=1, vmax=n_values)
            for si in range(len(stances)):
                for j in range(n_values):
                    val = rank_matrix[si, j]
                    if not np.isnan(val):
                        text_color = "white" if val > n_values * 0.6 else "black"
                        ax.text(j, si, f"{int(val)}", ha="center", va="center",
                                fontsize=9, color=text_color, fontweight="bold")

            ax.set_xticks(range(n_values))
            ax.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
            ax.set_title(label, fontsize=10, fontweight="bold")
            if panel_idx == 0:
                ax.set_yticks(range(len(stances)))
                ax.set_yticklabels(stances, fontsize=9)

        mode_tag = f"_{mode}" if mode != "mcq" else ""
        fig.suptitle(
            f"Value Rankings — {model} / {vs}{(' / ' + mode) if mode != 'mcq' else ''}\n"
            "(rank 1 = highest BT ability; lighter = higher priority)",
            fontsize=11, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        out = output_path.parent / f"{output_path.stem}_{model}_{vs}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_rank_shift_heatmap_scenario(all_results: list[dict], output_path: Path):
    """Rank-change heatmap from T0 baseline — analogue of rank_shift_heatmap.

    For each (value_set, mode):
      - Panels: one per num_turns
      - Rows: stances
      - Columns: values
      - Cell value: rank_T1 - rank_T0 (negative = value rose in priority)
      - Color: blue = rose, red = dropped (RdBu diverging)
    """
    if not all_results:
        return

    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["value_set"], r.get("mode", "mcq"))].append(r)

    for (model, vs, mode), records in groups.items():
        turn_counts = sorted({r["num_turns"] for r in records})
        stances = sorted({r.get("stance", "neutral") for r in records},
                         key=lambda s: ["neutral", "pro_v1", "pro_v2"].index(s)
                         if s in ["neutral", "pro_v1", "pro_v2"] else 99)
        values = sorted({v for r in records
                         for v in _ranking_list_to_dict(r["ranking_t0"]).keys()})
        if not values or not turn_counts:
            continue

        n_values = len(values)
        fig, axes = plt.subplots(
            1, len(turn_counts),
            figsize=(n_values * 1.3 * len(turn_counts), len(stances) * 0.75 + 2),
            sharey=True,
        )
        if len(turn_counts) == 1:
            axes = [axes]

        for ax, nt in zip(axes, turn_counts):
            delta_matrix = np.full((len(stances), n_values), np.nan)
            for si, stance in enumerate(stances):
                t0_rec = next((r for r in records if r.get("stance", "neutral") == stance), None)
                t1_rec = next(
                    (r for r in records
                     if r.get("stance", "neutral") == stance and r["num_turns"] == nt),
                    None,
                )
                if t0_rec is None or t1_rec is None:
                    continue
                t0_ranks = _ability_to_rank_scenario(_ranking_list_to_dict(t0_rec["ranking_t0"]))
                t1_ranks = _ability_to_rank_scenario(_ranking_list_to_dict(t1_rec["ranking_t1"]))
                for j, v in enumerate(values):
                    if v in t0_ranks and v in t1_ranks:
                        delta_matrix[si, j] = t1_ranks[v] - t0_ranks[v]

            max_abs = max(np.nanmax(np.abs(delta_matrix)) if not np.all(np.isnan(delta_matrix)) else 1, 1)
            im = ax.imshow(delta_matrix, cmap="RdBu", aspect="auto",
                           vmin=-max_abs, vmax=max_abs)
            for si in range(len(stances)):
                for j in range(n_values):
                    val = delta_matrix[si, j]
                    if not np.isnan(val):
                        sign = "+" if val > 0 else ""
                        ax.text(j, si, f"{sign}{int(val)}", ha="center", va="center",
                                fontsize=9, fontweight="bold")

            ax.set_xticks(range(n_values))
            ax.set_xticklabels(values, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"Rank change after {nt}t", fontsize=10, fontweight="bold")
            ax.set_yticks(range(len(stances)))
            ax.set_yticklabels(stances, fontsize=9)

        mode_tag = f"_{mode}" if mode != "mcq" else ""
        fig.suptitle(
            f"Value Rank Shifts from T0 — {model} / {vs}{(' / ' + mode) if mode != 'mcq' else ''}\n"
            "(blue = value rose in priority, red = dropped)",
            fontsize=11, fontweight="bold", y=1.02,
        )
        # Reserve right margin for colorbar so it never overlaps the matrix panels
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.018, 0.65])
        fig.colorbar(im, cax=cax, label="Rank change (−=rose, +=dropped)")
        out = output_path.parent / f"{output_path.stem}_{model}_{vs}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_radar_panel_scenario(all_results: list[dict], output_path: Path):
    """Multi-panel radar chart — analogue of radar_panel_*_shared/local.png.

    For each (value_set, mode): one subplot per (num_turns × stance) condition,
    each showing T0 vs T1 overlay on the same radar axes.

    Produces both shared-scale and local-scale variants.
    """
    if not all_results:
        return

    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["value_set"], r.get("mode", "mcq"))].append(r)

    for (model, vs, mode), records in groups.items():
        turn_counts = sorted({r["num_turns"] for r in records})
        stances = sorted({r.get("stance", "neutral") for r in records},
                         key=lambda s: ["neutral", "pro_v1", "pro_v2"].index(s)
                         if s in ["neutral", "pro_v1", "pro_v2"] else 99)
        values = sorted({v for r in records
                         for v in _ranking_list_to_dict(r["ranking_t0"]).keys()})
        if not values:
            continue

        # Build a flat list of (subplot_label, t0_dict, t1_dict)
        subplots = []
        for nt in turn_counts:
            for stance in stances:
                rec = next(
                    (r for r in records
                     if r["num_turns"] == nt and r.get("stance", "neutral") == stance),
                    None,
                )
                if rec is None:
                    continue
                label = f"{nt}t / {stance}" if len(stances) > 1 else f"{nt} turns"
                subplots.append((
                    label,
                    _ranking_list_to_dict(rec["ranking_t0"]),
                    _ranking_list_to_dict(rec["ranking_t1"]),
                ))

        if not subplots:
            continue

        n = len(subplots)
        n_cols = min(3, n)
        n_rows = (n + n_cols - 1) // n_cols
        angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
        angles += angles[:1]

        mode_tag = f"_{mode}" if mode != "mcq" else ""

        for scale_mode in ("shared", "local"):
            # Global shift so all scores > 0 for radar display
            if scale_mode == "shared":
                all_ab = [a for _, t0d, t1d in subplots
                          for a in list(t0d.values()) + list(t1d.values())]
                global_min = min(all_ab) if all_ab else 0
                global_shift = max(-global_min + 0.1, 0)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(5 * n_cols, 5 * n_rows),
                subplot_kw=dict(polar=True),
            )
            axes_flat = np.atleast_1d(axes).flatten()

            for idx, (label, t0d, t1d) in enumerate(subplots):
                ax = axes_flat[idx]
                if scale_mode == "local":
                    local_min = min(min(t0d.values()), min(t1d.values()))
                    shift = max(-local_min + 0.1, 0)
                else:
                    shift = global_shift

                t0_scores = [t0d.get(v, 0) + shift for v in values] + [t0d.get(values[0], 0) + shift]
                t1_scores = [t1d.get(v, 0) + shift for v in values] + [t1d.get(values[0], 0) + shift]

                ax.plot(angles, t0_scores, "o-", color="#4477AA", label="T0", linewidth=1.5, markersize=3)
                ax.plot(angles, t1_scores, "s--", color="#CC6677", label="T1", linewidth=1.5, markersize=3)
                ax.fill(angles, t0_scores, alpha=0.08, color="#4477AA")
                ax.fill(angles, t1_scores, alpha=0.08, color="#CC6677")
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(values, size=7)
                ax.set_title(label, size=10, pad=12)
                if idx == 0:
                    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.2), fontsize=7)

            for idx in range(n, n_rows * n_cols):
                axes_flat[idx].set_visible(False)

            scale_label = "shared-scale" if scale_mode == "shared" else "local-scale"
            fig.suptitle(
                f"BT Rankings T0 vs T1 — {model} / {vs}{(' / ' + mode) if mode != 'mcq' else ''} "
                f"({scale_label})",
                fontsize=11, fontweight="bold", y=1.02,
            )
            plt.tight_layout()
            out = (output_path.parent
                   / f"{output_path.stem}_{model}_{vs}{mode_tag}_{scale_mode}{output_path.suffix}")
            out.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()


def plot_drift_by_turns_scenario(all_results: list[dict], output_path: Path):
    """L2 drift vs num_turns, one line per value_set, faceted by stance×mode.

    Equivalent of plot_drift_by_turns for the scenario experiment.
    """
    if not all_results:
        return

    from collections import defaultdict
    # Group by (model, stance, mode)
    panels: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        panels[(r["model"], r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (model, stance, mode), records in panels.items():
        df = pd.DataFrame([{
            "value_set": r["value_set"],
            "num_turns": r["num_turns"],
            "l2_distance": r["drift"]["l2_distance"],
        } for r in records])
        if df.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        for vs, grp in df.groupby("value_set"):
            by_turns = grp.groupby("num_turns")["l2_distance"].mean()
            ax.plot(by_turns.index, by_turns.values, "o-", label=vs, linewidth=2)

        ax.set_xlabel("Number of Turns")
        ax.set_ylabel("L2 Drift (BT ability vector)")
        ax.set_title(f"Drift vs Conversation Length — {model} / stance={stance}, mode={mode}")
        ax.legend(title="Value Set")
        turns = sorted(df["num_turns"].unique())
        ax.set_xticks(turns)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}_{model}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_l2_heatmap_scenario(all_results: list[dict], output_path: Path):
    """Heatmap of L2 drift: value_set (rows) × num_turns (cols).

    One heatmap per (stance, mode). Equivalent of plot_drift_heatmap.
    """
    if not all_results:
        return

    from collections import defaultdict
    panels: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        panels[(r["model"], r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (model, stance, mode), records in panels.items():
        rows = [{
            "value_set": r["value_set"],
            "num_turns": r["num_turns"],
            "l2_distance": r["drift"]["l2_distance"],
            "flip_rate": r["flip_stats"]["overall_flip_rate"],
        } for r in records]
        df = pd.DataFrame(rows)
        if df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, max(3, len(df["value_set"].unique()) * 0.9)))
        for ax, metric, label in zip(
            axes,
            ["l2_distance", "flip_rate"],
            ["L2 Drift", "Overall Flip Rate"],
        ):
            try:
                pivot = df.pivot_table(values=metric, index="value_set", columns="num_turns", aggfunc="mean")
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, linewidths=0.4)
                ax.set_title(label)
                ax.set_ylabel("")
            except Exception:
                ax.set_visible(False)

        fig.suptitle(f"Drift Metrics — {model} / stance={stance}, mode={mode}", fontsize=11)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}_{model}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_stance_comparison(all_results: list[dict], output_path: Path):
    """Compare flip-toward rate per value across stances (neutral / pro_v1 / pro_v2).

    For each (value_set, num_turns, mode), plot grouped bars: one cluster per value,
    bars colored by stance. Shows whether biasing the conversation actually moves
    the model in the expected direction.
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["value_set"], r["num_turns"], r.get("mode", "mcq"))
        groups[key].append(r)

    for (model, vs, turns, mode), records in groups.items():
        if len({r.get("stance", "neutral") for r in records}) < 2:
            continue

        stances = sorted({r.get("stance", "neutral") for r in records},
                         key=lambda s: ["neutral", "pro_v1", "pro_v2"].index(s)
                         if s in ["neutral", "pro_v1", "pro_v2"] else 99)
        all_values = sorted({v for r in records for v in r.get("per_value_flip_stats", {})})
        if not all_values:
            continue

        x = np.arange(len(all_values))
        width = 0.8 / len(stances)
        stance_colors = {"neutral": "steelblue", "pro_v1": "seagreen", "pro_v2": "tomato"}

        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(all_values) * 1.4), 5),
                                 sharey=False)
        for ax, metric, ylabel in zip(
            axes,
            ["flip_rate_toward", "flip_rate_away"],
            ["Flip-toward rate", "Flip-away rate"],
        ):
            for si, stance in enumerate(stances):
                rec = next((r for r in records if r.get("stance", "neutral") == stance), None)
                if rec is None:
                    continue
                pvfs = rec.get("per_value_flip_stats", {})
                heights = [
                    0.0 if (h := pvfs.get(v, {}).get(metric)) is None
                    or (isinstance(h, float) and np.isnan(h)) else float(h)
                    for v in all_values
                ]
                offset = (si - (len(stances) - 1) / 2) * width
                ax.bar(x + offset, heights, width, label=stance,
                       color=stance_colors.get(stance, "gray"), alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.legend(title="Stance", fontsize=8)

        fig.suptitle(f"Stance Comparison — {model} / {vs} / {turns}t / {mode}", fontsize=10)

        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}_{model}_{vs}_{turns}t{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def _agg_drift_ax(ax, value_deltas: dict, title: str):
    """Draw an aggregated drift bar (mean ± SD) on *ax*."""
    from collections import defaultdict
    values = sorted(value_deltas, key=lambda v: np.mean(value_deltas[v]))
    means  = [np.mean(value_deltas[v]) for v in values]
    sds    = [np.std(value_deltas[v])  for v in values]
    colors = ["steelblue" if m >= 0 else "tomato" for m in means]
    ax.barh(values, means, xerr=sds, color=colors, alpha=0.85,
            error_kw=dict(ecolor="black", capsize=3, linewidth=0.9))
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean BT delta (T1 − T0)  ±1 SD")
    ax.set_title(title, fontsize=9)


def plot_aggregated_drift_bars(all_results: list[dict], output_path: Path):
    """Turn-count-averaged per-value BT delta.

    One file per (model, value_set, stance, mode).
    For non-neutral stances, shows two panels:
      Left  — Overall: global drift averaged over turn counts
      Right — Role-filtered: delta from favoured-role scenarios only
    Error bars = ±1 SD across turn counts.
    """
    if not all_results:
        return

    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["value_set"], r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (model, vs, stance, mode), records in groups.items():
        overall_deltas:  dict[str, list[float]] = defaultdict(list)
        filtered_deltas: dict[str, list[float]] = defaultdict(list)
        for r in records:
            for v, d in r.get("drift", {}).get("per_value_delta", {}).items():
                overall_deltas[v].append(d)
            rf = r.get("role_filtered_drift", {}).get("per_value_delta", {})
            for v, d in rf.items():
                filtered_deltas[v].append(d)

        if not overall_deltas:
            continue

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag   = f"_{mode}"   if mode   != "mcq"     else ""
        n_cond     = len(records)
        base       = f"{model} · {vs} · {stance}" + (f" · {mode}" if mode != "mcq" else "") \
                     + f"  ({n_cond} turn condition{'s' if n_cond != 1 else ''})"

        if stance == "neutral" or not filtered_deltas:
            n_vals = len(overall_deltas)
            fig, ax = plt.subplots(figsize=(7, max(3.5, n_vals * 0.55 + 1.0)))
            _agg_drift_ax(ax, overall_deltas, f"Aggregated Value Drift — {base}")
        else:
            n_vals = max(len(overall_deltas), len(filtered_deltas))
            fig, (ax_l, ax_r) = plt.subplots(
                1, 2, figsize=(13, max(3.5, n_vals * 0.55 + 1.0))
            )
            _agg_drift_ax(ax_l, overall_deltas,
                          f"Overall (all pairs) — {base}")
            _agg_drift_ax(ax_r, filtered_deltas,
                          f"Role-filtered (favoured role only) — {base}")
            fig.suptitle(f"Aggregated Value Drift — {base}", fontsize=10, fontweight="bold")

        out = output_path.parent / f"{output_path.stem}_{model}_{vs}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def _agg_flip_ax(ax, toward_all: dict, away_all: dict, title: str):
    """Draw aggregated flip-toward/away grouped bars (mean ± SD) on *ax*."""
    all_values   = sorted(set(toward_all) | set(away_all))
    toward_means = [np.mean(toward_all[v]) if v in toward_all else 0.0 for v in all_values]
    toward_sds   = [np.std(toward_all[v])  if v in toward_all else 0.0 for v in all_values]
    away_means   = [np.mean(away_all[v])   if v in away_all   else 0.0 for v in all_values]
    away_sds     = [np.std(away_all[v])    if v in away_all   else 0.0 for v in all_values]

    x   = np.arange(len(all_values))
    w   = 0.35
    ekw = dict(ecolor="black", capsize=3, linewidth=0.9)
    b_t = ax.bar(x - w / 2, toward_means, w, yerr=toward_sds,
                 label="Flip toward", color="steelblue", alpha=0.85, error_kw=ekw)
    b_a = ax.bar(x + w / 2, away_means,   w, yerr=away_sds,
                 label="Flip away",   color="tomato",    alpha=0.85, error_kw=ekw)
    ax.set_xticks(x)
    ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Mean rate  ±1 SD")
    peak = float(np.nanmax(toward_means + away_means)) if any(v > 0 for v in toward_means + away_means) else 0.1
    ax.set_ylim(0, max(peak * 1.35, 0.05))
    ax.bar_label(b_t, fmt="%.2f", fontsize=7, padding=2)
    ax.bar_label(b_a, fmt="%.2f", fontsize=7, padding=2)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=9)


def plot_aggregated_per_value_flip_rate(all_results: list[dict], output_path: Path):
    """Turn-count-averaged per-value flip-toward / flip-away rates.

    One file per (model, value_set, stance, mode).  Error bars = ±1 SD.
    For non-neutral stances, shows two panels:
      Left  — Overall: all appearances (per_value_flip_stats_overall)
      Right — Role-filtered: appearances in favoured role only (per_value_flip_stats)
    """
    if not all_results:
        return

    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        groups[(r["model"], r["value_set"], r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (model, vs, stance, mode), records in groups.items():
        def _safe(x):
            return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)

        # Collect from overall (unfiltered) and from role-filtered sources
        toward_overall: dict[str, list[float]] = defaultdict(list)
        away_overall:   dict[str, list[float]] = defaultdict(list)
        toward_filt:    dict[str, list[float]] = defaultdict(list)
        away_filt:      dict[str, list[float]] = defaultdict(list)

        for r in records:
            for v, stats in r.get("per_value_flip_stats_overall", r.get("per_value_flip_stats", {})).items():
                toward_overall[v].append(_safe(stats.get("flip_rate_toward")))
                away_overall[v].append(_safe(stats.get("flip_rate_away")))
            for v, stats in r.get("per_value_flip_stats", {}).items():
                toward_filt[v].append(_safe(stats.get("flip_rate_toward")))
                away_filt[v].append(_safe(stats.get("flip_rate_away")))

        if not toward_overall:
            continue

        n_cond     = len(records)
        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag   = f"_{mode}"   if mode   != "mcq"     else ""
        base       = f"{model} · {vs} · {stance}" + (f" · {mode}" if mode != "mcq" else "") \
                     + f"  ({n_cond} turn condition{'s' if n_cond != 1 else ''})"
        n_vals     = len(toward_overall)

        if stance == "neutral" or toward_overall == toward_filt:
            fig, ax = plt.subplots(figsize=(max(7, n_vals * 0.9), 5))
            _agg_flip_ax(ax, toward_overall, away_overall,
                         f"Aggregated Flip Rates — {base}")
        else:
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(max(14, n_vals * 1.8), 5),
                                              sharey=True)
            _agg_flip_ax(ax_l, toward_overall, away_overall,
                         f"Overall (all pairs) — {base}")
            _agg_flip_ax(ax_r, toward_filt,    away_filt,
                         f"Role-filtered (favoured role only) — {base}")
            fig.suptitle(f"Aggregated Flip Rates — {base}", fontsize=10, fontweight="bold")

        out = output_path.parent / f"{output_path.stem}_{model}_{vs}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_model_comparison(all_results: list[dict], output_path: Path):
    """Compare per-value BT delta and flip rates across models.

    For each (value_set, stance, num_turns, mode), one grouped-bar chart per metric:
      - BT delta (role_filtered_drift for non-neutral stances, else global drift)
      - flip-toward rate and flip-away rate (role-filtered for non-neutral stances)

    One file per (value_set, stance, num_turns, mode) combination.
    Skipped if fewer than 2 models are present for a given combination.
    """
    if not all_results:
        return

    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["value_set"], r.get("stance", "neutral"), r["num_turns"], r.get("mode", "mcq"))
        groups[key].append(r)

    model_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for (vs, stance, turns, mode), records in groups.items():
        models = sorted({r["model"] for r in records})
        if len(models) < 2:
            continue

        all_values = sorted({
            v for r in records
            for v in (
                r.get("role_filtered_drift", {}).get("per_value_delta", {})
                if stance != "neutral" and r.get("role_filtered_drift", {}).get("per_value_delta")
                else r.get("drift", {}).get("per_value_delta", {})
            )
        })
        if not all_values:
            continue

        x = np.arange(len(all_values))
        width = 0.8 / len(models)
        color_map = {m: model_colors[i % 10] for i, m in enumerate(models)}

        role_note = f" [role-filtered: {stance}]" if stance != "neutral" else ""
        fig, axes = plt.subplots(1, 3, figsize=(max(12, len(all_values) * 1.8), 5))

        metrics = [
            ("bt_delta", "BT score delta (T1−T0)"),
            ("flip_rate_toward", "Flip-toward rate"),
            ("flip_rate_away", "Flip-away rate"),
        ]
        for ax, (metric_key, ylabel) in zip(axes, metrics):
            for mi, model in enumerate(models):
                rec = next((r for r in records if r["model"] == model), None)
                if rec is None:
                    continue
                def _h(x):
                    return 0.0 if (x is None or (isinstance(x, float) and np.isnan(x))) else float(x)
                if metric_key == "bt_delta":
                    rf = rec.get("role_filtered_drift", {})
                    source = (
                        rf["per_value_delta"]
                        if stance != "neutral" and rf.get("per_value_delta")
                        else rec.get("drift", {}).get("per_value_delta", {})
                    )
                    heights = [_h(source.get(v)) for v in all_values]
                else:
                    pvfs = rec.get("per_value_flip_stats", {})
                    heights = [_h(pvfs.get(v, {}).get(metric_key)) for v in all_values]
                offset = (mi - (len(models) - 1) / 2) * width
                ax.bar(x + offset, heights, width, label=model,
                       color=color_map[model], alpha=0.85)
            if metric_key == "bt_delta":
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(title="Model", fontsize=7)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        fig.suptitle(
            f"Model Comparison — {vs} / {stance} / {turns}t"
            f"{(' / ' + mode) if mode != 'mcq' else ''}{role_note}",
            fontsize=10, fontweight="bold",
        )
        out = (output_path.parent
               / f"{output_path.stem}_{vs}{stance_tag}_{turns}t{mode_tag}{output_path.suffix}")
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_mode_comparison(all_results: list[dict], output_path: Path):
    """Compare per-value flip rates between MCQ and open-ended probing modes.

    For each (value_set, num_turns, stance), plot grouped bars: one cluster per
    value, bars colored by mode. Shows whether free-form responses reveal stronger
    or different drifts than forced-choice MCQ.
    """
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["model"], r["value_set"], r["num_turns"], r.get("stance", "neutral"))
        groups[key].append(r)

    for (model, vs, turns, stance), records in groups.items():
        if len({r.get("mode", "mcq") for r in records}) < 2:
            continue

        modes = sorted({r.get("mode", "mcq") for r in records})
        all_values = sorted({v for r in records for v in r.get("per_value_flip_stats", {})})
        if not all_values:
            continue

        x = np.arange(len(all_values))
        width = 0.8 / len(modes)
        mode_colors = {"mcq": "steelblue", "openended": "darkorange"}

        fig, axes = plt.subplots(1, 2, figsize=(max(10, len(all_values) * 1.4), 5),
                                 sharey=False)
        for ax, metric, ylabel in zip(
            axes,
            ["flip_rate_toward", "flip_rate_away"],
            ["Flip-toward rate", "Flip-away rate"],
        ):
            for mi, mode in enumerate(modes):
                rec = next((r for r in records if r.get("mode", "mcq") == mode), None)
                if rec is None:
                    continue
                pvfs = rec.get("per_value_flip_stats", {})
                heights = [
                    0.0 if (h := pvfs.get(v, {}).get(metric)) is None
                    or (isinstance(h, float) and np.isnan(h)) else float(h)
                    for v in all_values
                ]
                offset = (mi - (len(modes) - 1) / 2) * width
                ax.bar(x + offset, heights, width, label=mode,
                       color=mode_colors.get(mode, "gray"), alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.legend(title="Mode", fontsize=8)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        fig.suptitle(f"MCQ vs Open-Ended — {model} / {vs} / {turns}t{stance_tag}", fontsize=10)

        out = output_path.parent / f"{output_path.stem}_{model}_{vs}_{turns}t{stance_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
