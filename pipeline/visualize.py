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

    Rows = value pairs, two columns: flip_rate and directional_consistency.
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
    # Short labels: "auto vs harm"
    short = [p.replace("_vs_", " vs ") for p in pairs]

    data = pd.DataFrame({
        "flip_rate": flip_rates,
        "directional_consistency": consistencies,
    }, index=short)

    fig, ax = plt.subplots(figsize=(5, max(4, len(pairs) * 0.4)))
    sns.heatmap(
        data, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title(
        f"Pair Flip Stats — {result['model']} / {result['value_set']} / {result['num_turns']}t",
        fontsize=10,
    )
    # Annotate dominant value per row
    for i, dom in enumerate(dominant):
        ax.text(2.05, i + 0.5, dom, va="center", fontsize=7, color="navy")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_per_value_flip_stats(result: dict, output_path: Path):
    """Grouped bar chart: for each value, flip-toward-rate and flip-away-rate.

    Answers: "in all scenarios where value X appeared, what % flipped
    toward X and what % flipped away from X?"
    """
    pvfs = result.get("per_value_flip_stats", {})
    if not pvfs:
        return

    values = sorted(pvfs.keys())
    toward = [pvfs[v]["flip_rate_toward"] for v in values]
    away = [pvfs[v]["flip_rate_away"] for v in values]

    x = np.arange(len(values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, len(values) * 0.8), 5))
    bars_toward = ax.bar(x - width / 2, toward, width, label="Flip toward", color="steelblue")
    bars_away = ax.bar(x + width / 2, away, width, label="Flip away", color="tomato")

    ax.set_xticks(x)
    ax.set_xticklabels(values, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Rate (out of all scenarios value appeared in)")
    ax.set_ylim(0, max(max(toward + away, default=0) * 1.25, 0.05))
    ax.set_title(
        f"Per-Value Flip Rates — {result['model']} / {result['value_set']} / {result['num_turns']}t",
        fontsize=10,
    )
    ax.legend()
    ax.bar_label(bars_toward, fmt="%.2f", fontsize=7)
    ax.bar_label(bars_away, fmt="%.2f", fontsize=7)

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


def plot_scenario_drift_bars(result: dict, output_path: Path):
    """Horizontal bar chart of per-value BT score delta (T1 - T0)."""
    delta = result.get("drift", {}).get("per_value_delta", {})
    if not delta:
        return

    values = sorted(delta.keys(), key=lambda v: delta[v])
    deltas = [delta[v] for v in values]
    colors = ["steelblue" if d >= 0 else "tomato" for d in deltas]

    fig, ax = plt.subplots(figsize=(6, max(3, len(values) * 0.4)))
    ax.barh(values, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("BT score delta (T1 − T0)")
    ax.set_title(
        f"Value Drift — {result['model']} / {result['value_set']} / {result['num_turns']}t",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_scenario_experiment_plots(all_results: list[dict], results_dir: Path):
    """Generate all plots for the scenario conversation experiment.

    Called at the end of run_scenario_conversation_experiment.py.

    Per-condition plots (one per result record):
      - radar_t0_t1.png           spider chart of BT scores before/after
      - bt_ranking_bars.png       absolute BT abilities T0 vs T1 with 95% CIs
      - drift_bars.png            per-value BT score delta bar chart
      - pair_consistency.png      heatmap: pairs × (flip_rate, directional_consistency)
      - per_value_flip_rates.png  grouped bars: flip-toward vs flip-away per value

    Cross-condition plots (aggregated over all results):
      - trajectories_pca.png      PCA arrows T0→T1, colored by num_turns
      - drift_by_turns.png        L2 drift vs num_turns, faceted by value_set
      - l2_heatmap.png            L2 drift heatmap: value_set × num_turns
      - attractor.png             PCA of T1 vectors, colored by value_set
      - stance_comparison.png     per-value flip-toward rate by stance (if >1 stance)
      - mode_comparison.png       per-value flip-toward rate MCQ vs open-ended (if >1 mode)
    """
    if not all_results:
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-condition plots ---
    for r in all_results:
        stance = r.get("stance", "neutral")
        mode = r.get("mode", "mcq")
        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        tag = f"{r['model']}_{r['value_set']}_{r['num_turns']}t{stance_tag}{mode_tag}"
        cond_dir = plots_dir / tag
        cond_dir.mkdir(parents=True, exist_ok=True)

        plot_pair_consistency_heatmap(r, cond_dir / "pair_consistency.png")
        plot_per_value_flip_stats(r, cond_dir / "per_value_flip_rates.png")
        plot_scenario_radar(r, cond_dir / "radar_t0_t1.png")
        plot_scenario_drift_bars(r, cond_dir / "drift_bars.png")
        plot_bt_ranking_bars(r, cond_dir / "bt_ranking_bars.png")

    # --- Cross-condition plots ---
    plot_trajectories_pca_scenario(all_results, plots_dir / "trajectories_pca.png")
    plot_drift_by_turns_scenario(all_results, plots_dir / "drift_by_turns.png")
    plot_l2_heatmap_scenario(all_results, plots_dir / "l2_heatmap.png")
    plot_attractor_scenario(all_results, plots_dir / "attractor.png")

    stances_present = {r.get("stance", "neutral") for r in all_results}
    if len(stances_present) > 1:
        plot_stance_comparison(all_results, plots_dir / "stance_comparison.png")

    modes_present = {r.get("mode", "mcq") for r in all_results}
    if len(modes_present) > 1:
        plot_mode_comparison(all_results, plots_dir / "mode_comparison.png")


# ---------------------------------------------------------------------------
# Per-condition helpers
# ---------------------------------------------------------------------------

def _ranking_list_to_dict(ranking_list: list[dict]) -> dict[str, float]:
    """Convert [{value, ability, ...}, ...] to {value: ability}."""
    return {r["value"]: r["ability"] for r in ranking_list}


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

    bars_t0 = ax.barh(y - height / 2, t0_ab, height, label="T0 (no context)",
                      color="steelblue", alpha=0.8)
    bars_t1 = ax.barh(y + height / 2, t1_ab, height, label="T1 (post-conv)",
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
    mode = result.get("mode", "mcq")
    ax.set_title(
        f"BT Rankings — {result['model']} / {result['value_set']} / "
        f"{result['num_turns']}t / {stance} / {mode}",
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

def plot_trajectories_pca_scenario(all_results: list[dict], output_path: Path):
    """PCA trajectory arrows T0→T1 in value-space, colored by num_turns.

    One plot per (value_set, stance, mode). Equivalent of plot_trajectories_pca
    for the scenario experiment where the axis of variation is num_turns.
    """
    if not all_results:
        return

    # Group by (value_set, stance, mode)
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        key = (r["value_set"], r.get("stance", "neutral"), r.get("mode", "mcq"))
        groups[key].append(r)

    for (vs, stance, mode), records in groups.items():
        t0_vecs, t1_vecs, turn_labels = [], [], []
        for r in records:
            t0d = _ranking_list_to_dict(r["ranking_t0"])
            t1d = _ranking_list_to_dict(r["ranking_t1"])
            vals = sorted(set(t0d) & set(t1d))
            if not vals:
                continue
            t0_vecs.append([t0d[v] for v in vals])
            t1_vecs.append([t1d[v] for v in vals])
            turn_labels.append(r["num_turns"])

        if len(t0_vecs) < 2:
            continue

        all_vecs = np.array(t0_vecs + t1_vecs)
        n_comp = min(2, all_vecs.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(all_vecs)
        n = len(t0_vecs)
        t0_proj, t1_proj = proj[:n], proj[n:]

        unique_turns = sorted(set(turn_labels))
        cmap = plt.cm.viridis(np.linspace(0.2, 0.9, max(len(unique_turns), 1)))
        turn_color = {t: cmap[i] for i, t in enumerate(unique_turns)}

        fig, ax = plt.subplots(figsize=(9, 7))
        for i in range(n):
            c = turn_color[turn_labels[i]]
            ax.annotate(
                "",
                xy=t1_proj[i], xytext=t0_proj[i],
                arrowprops=dict(arrowstyle="->", color=c, lw=2),
            )
            ax.scatter(*t0_proj[i], c=[c], marker="o", s=60, zorder=5)
            ax.scatter(*t1_proj[i], c=[c], marker="s", s=60, zorder=5)

        for t in unique_turns:
            ax.scatter([], [], c=[turn_color[t]], marker="o", label=f"{t} turns")
        ax.scatter([], [], marker="o", c="gray", label="T0", s=60)
        ax.scatter([], [], marker="s", c="gray", label="T1", s=60)
        ax.legend(title="Turns", fontsize=8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        if n_comp > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"Value Space Trajectories — {vs} / {stance} / {mode}")

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}_{vs}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_drift_by_turns_scenario(all_results: list[dict], output_path: Path):
    """L2 drift vs num_turns, one line per value_set, faceted by stance×mode.

    Equivalent of plot_drift_by_turns for the scenario experiment.
    """
    if not all_results:
        return

    from collections import defaultdict
    # Group by (stance, mode)
    panels: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        panels[(r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (stance, mode), records in panels.items():
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
        ax.set_title(f"Drift vs Conversation Length — stance={stance}, mode={mode}")
        ax.legend(title="Value Set")
        turns = sorted(df["num_turns"].unique())
        ax.set_xticks(turns)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}{stance_tag}{mode_tag}{output_path.suffix}"
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
        panels[(r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (stance, mode), records in panels.items():
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

        fig.suptitle(f"Drift Metrics — stance={stance}, mode={mode}", fontsize=11)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}{stance_tag}{mode_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()


def plot_attractor_scenario(all_results: list[dict], output_path: Path):
    """Cluster T1 BT vectors in PCA space; colored by value_set.

    One plot per (stance, mode). Equivalent of plot_attractor_analysis.
    """
    if len(all_results) < 3:
        return

    from collections import defaultdict
    panels: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_results:
        panels[(r.get("stance", "neutral"), r.get("mode", "mcq"))].append(r)

    for (stance, mode), records in panels.items():
        t1_vecs, labels = [], []
        for r in records:
            t1d = _ranking_list_to_dict(r["ranking_t1"])
            vals = sorted(t1d)
            t1_vecs.append([t1d[v] for v in vals])
            labels.append(f"{r['value_set']} / {r['num_turns']}t")

        if len(t1_vecs) < 3:
            continue

        arr = np.array(t1_vecs)
        n_comp = min(2, arr.shape[1])
        pca = PCA(n_components=n_comp)
        proj = pca.fit_transform(arr)

        n_clusters = min(4, len(proj))
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_ids = km.fit_predict(proj)

        value_sets = [r["value_set"] for r in records]
        unique_vs = sorted(set(value_sets))
        cmap = plt.cm.tab10(np.linspace(0, 1, max(len(unique_vs), 1)))
        vs_color = {vs: cmap[i] for i, vs in enumerate(unique_vs)}

        fig, ax = plt.subplots(figsize=(10, 7))
        for i, label in enumerate(labels):
            vs = records[i]["value_set"]
            ax.scatter(
                proj[i, 0], proj[i, 1] if n_comp > 1 else 0,
                c=[vs_color[vs]], s=70, zorder=5,
            )
            ax.annotate(str(records[i]["num_turns"]) + "t",
                        (proj[i, 0], proj[i, 1] if n_comp > 1 else 0),
                        fontsize=7, alpha=0.7)

        ax.scatter(
            km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1] if n_comp > 1 else np.zeros(n_clusters),
            marker="*", s=250, c="red", label="Centroids", zorder=10,
        )
        for vs in unique_vs:
            ax.scatter([], [], c=[vs_color[vs]], label=vs)
        ax.legend(title="Value Set", fontsize=8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        if n_comp > 1:
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"T1 Attractor Analysis — stance={stance}, mode={mode}")

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}{stance_tag}{mode_tag}{output_path.suffix}"
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
        key = (r["value_set"], r["num_turns"], r.get("mode", "mcq"))
        groups[key].append(r)

    for (vs, turns, mode), records in groups.items():
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
                heights = [pvfs.get(v, {}).get(metric, 0) or 0 for v in all_values]
                offset = (si - (len(stances) - 1) / 2) * width
                ax.bar(x + offset, heights, width, label=stance,
                       color=stance_colors.get(stance, "gray"), alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.legend(title="Stance", fontsize=8)

        fig.suptitle(f"Stance Comparison — {vs} / {turns}t / {mode}", fontsize=10)

        mode_tag = f"_{mode}" if mode != "mcq" else ""
        out = output_path.parent / f"{output_path.stem}_{vs}_{turns}t{mode_tag}{output_path.suffix}"
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
        key = (r["value_set"], r["num_turns"], r.get("stance", "neutral"))
        groups[key].append(r)

    for (vs, turns, stance), records in groups.items():
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
                heights = [pvfs.get(v, {}).get(metric, 0) or 0 for v in all_values]
                offset = (mi - (len(modes) - 1) / 2) * width
                ax.bar(x + offset, heights, width, label=mode,
                       color=mode_colors.get(mode, "gray"), alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(all_values, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel)
            ax.legend(title="Mode", fontsize=8)

        stance_tag = f"_{stance}" if stance != "neutral" else ""
        fig.suptitle(f"MCQ vs Open-Ended — {vs} / {turns}t{stance_tag}", fontsize=10)

        out = output_path.parent / f"{output_path.stem}_{vs}_{turns}t{stance_tag}{output_path.suffix}"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
