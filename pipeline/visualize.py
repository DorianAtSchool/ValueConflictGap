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
    Creates one sub-folder per (model, value_set, num_turns) condition.
    """
    if not all_results:
        return

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for r in all_results:
        tag = f"{r['model']}_{r['value_set']}_{r['num_turns']}t"
        cond_dir = plots_dir / tag
        cond_dir.mkdir(parents=True, exist_ok=True)

        plot_pair_consistency_heatmap(r, cond_dir / "pair_consistency.png")
        plot_per_value_flip_stats(r, cond_dir / "per_value_flip_rates.png")
        plot_scenario_radar(r, cond_dir / "radar_t0_t1.png")
        plot_scenario_drift_bars(r, cond_dir / "drift_bars.png")
