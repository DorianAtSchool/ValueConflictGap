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
