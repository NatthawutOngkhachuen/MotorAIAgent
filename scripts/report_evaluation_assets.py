from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from app.services.recommendation.recommenders.user_based import UserBasedRecommender
from app.services.recommendation.vectorizer import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "report_evaluation"
VECTOR_CSV = ROOT / "data" / "recommendation_output" / "final" / "user_preference_vectors_expanded.csv"
ITEMS_CSV = ROOT / "data" / "Items_Feature.csv"


def plot_k_selection(k_scores: pd.DataFrame) -> Path:
    path = OUT / "silhouette_k_selection.png"
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(k_scores["k"], k_scores["silhouette"], marker="o", linewidth=2.4, color="#2563EB")
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Silhouette Score", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.grid(axis="y", linestyle="--", alpha=0.35)

    ax2 = ax1.twinx()
    ax2.plot(k_scores["k"], k_scores["inertia"], marker="s", linewidth=2.0, color="#F97316")
    ax2.set_ylabel("Inertia", color="#F97316")
    ax2.tick_params(axis="y", labelcolor="#F97316")

    best = k_scores.sort_values("silhouette", ascending=False).iloc[0]
    ax1.scatter([best["k"]], [best["silhouette"]], s=110, color="#DC2626", zorder=5)
    ax1.annotate(
        f"Best K={int(best['k'])}\nSilhouette={best['silhouette']:.4f}",
        xy=(best["k"], best["silhouette"]),
        xytext=(best["k"] + 0.35, best["silhouette"] + 0.01),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    fig.suptitle("K-Means Cluster Selection by Silhouette Score")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_cluster_grid(vectors: pd.DataFrame, k_scores: pd.DataFrame) -> Path:
    path = OUT / "cluster_scatter_all_k.png"
    x = vectors[FEATURE_NAMES].astype(float).to_numpy()
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(x)

    ks = list(k_scores["k"].astype(int))
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    axes = axes.flatten()
    cmap = plt.get_cmap("tab10")

    for ax, k in zip(axes, ks):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(x)
        centroid_xy = pca.transform(model.cluster_centers_)
        sil = float(k_scores.loc[k_scores["k"] == k, "silhouette"].iloc[0])

        for cluster_id in range(k):
            part = xy[labels == cluster_id]
            ax.scatter(part[:, 0], part[:, 1], s=22, alpha=0.78, color=cmap(cluster_id % 10))
        ax.scatter(centroid_xy[:, 0], centroid_xy[:, 1], s=135, marker="X", c="#111827")
        for cluster_id, point in enumerate(centroid_xy):
            ax.annotate(f"C{cluster_id}", (point[0], point[1]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(f"K={k}, silhouette={sil:.4f}", fontsize=10)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(alpha=0.18)

    for ax in axes[len(ks):]:
        ax.axis("off")
    fig.suptitle("Cluster Scatter Comparison with Centroids (K=2 to K=8)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_best_cluster(vectors: pd.DataFrame, best_k: int) -> Path:
    path = OUT / "best_cluster_centroids.png"
    x = vectors[FEATURE_NAMES].astype(float).to_numpy()
    model = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    labels = model.fit_predict(x)
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(x)
    centroid_xy = pca.transform(model.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    cmap = plt.get_cmap("tab10")
    for cluster_id in range(best_k):
        part = xy[labels == cluster_id]
        ax.scatter(part[:, 0], part[:, 1], s=55, alpha=0.85, label=f"Cluster {cluster_id}", color=cmap(cluster_id))
    ax.scatter(centroid_xy[:, 0], centroid_xy[:, 1], s=230, marker="X", c="#111827", label="Centroid")
    for cluster_id, point in enumerate(centroid_xy):
        ax.annotate(f"C{cluster_id}", (point[0], point[1]), xytext=(7, 7), textcoords="offset points", fontsize=10)
    ax.set_title(f"Best User Preference Clusters (K={best_k})")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_user_scores() -> Path:
    path = OUT / "user_satisfaction_scores.png"
    labels = [
        "Ease of use",
        "Recommendation accuracy",
        "Overall satisfaction",
        "UI design",
        "Response speed",
    ]
    scores = [3.916666667, 4.25, 4.0, 3.916666667, 3.416666667]
    colors = ["#EF4444", "#F59E0B", "#3B82F6", "#F97316", "#16A34A"]
    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    y = np.arange(len(labels))
    ax.barh(y, scores, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xlabel("Average score (5-point scale)")
    ax.set_title("Average User Evaluation Scores")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(scores):
        ax.text(value + 0.05, i, f"{value:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build_recommendation_examples() -> dict:
    rec = UserBasedRecommender(
        vector_path=ROOT / "data" / "recommendation_output" / "final" / "user_preference_vectors_expanded.csv",
        cluster_path=ROOT / "data" / "recommendation_output" / "final" / "user_preference_clusters.csv",
        centroid_path=ROOT / "data" / "recommendation_output" / "final" / "user_preference_cluster_centroids.csv",
        items_path=ITEMS_CSV,
    )
    preference = {
        "budget_level": "low",
        "usage_fit": ["city", "daily"],
        "style": ["modern", "sporty"],
        "performance": "high",
        "comfort": "medium",
        "safety_level": "medium",
        "technology_level": "medium",
        "easy_to_ride": True,
        "fuel_saving": True,
        "storage_need": False,
        "maintenance_easy": True,
    }
    user_based = rec.recommend_nearest_user(preference, top_k=3)
    cluster_based = rec.recommend_cluster(preference, top_k=5)
    return {
        "input_preference": preference,
        "user_based": user_based,
        "cluster_based": cluster_based,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vectors = pd.read_csv(VECTOR_CSV)
    x = vectors[FEATURE_NAMES].astype(float).to_numpy()
    rows = []
    for k in range(2, min(8, len(vectors) - 1) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(x)
        rows.append({"k": k, "silhouette": float(silhouette_score(x, labels)), "inertia": float(model.inertia_)})
    k_scores = pd.DataFrame(rows)
    best_k = int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"])

    existing_profile = pd.read_csv(ROOT / "data" / "recommendation_output" / "final" / "user_preference_cluster_profile.csv")
    examples = build_recommendation_examples()
    assets = {
        "k_scores": k_scores.to_dict(orient="records"),
        "best_k": best_k,
        "cluster_profile": existing_profile.to_dict(orient="records"),
        "figures": {
            "silhouette": str(plot_k_selection(k_scores)),
            "all_k_scatter": str(plot_cluster_grid(vectors, k_scores)),
            "best_cluster": str(plot_best_cluster(vectors, best_k)),
            "user_scores": str(plot_user_scores()),
        },
        "examples": examples,
    }
    (OUT / "evaluation_summary.json").write_text(json.dumps(assets, ensure_ascii=False, indent=2), encoding="utf-8")
    k_scores.to_csv(OUT / "k_selection_recomputed.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(assets, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
