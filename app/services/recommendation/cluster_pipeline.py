from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.services.recommendation.vectorizer import FEATURE_NAMES


def choose_best_k(x, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    rows = []
    max_k = min(k_max, len(x) - 1)
    for k in range(k_min, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=30)
        labels = model.fit_predict(x)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "inertia": float(model.inertia_),
            }
        )
    return pd.DataFrame(rows)


def run_user_vector_clustering(
    vector_csv: Path,
    output_dir: Path,
    k_min: int = 2,
    k_max: int = 8,
) -> dict[str, Path | int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors = pd.read_csv(vector_csv)
    x = vectors[FEATURE_NAMES].astype(float).to_numpy()

    k_scores = choose_best_k(x, k_min=k_min, k_max=k_max)
    best_k = int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    model = KMeans(n_clusters=best_k, random_state=42, n_init=30)
    labels = model.fit_predict(x)

    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(x)
    centroid_xy = pca.transform(model.cluster_centers_)

    clustered = vectors.copy()
    clustered["cluster"] = labels
    clustered["pca_x"] = xy[:, 0]
    clustered["pca_y"] = xy[:, 1]

    centroids = pd.DataFrame(model.cluster_centers_, columns=FEATURE_NAMES)
    centroids.insert(0, "cluster", range(best_k))
    centroids["pca_x"] = centroid_xy[:, 0]
    centroids["pca_y"] = centroid_xy[:, 1]

    profile = (
        clustered.groupby("cluster")
        .agg(
            n_users=("user_id", "count"),
            top_models=("mapped_model", lambda s: ", ".join(s.value_counts().head(3).index.astype(str))),
            item_ids=("item_id", lambda s: ",".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
    )

    paths = {
        "k_selection": output_dir / "user_preference_k_selection.csv",
        "clusters": output_dir / "user_preference_clusters.csv",
        "centroids": output_dir / "user_preference_cluster_centroids.csv",
        "profile": output_dir / "user_preference_cluster_profile.csv",
        "plot": output_dir / "user_preference_clusters.png",
    }
    k_scores.to_csv(paths["k_selection"], index=False, encoding="utf-8-sig")
    clustered.to_csv(paths["clusters"], index=False, encoding="utf-8-sig")
    centroids.to_csv(paths["centroids"], index=False, encoding="utf-8-sig")
    profile.to_csv(paths["profile"], index=False, encoding="utf-8-sig")
    _plot_clusters(clustered, centroids, best_k, paths["plot"])

    return {**paths, "best_k": best_k}


def _plot_clusters(clustered: pd.DataFrame, centroids: pd.DataFrame, best_k: int, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for cluster_id in sorted(clustered["cluster"].unique()):
        part = clustered[clustered["cluster"] == cluster_id]
        plt.scatter(part["pca_x"], part["pca_y"], s=55, alpha=0.85, label=f"Cluster {cluster_id}")

    plt.scatter(
        centroids["pca_x"],
        centroids["pca_y"],
        s=220,
        marker="X",
        c="black",
        label="Centroid",
    )
    for _, row in centroids.iterrows():
        plt.annotate(f"C{int(row['cluster'])}", (row["pca_x"], row["pca_y"]), xytext=(6, 6), textcoords="offset points")

    plt.title(f"User Preference Clusters (K={best_k})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
