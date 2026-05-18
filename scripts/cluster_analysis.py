from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ncf import read_xlsx_sheet


def _top_counts(series: pd.Series, n: int = 3) -> str:
    counts = series.fillna("unknown").astype(str).value_counts().head(n)
    return ", ".join(f"{idx}={val}" for idx, val in counts.items())


def choose_k(x: np.ndarray, k_min: int = 2, k_max: int = 6) -> pd.DataFrame:
    rows = []
    max_k = min(k_max, len(x) - 1)
    for k in range(k_min, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=30).fit_predict(x)
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(x, labels)),
                "inertia": float(KMeans(n_clusters=k, random_state=42, n_init=30).fit(x).inertia_),
            }
        )
    return pd.DataFrame(rows)


def plot_clusters(df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(df["cluster"].unique()):
        part = df[df["cluster"] == cluster_id]
        plt.scatter(part["pca_x"], part["pca_y"], s=55, label=f"Cluster {cluster_id}")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def profile_clusters(clustered: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for cluster_id, part in clustered.groupby("cluster"):
        rows.append(
            {
                "method": method,
                "cluster": int(cluster_id),
                "n_users": int(len(part)),
                "top_models": _top_counts(part["mapped_model"]),
                "age_group": _top_counts(part["age_group"]),
                "gender": _top_counts(part["gender"]),
                "usage_type": _top_counts(part["usage_type"]),
                "style": _top_counts(part["style"]),
                "performance": _top_counts(part["performance"]),
                "comfort": _top_counts(part["comfort"]),
                "fuel_saving": _top_counts(part["fuel_saving"]),
            }
        )
    return pd.DataFrame(rows)


def cluster_matrix(
    base: pd.DataFrame,
    x: np.ndarray,
    method: str,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    k_scores = choose_k(x)
    best_k = int(k_scores.sort_values("silhouette", ascending=False).iloc[0]["k"])
    labels = KMeans(n_clusters=best_k, random_state=42, n_init=30).fit_predict(x)
    xy = PCA(n_components=2, random_state=42).fit_transform(x)

    clustered = base.copy()
    clustered["cluster"] = labels
    clustered["pca_x"] = xy[:, 0]
    clustered["pca_y"] = xy[:, 1]

    prefix = method.lower().replace(" ", "_").replace("+", "plus")
    clustered.to_csv(output_dir / f"{prefix}_clustered_users.csv", index=False, encoding="utf-8-sig")
    k_scores.to_csv(output_dir / f"{prefix}_k_selection.csv", index=False, encoding="utf-8-sig")
    plot_clusters(clustered, f"{method} Clusters (K={best_k})", output_dir / f"{prefix}_clusters.png")

    return k_scores, clustered, profile_clusters(clustered, method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel", type=Path, default=Path("CF_NCF_Phase1_Backbone.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("phase2_outputs"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    interactions = read_xlsx_sheet(args.excel, "Interactions")
    user_text = read_xlsx_sheet(args.excel, "User_Text_Features")
    ncf_vectors = pd.read_csv(args.output_dir / "user_vectors.csv", encoding="utf-8-sig")

    base_cols = [
        "user_id",
        "mapped_model",
        "age_group",
        "gender",
        "usage_type",
        "style",
        "performance",
        "comfort",
        "fuel_saving",
    ]
    base = (
        interactions[["user_id", "mapped_model", "age_group", "gender"]]
        .merge(
            user_text[["user_id", "usage_type", "style", "performance", "comfort", "fuel_saving"]],
            on="user_id",
            how="left",
        )
        .loc[:, base_cols]
    )

    emb_cols = [col for col in ncf_vectors.columns if col.startswith("emb_")]
    ncf_base = base.merge(ncf_vectors[["user_id"] + emb_cols], on="user_id", how="left")
    ncf_x = StandardScaler().fit_transform(ncf_base[emb_cols].to_numpy())

    feature_cols = ["age_group", "gender", "usage_type", "style", "performance", "comfort", "fuel_saving"]
    feature_base = base.copy()
    for col in feature_cols:
        feature_base[col] = feature_base[col].fillna("unknown").astype(str)
    feature_x = OneHotEncoder(handle_unknown="ignore").fit_transform(feature_base[feature_cols]).toarray()

    all_profiles = []
    all_k_scores = []
    for method, method_base, x in [
        ("NCF User Vector", ncf_base.drop(columns=emb_cols), ncf_x),
        ("User-Based Feature Vector", feature_base, feature_x),
    ]:
        k_scores, _, profile = cluster_matrix(method_base, x, method, args.output_dir)
        k_scores.insert(0, "method", method)
        all_k_scores.append(k_scores)
        all_profiles.append(profile)

    pd.concat(all_k_scores, ignore_index=True).to_csv(
        args.output_dir / "cluster_k_selection_summary.csv", index=False, encoding="utf-8-sig"
    )
    profile_summary = pd.concat(all_profiles, ignore_index=True)
    profile_summary.to_csv(args.output_dir / "cluster_profile_summary.csv", index=False, encoding="utf-8-sig")
    print(profile_summary.to_string(index=False))


if __name__ == "__main__":
    main()
