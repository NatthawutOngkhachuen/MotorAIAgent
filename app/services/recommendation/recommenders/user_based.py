from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.services.recommendation.vectorizer import FEATURE_NAMES, preference_to_vector


class UserBasedRecommender:
    def __init__(
        self,
        vector_path: str | Path | None = None,
        cluster_path: str | Path | None = None,
        centroid_path: str | Path | None = None,
        items_path: str | Path | None = None,
    ):
        self.vector_path = Path(
            vector_path
            or os.getenv("USER_PREFERENCE_VECTOR_PATH", "data/recommendation_output/final/user_preference_vectors_expanded.csv")
        )
        self.cluster_path = Path(
            cluster_path
            or os.getenv("USER_PREFERENCE_CLUSTER_PATH", "data/recommendation_output/final/user_preference_clusters.csv")
        )
        self.centroid_path = Path(
            centroid_path
            or os.getenv("USER_PREFERENCE_CENTROID_PATH", "data/recommendation_output/final/user_preference_cluster_centroids.csv")
        )
        self.items_path = Path(items_path or os.getenv("ITEMS_FEATURE_PATH", "data/Items_Feature.csv"))

    def recommend_nearest_user(
        self,
        preference: dict[str, Any],
        top_k: int = 3,
    ) -> dict[str, Any]:
        vectors = self._load_vectors()
        x = vectors[FEATURE_NAMES].astype(float).to_numpy()
        query = np.array(preference_to_vector(preference), dtype=float).reshape(1, -1)
        sims = cosine_similarity(query, x)[0]

        ranked = vectors.assign(similarity=sims).sort_values("similarity", ascending=False)
        candidates = self._dedupe_candidates(
            self._candidates_from_rows(ranked, method="nearest_user"),
            top_k=top_k,
        )
        best = ranked.iloc[0].to_dict()
        return {
            "method": "nearest_user",
            "matched_user_id": best.get("user_id"),
            "matched_similarity": float(best.get("similarity", 0.0)),
            "candidates": candidates,
        }

    def recommend_cluster(
        self,
        preference: dict[str, Any],
        top_k: int | None = 5,
    ) -> dict[str, Any]:
        clusters = self._load_clusters()
        centroids = self._load_centroids()
        query = np.array(preference_to_vector(preference), dtype=float).reshape(1, -1)
        centroid_x = centroids[FEATURE_NAMES].astype(float).to_numpy()
        sims = cosine_similarity(query, centroid_x)[0]
        best_index = int(np.argmax(sims))
        cluster_id = int(centroids.iloc[best_index]["cluster"])

        members = clusters[clusters["cluster"].astype(int) == cluster_id].copy()
        members = members.sort_values(["mapped_model", "user_id"])
        candidates = self._candidates_from_rows(members, method="cluster_member")
        deduped = self._dedupe_candidates(candidates, top_k=top_k)

        return {
            "method": "cluster",
            "cluster": cluster_id,
            "cluster_similarity": float(sims[best_index]),
            "cluster_size": int(len(members)),
            "candidates": deduped,
        }

    def _load_vectors(self) -> pd.DataFrame:
        if not self.vector_path.exists():
            raise FileNotFoundError(f"User vector file not found: {self.vector_path}")
        return pd.read_csv(self.vector_path)

    def _load_clusters(self) -> pd.DataFrame:
        if not self.cluster_path.exists():
            raise FileNotFoundError(f"Cluster file not found: {self.cluster_path}")
        return pd.read_csv(self.cluster_path)

    def _load_centroids(self) -> pd.DataFrame:
        if not self.centroid_path.exists():
            raise FileNotFoundError(f"Centroid file not found: {self.centroid_path}")
        return pd.read_csv(self.centroid_path)

    def _load_items(self) -> pd.DataFrame:
        return pd.read_csv(self.items_path)

    def _candidates_from_rows(self, rows: pd.DataFrame, method: str) -> list[dict[str, Any]]:
        items = self._load_items()
        item_by_id = {str(row["item_id"]): row.to_dict() for _, row in items.iterrows()}
        candidates = []
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            item_id = str(row.get("item_id"))
            item = item_by_id.get(item_id, {})
            candidates.append(
                {
                    "rank": rank,
                    "item_id": item_id,
                    "brand": item.get("brand"),
                    "model": row.get("mapped_model") or item.get("model"),
                    "price_est_thb": item.get("price_est_thb"),
                    "method": method,
                    "matched_user_id": row.get("user_id"),
                    "similarity": float(row.get("similarity", 0.0)) if "similarity" in row else None,
                }
            )
        return candidates

    def _dedupe_candidates(self, candidates: list[dict[str, Any]], top_k: int | None) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen_items: set[str] = set()
        for candidate in candidates:
            item_id = str(candidate.get("item_id"))
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            candidate["rank"] = len(deduped) + 1
            deduped.append(candidate)
            if top_k is not None and len(deduped) >= top_k:
                break
        return deduped


def parse_vector_string(value: str) -> list[float]:
    parsed = ast.literal_eval(value)
    return [float(v) for v in parsed]
