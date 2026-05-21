from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main.main import app
from app.services.recommendation.recommenders.user_based import UserBasedRecommender
from app.services.recommendation.vectorizer import FEATURE_NAMES, empty_preference_state



def assert_expected_routes() -> None:
    actual_routes = {
        (method, route.path)
        for route in app.routes
        for method in getattr(route, "methods", set())
    }
    expected_routes = {
        ("GET", "/api/v1/auth/login"),
        ("POST", "/api/v1/auth/register"),
        ("GET", "/api/v1/sessions"),
        ("GET", "/api/v1/history/{session_id}"),
        ("DELETE", "/api/v1/session/{session_id}"),
        ("POST", "/api/v1/recommendation/user-based/start"),
        ("POST", "/api/v1/recommendation/user-based/chat"),
        ("POST", "/api/v1/recommendation/cluster-based/start"),
        ("POST", "/api/v1/recommendation/cluster-based/chat"),
        ("POST", "/api/v1/chat"),
    }
    missing = expected_routes - actual_routes
    if missing:
        formatted = ", ".join(f"{method} {path}" for method, path in sorted(missing))
        raise AssertionError(f"Missing expected API routes: {formatted}")


def assert_recommendation_assets() -> None:
    required_files = [
        "data/Items_Feature.csv",
        "data/recommendation_output/final/user_preference_vectors_expanded.csv",
        "data/recommendation_output/final/user_preference_clusters.csv",
        "data/recommendation_output/final/user_preference_cluster_centroids.csv",
    ]
    missing = [path for path in required_files if not (PROJECT_ROOT / path).exists()]
    if missing:
        raise AssertionError(f"Missing recommendation data files: {', '.join(missing)}")


def assert_recommender_smoke() -> None:
    preferences = empty_preference_state()
    preferences.update(
        {
            "budget_level": "medium",
            "usage_fit": ["daily", "city"],
            "style": ["modern"],
            "performance": "medium",
            "comfort": "medium",
            "safety_level": "high",
            "technology_level": "unknown",
            "easy_to_ride": True,
            "fuel_saving": True,
            "storage_need": False,
            "maintenance_easy": True,
        }
    )

    recommender = UserBasedRecommender()
    nearest = recommender.recommend_nearest_user(preferences, top_k=1)
    cluster = recommender.recommend_cluster(preferences, top_k=None)

    if len(FEATURE_NAMES) != 27:
        raise AssertionError(f"Expected 27 user preference features, got {len(FEATURE_NAMES)}")
    if len(nearest.get("candidates", [])) != 1:
        raise AssertionError("User-based recommendation must return exactly one candidate")
    if not cluster.get("candidates"):
        raise AssertionError("Cluster-based recommendation returned no candidates")

    item_ids = [str(candidate.get("item_id")) for candidate in cluster["candidates"]]
    if len(item_ids) != len(set(item_ids)):
        raise AssertionError("Cluster-based recommendation returned duplicate item_ids")


def main() -> None:
    assert_expected_routes()
    assert_recommendation_assets()
    assert_recommender_smoke()
    print("CI smoke checks passed")


if __name__ == "__main__":
    main()
