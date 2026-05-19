from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.neo4j import run_query


def split_tokens(value) -> list[str]:
    return [
        token.strip()
        for token in str(value or "").split(",")
        if token.strip() and token.strip().lower() not in {"unknown", "nan"}
    ]


def main() -> None:
    items = pd.read_csv(PROJECT_ROOT / "data/Items_Feature.csv")
    run_query("MATCH (n) DETACH DELETE n")

    for _, item in items.iterrows():
        params = {
            "item_id": str(item["item_id"]),
            "brand": str(item["brand"]),
            "model": str(item["model"]),
            "price_est_thb": str(item.get("price_est_thb", "")),
            "cc": str(item.get("cc", "")),
            "budget_level": str(item.get("budget_level", "")),
            "performance": str(item.get("performance", "")),
            "comfort": str(item.get("comfort", "")),
            "usage": split_tokens(item.get("usage_fit")),
            "style": split_tokens(item.get("style")),
            "easy_to_ride": str(item.get("easy_to_ride", "")),
            "fuel_saving": str(item.get("fuel_saving", "")),
            "storage_need": str(item.get("storage_need", "")),
        }
        run_query(
            """
            MERGE (m:VehicleModel {name: $model})
            SET m.item_id = $item_id, m.price_est_thb = $price_est_thb
            MERGE (b:Brand {name: $brand})
            MERGE (m)-[:HAS_BRAND]->(b)
            MERGE (cc:EngineSpec {name: $cc})
            MERGE (m)-[:HAS_ENGINE_CC]->(cc)
            MERGE (budget:DecisionFactor {name: 'budget_' + $budget_level})
            MERGE (m)-[:DECISION_FACTOR]->(budget)
            MERGE (perf:Performance {name: 'performance_' + $performance})
            MERGE (m)-[:HAS_PERFORMANCE]->(perf)
            MERGE (comfort:ComfortFeature {name: 'comfort_' + $comfort})
            MERGE (m)-[:HAS_COMFORT_FEATURE]->(comfort)
            WITH m
            UNWIND $usage AS usage_name
            MERGE (u:UseCase {name: usage_name})
            MERGE (m)-[:SUITABLE_FOR]->(u)
            WITH m
            UNWIND $style AS style_name
            MERGE (s:Style {name: style_name})
            MERGE (m)-[:HAS_STYLE]->(s)
            """,
            params,
        )

        if params["easy_to_ride"] in {"1", "TRUE", "True", "true"}:
            run_query(
                """
                MATCH (m:VehicleModel {name: $model})
                MERGE (c:ConvenienceFeature {name: 'easy_to_ride'})
                MERGE (m)-[:HAS_CONVENIENCE_FEATURE]->(c)
                """,
                params,
            )
        if params["fuel_saving"] in {"1", "TRUE", "True", "true"}:
            run_query(
                """
                MATCH (m:VehicleModel {name: $model})
                MERGE (e:Efficiency {name: 'fuel_saving'})
                MERGE (m)-[:HAS_EFFICIENCY]->(e)
                """,
                params,
            )
        if params["storage_need"] in {"1", "TRUE", "True", "true"}:
            run_query(
                """
                MATCH (m:VehicleModel {name: $model})
                MERGE (s:StorageFeature {name: 'storage_need'})
                MERGE (m)-[:HAS_STORAGE_FEATURE]->(s)
                """,
                params,
            )

    count = run_query("MATCH (m:VehicleModel) RETURN count(m) AS count")[0]["count"]
    print(f"SEEDED_VEHICLE_MODELS={count}")


if __name__ == "__main__":
    main()
