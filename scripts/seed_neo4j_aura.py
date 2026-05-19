import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT_DIR / "data" / "Items_Feature.csv"


def clean_value(value: str | None) -> str:
    return (value or "").strip()


def split_values(value: str | None) -> list[str]:
    return [item.strip() for item in clean_value(value).split(",") if item.strip()]


def to_bool(value: str | None) -> bool:
    return clean_value(value) in {"1", "true", "True", "yes", "Yes"}


def create_entity(tx, label: str, name: str, semantic: str, relationship: str) -> None:
    if not name:
        return

    tx.run(
        f"""
        MERGE (m:VehicleModel {{name: $model}})
        MERGE (e:{label} {{name: $name}})
        MERGE (s:SemanticTag {{name: $semantic}})
        MERGE (m)-[:{relationship}]->(e)
        MERGE (e)-[:HAS_SEMANTIC]->(s)
        """,
        model=create_entity.model_name,
        name=name,
        semantic=semantic,
    )


def seed_row(tx, row: dict[str, str]) -> None:
    model = clean_value(row.get("model"))
    brand = clean_value(row.get("brand"))
    item_id = clean_value(row.get("item_id"))
    model_key = clean_value(row.get("model_key"))

    tx.run(
        """
        MERGE (b:Brand {name: $brand})
        MERGE (m:VehicleModel {name: $model})
        SET
            m.item_id = $item_id,
            m.model_key = $model_key,
            m.cc = $cc,
            m.price_est_thb = $price_est_thb,
            m.type = $type,
            m.budget_level = $budget_level
        MERGE (m)-[:MADE_BY]->(b)
        """,
        brand=brand,
        model=model,
        item_id=item_id,
        model_key=model_key,
        cc=int(clean_value(row.get("cc")) or 0),
        price_est_thb=int(clean_value(row.get("price_est_thb")) or 0),
        type=clean_value(row.get("type")),
        budget_level=clean_value(row.get("budget_level")),
    )

    create_entity.model_name = model

    create_entity(tx, "BudgetLevel", clean_value(row.get("budget_level")), "budget", "HAS_BUDGET")
    create_entity(tx, "Style", clean_value(row.get("style")), "style", "HAS_STYLE")
    create_entity(tx, "Performance", clean_value(row.get("performance")), "performance", "HAS_PERFORMANCE")
    create_entity(tx, "Comfort", clean_value(row.get("comfort")), "comfort", "HAS_COMFORT")
    create_entity(tx, "Engine", f"{clean_value(row.get('cc'))} cc", "engine", "HAS_ENGINE")
    create_entity(tx, "VehicleType", clean_value(row.get("type")), "type", "HAS_TYPE")

    for usage in split_values(row.get("usage_fit")):
        create_entity(tx, "UseCase", usage, "use_case", "HAS_USE_CASE")

    if to_bool(row.get("easy_to_ride")):
        create_entity(tx, "DecisionFactor", "ขับขี่ง่าย", "easy_to_ride", "HAS_DECISION_FACTOR")
    if to_bool(row.get("fuel_saving")):
        create_entity(tx, "Efficiency", "ประหยัดน้ำมัน", "fuel_saving", "HAS_EFFICIENCY")
    if to_bool(row.get("storage_need")):
        create_entity(tx, "Storage", "มีพื้นที่เก็บของ", "storage", "HAS_STORAGE")


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        raise RuntimeError("Please set NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD in .env")

    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT vehicle_model_name IF NOT EXISTS FOR (m:VehicleModel) REQUIRE m.name IS UNIQUE"))
            session.execute_write(lambda tx: tx.run("CREATE CONSTRAINT brand_name IF NOT EXISTS FOR (b:Brand) REQUIRE b.name IS UNIQUE"))

            for row in rows:
                session.execute_write(seed_row, row)

            counts = session.run(
                """
                MATCH (n)
                WITH count(n) AS nodes
                MATCH ()-[r]->()
                RETURN nodes, count(r) AS relationships
                """
            ).single()

        print(f"Seeded Neo4j Aura from {CSV_PATH}")
        print(f"Nodes: {counts['nodes']}")
        print(f"Relationships: {counts['relationships']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
