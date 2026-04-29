import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "")),
)


def run_query(cypher: str, params: dict | None = None):
    with _driver.session() as session:
        result = session.run(cypher, params or {})
        return [record.data() for record in result]
