import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_driver = GraphDatabase.driver(
    os.getenv('NEO4J_URI'),
    auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD')),
)

def run_query(cypher: str, params: dict = {}):
    with _driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]