import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

_driver = None
_driver_config = None


def _get_driver():
    global _driver, _driver_config

    load_dotenv(ENV_PATH, override=True)
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not uri or not user or not password:
        raise RuntimeError("Missing NEO4J_URI, NEO4J_USER, or NEO4J_PASSWORD")

    config = (uri, user, password)
    if _driver is None or _driver_config != config:
        if _driver is not None:
            _driver.close()
        _driver = GraphDatabase.driver(uri, auth=(user, password))
        _driver_config = config

    return _driver


def run_query(cypher: str, params: dict | None = None):
    params = params or {}
    _driver = _get_driver()
    with _driver.session() as session:
        result = session.run(cypher, params)
        return [record.data() for record in result]
