import os
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable, SessionExpired
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


def _close_driver():
    global _driver, _driver_config

    if _driver is not None:
        _driver.close()
    _driver = None
    _driver_config = None


def run_query(cypher: str, params: dict | None = None):
    params = params or {}

    for attempt in range(2):
        driver = _get_driver()
        try:
            with driver.session() as session:
                result = session.run(cypher, params)
                return [record.data() for record in result]
        except (ServiceUnavailable, SessionExpired, ConnectionResetError) as exc:
            _close_driver()
            if attempt == 0:
                print(f"[WARN] Neo4j connection reset. Reconnecting once. {exc}")
                continue
            raise
        except Neo4jError:
            raise

    return []
