from app.db.neo4j import run_query


def search_by_keyword(keyword: str) -> dict:
    cypher_nodes = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($keyword))
        RETURN n
    """
    cypher_rels = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($keyword))
        MATCH (n)-[r]-(m)
        RETURN
            id(n) AS from_id, labels(n) AS from_labels, properties(n) AS from_props,
            type(r)  AS rel_type,
            id(m) AS to_id,   labels(m) AS to_labels,   properties(m) AS to_props
    """
    matched = run_query(cypher_nodes, {"keyword": keyword})
    relations = run_query(cypher_rels, {"keyword": keyword})

    nodes_map = {}
    for row in matched:
        node = dict(row["n"])
        nid = next(iter(node.values()), "?")
        nodes_map[str(nid)] = {"id": str(nid), "props": node, "matched": True}

    edges = []
    for row in relations:
        fid = str(row["from_id"])
        tid = str(row["to_id"])
        if fid not in nodes_map:
            nodes_map[fid] = {"id": fid, "props": dict(row["from_props"]), "matched": False}
        if tid not in nodes_map:
            nodes_map[tid] = {"id": tid, "props": dict(row["to_props"]), "matched": False}
        edges.append({"from": fid, "to": tid, "rel": row["rel_type"]})

    return {"nodes": list(nodes_map.values()), "edges": edges, "count": len(nodes_map)}


def get_full_graph() -> dict:
    """Fetch all nodes and edges from Neo4j."""
    cypher = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN
            id(n)          AS from_id,
            labels(n)      AS from_labels,
            properties(n)  AS from_props,
            type(r)        AS rel_type,
            id(m)          AS to_id,
            labels(m)      AS to_labels,
            properties(m)  AS to_props
    """
    rows = run_query(cypher, {})

    nodes_map = {}
    edges = []

    for row in rows:
        fid = str(row["from_id"])
        flabel = row["from_labels"][0] if row["from_labels"] else "Node"
        fprops = dict(row["from_props"])

        if fid not in nodes_map:
            nodes_map[fid] = {
                "id": fid,
                "label": flabel,
                "name": fprops.get("name", fprops.get("id", fid)),
                "props": fprops,
            }

        if row["rel_type"] and row["to_id"] is not None:
            tid = str(row["to_id"])
            tlabel = row["to_labels"][0] if row["to_labels"] else "Node"
            tprops = dict(row["to_props"])

            if tid not in nodes_map:
                nodes_map[tid] = {
                    "id": tid,
                    "label": tlabel,
                    "name": tprops.get("name", tprops.get("id", tid)),
                    "props": tprops,
                }

            edges.append({
                "from": fid,
                "to": tid,
                "rel": row["rel_type"],
            })

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges,
        "node_count": len(nodes_map),
        "edge_count": len(edges),
    }
