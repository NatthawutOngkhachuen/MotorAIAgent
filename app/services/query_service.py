from app.db.neo4j import run_query

def search_by_keyword(keyword: str) -> dict:
    # ค้นหา node ที่ตรงกับ keyword ก่อน
    cypher_nodes = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($keyword))
        RETURN n
    """
    # ดึง relationship ทั้งหมดที่เชื่อมกับ node ที่เจอ (ทั้ง 2 ทิศทาง)
    cypher_rels = """
        MATCH (n)
        WHERE any(prop IN keys(n) WHERE toLower(toString(n[prop])) CONTAINS toLower($keyword))
        MATCH (n)-[r]-(m)
        RETURN 
            id(n) AS from_id, labels(n) AS from_labels, properties(n) AS from_props,
            type(r)  AS rel_type,
            id(m) AS to_id,   labels(m) AS to_labels,   properties(m) AS to_props
    """

    matched   = run_query(cypher_nodes, {"keyword": keyword})
    relations = run_query(cypher_rels,  {"keyword": keyword})

    # รวม node ที่ไม่ซ้ำ
    nodes_map = {}
    for row in matched:
        node = dict(row["n"])
        nid  = next(iter(node.values()), "?")   # ใช้ค่าแรกเป็น key ชั่วคราว
        nodes_map[str(nid)] = {"id": str(nid), "props": node, "matched": True}

    edges = []
    for row in relations:
        fp = row["from_props"]
        tp = row["to_props"]
        fid = str(row["from_id"])
        tid = str(row["to_id"])

        if fid not in nodes_map:
            nodes_map[fid] = {"id": fid, "props": dict(fp), "matched": False}
        if tid not in nodes_map:
            nodes_map[tid] = {"id": tid, "props": dict(tp), "matched": False}

        edges.append({
            "from": fid,
            "to":   tid,
            "rel":  row["rel_type"],
        })

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges,
        "count": len(nodes_map),
    }