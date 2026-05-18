from app.services.recommendation.router import RecommendationRouter
from app.services.recommendation.graph_retriever import GraphRetriever


def main():
    router = RecommendationRouter()
    graph_retriever = GraphRetriever()

    test_messages = [
        "อยากได้รถขี่ในเมือง ประหยัดน้ำมัน งบไม่แรง ขับง่าย",
        "ขอข้อมูล PCX 160 หน่อย",
        "แนะนำรถคล้าย PCX 160 ให้หน่อย",
        "ขอข้อมูล CBR หน่อย",
        "แนะนำรถคล้าย R7 ให้หน่อย",
    ]

    for message in test_messages:
        print("=" * 100)
        print("USER:", message)

        route_result = router.route(
            user_message=message,
            top_k=3,
        )

        print("ROUTE:", route_result.route)
        print("RESPONSE TYPE:", route_result.response_type)
        print("GRAPH ITEM IDS:", route_result.graph_item_ids)

        if route_result.message:
            print("ROUTER MESSAGE:")
            print(route_result.message)

        if not route_result.graph_item_ids:
            print("GRAPH EVIDENCE:")
            print([])
            continue

        evidence = graph_retriever.retrieve_by_item_ids(
            item_ids=route_result.graph_item_ids,
        )

        print("GRAPH EVIDENCE SUMMARY:")
        for item_evidence in evidence:
            print("-" * 80)
            print("ITEM ID:", item_evidence["item_id"])
            print("MODEL:", item_evidence["model"])
            print("FOUND:", item_evidence["found"])
            print("GRAPH LOOKUP KEY:", item_evidence["graph_lookup_key"])
            print("RAW ROWS:", item_evidence["raw_graph_rows_count"])
            print(item_evidence["summary_text"])


if __name__ == "__main__":
    main()