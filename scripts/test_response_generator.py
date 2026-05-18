from app.services.recommendation.graph_retriever import GraphRetriever
from app.services.recommendation.response_generator import ResponseGenerator
from app.services.recommendation.router import RecommendationRouter


def main():
    router = RecommendationRouter()
    graph_retriever = GraphRetriever()
    response_generator = ResponseGenerator()

    test_messages = [
        "เวฟ125ดีกว่าคลิกยังไง",
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

        graph_evidence = []
        if route_result.graph_item_ids:
            graph_evidence = graph_retriever.retrieve_by_item_ids(
                item_ids=route_result.graph_item_ids,
            )

        answer = response_generator.generate(
            user_message=message,
            route_result=route_result,
            graph_evidence=graph_evidence,
        )

        print("\nFINAL ANSWER:")
        print(answer)


if __name__ == "__main__":
    main()