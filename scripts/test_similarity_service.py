from app.services.recommendation.catalog_resolver import CatalogResolver
from app.services.recommendation.similarity_service import SimilarityService


def main():
    catalog_resolver = CatalogResolver()
    similarity_service = SimilarityService()

    test_models = [
        "PCX 160",
        "PCX",
        "N-MAX",
        "Click 160",
        "Wave",
        "CBR",
        "R7",
        "unknown",
    ]

    for model_name in test_models:
        print("=" * 100)
        print("SOURCE MODEL RAW:", model_name)

        resolved = catalog_resolver.resolve(model_name)

        print("CATALOG FOUND:", resolved.found)
        print("CATALOG STATUS:", resolved.status)
        print("SOURCE ITEM ID:", resolved.item_id)
        print("SOURCE BRAND:", resolved.brand)
        print("SOURCE MODEL:", resolved.model)

        if not resolved.found:
            print("RESULT:")
            print(f"ไม่มีรุ่น {model_name} ในฐานข้อมูล จึงไม่สามารถหา similar items ได้")
            continue

        candidates = similarity_service.get_similar_items(
            source_item_id=resolved.item_id,
            top_k=3,
        )

        print("SIMILAR CANDIDATES:")
        for candidate in candidates:
            print(candidate)

        similar_item_ids = similarity_service.get_similar_item_ids(
            source_item_id=resolved.item_id,
            top_k=3,
        )

        print("SIMILAR ITEM IDS FOR GRAPHRAG:")
        print(similar_item_ids)


if __name__ == "__main__":
    main()