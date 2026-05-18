from app.services.recommendation.catalog_resolver import CatalogResolver


def main():
    resolver = CatalogResolver()

    test_models = [
        "PCX 160",
        "PCX",
        "N-MAX",
        "nmax",
        "Click",
        "Click 160",
        "Wave",
        "CBR",
        "R7",
        "unknown",
    ]

    print("CATALOG ITEMS")
    print(resolver.get_all_catalog_items())

    for model_name in test_models:
        print("=" * 80)
        print("RAW MODEL:", model_name)

        result = resolver.resolve(model_name)

        print("FOUND:", result.found)
        print("STATUS:", result.status)
        print("ITEM ID:", result.item_id)
        print("BRAND:", result.brand)
        print("MODEL:", result.model)


if __name__ == "__main__":
    main()