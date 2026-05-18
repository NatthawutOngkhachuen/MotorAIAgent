from app.services.recommendation.data_loader import RecommendationDataLoader


def main():
    loader = RecommendationDataLoader()

    print("=" * 80)
    print("LOAD ITEMS FEATURE")
    items_df = loader.load_items_feature()
    print("Rows:", len(items_df))
    print("Columns:", list(items_df.columns))
    print(items_df[["item_id", "brand", "model"]].head())

    print("=" * 80)
    print("LOAD SIMILARITY BASELINE")
    sim_df = loader.load_similarity_baseline()
    print("Rows:", len(sim_df))
    print("Columns:", list(sim_df.columns))
    print(sim_df.head())

    print("=" * 80)
    print("GET ITEM BY ID")
    item = loader.get_item_by_id("I007")
    print(item)

    print("=" * 80)
    print("CACHE TEST")
    items_df_2 = loader.load_items_feature()
    sim_df_2 = loader.load_similarity_baseline()
    print("Items loaded again:", len(items_df_2))
    print("Similarity loaded again:", len(sim_df_2))


if __name__ == "__main__":
    main()