from app.services.recommendation.preference_extractor import PreferenceExtractorService
from app.services.recommendation.recommenders.item_based import ItemBasedRecommender


def main():
    extractor = PreferenceExtractorService()
    recommender = ItemBasedRecommender()

    test_messages = [
        "อยากได้รถขี่ในเมือง ประหยัดน้ำมัน งบไม่แรง ขับง่าย",
        "ผมอยากได้รถทรงสปอร์ต ใช้ไปเรียนทุกวัน ขอแรงนิดนึง",
        "อยากได้รถสำหรับออกทริปไกลๆ นั่งสบาย ดูพรีเมียม",
        "หารถไว้ส่งของ ประหยัดน้ำมัน มีที่เก็บของเยอะๆ",
        "อยากได้ Honda Click 160 ขี่ในเมือง ประหยัดน้ำมัน",
    ]

    for message in test_messages:
        print("=" * 100)
        print("USER:", message)

        extract_result = extractor.extract(
            user_message=message,
            schema_type="item_based",
        )

        print("\nPREFERENCE:")
        print(extract_result.preference)

        candidates = recommender.recommend(
            preference=extract_result.preference,
            top_k=3,
        )

        print("\nTOP-K CANDIDATES:")
        for candidate in candidates:
            print(candidate)


if __name__ == "__main__":
    main()