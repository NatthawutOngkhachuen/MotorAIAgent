from app.services.recommendation.router import RecommendationRouter


def main():
    router = RecommendationRouter()

    test_messages = [
        # recommendation ปกติ
        "อยากได้รถขี่ในเมือง ประหยัดน้ำมัน งบไม่แรง ขับง่าย",
        "ผมอยากได้รถทรงสปอร์ต ใช้ไปเรียนทุกวัน ขอแรงนิดนึง",
        "หารถไว้ส่งของ ประหยัดน้ำมัน มีที่เก็บของเยอะๆ",

        # info_lookup รุ่นที่มีในระบบ
        "ขอข้อมูล PCX 160 หน่อย",
        "Honda Click 160 ดีไหม",

        # info_lookup รุ่นนอกระบบ
        "ขอข้อมูล CBR หน่อย",

        # similar_to_model รุ่นที่มีในระบบ
        "แนะนำรถคล้าย PCX 160 ให้หน่อย",
        "อยากได้รถที่คล้าย N-MAX แนะนำหน่อย",

        # similar_to_model รุ่นนอกระบบ
        "แนะนำรถคล้าย R7 ให้หน่อย",
    ]

    for message in test_messages:
        print("=" * 100)
        print("USER:", message)

        result = router.route(
            user_message=message,
            top_k=3,
        )

        print("ROUTE:", result.route)
        print("RESPONSE TYPE:", result.response_type)
        print("PREFERENCE:")
        print(result.preference)

        if result.catalog_result is not None:
            print("CATALOG:")
            print(result.catalog_result)

        print("CANDIDATES:")
        for candidate in result.candidates:
            print(candidate)

        print("GRAPH ITEM IDS:")
        print(result.graph_item_ids)

        if result.message:
            print("MESSAGE:")
            print(result.message)


if __name__ == "__main__":
    main()