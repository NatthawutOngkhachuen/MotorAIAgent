from typing import Any

from app.services.recommendation.data_loader import RecommendationDataLoader


class SimilarityService:
    """
    Similarity Service สำหรับ MotorAiAgent

    ใช้กับ route:
    similar_to_model
      → Catalog Resolver
      → Similarity Service
      → GraphRAG by similar item_ids
      → Response LLM

    หน้าที่:
    - รับ source_item_id ที่ผ่าน CatalogResolver แล้วเท่านั้น
    - อ่าน item_based_similarity_baseline.csv ผ่าน RecommendationDataLoader
    - คืน Top-K รุ่นที่คล้ายกัน
    - ไม่ตัดสินเองว่ารุ่นต้นทางมีใน catalog หรือไม่ เพราะเป็นหน้าที่ของ Router + CatalogResolver
    """

    def __init__(self, data_loader: RecommendationDataLoader | None = None):
        self.data_loader = data_loader or RecommendationDataLoader()
        self.similarity_df = self.data_loader.load_similarity_baseline()

    def get_similar_items(
        self,
        source_item_id: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        คืนรายการรุ่นที่คล้ายกับ source_item_id

        Parameters:
            source_item_id:
                item_id ของรุ่นต้นทาง เช่น I007

            top_k:
                จำนวนรายการที่ต้องการ รองรับ 1, 3, 5

        Returns:
            [
                {
                    "rank": 1,
                    "source_item_id": "I007",
                    "source_model": "PCX 160",
                    "item_id": "I006",
                    "model": "N-MAX",
                    "similarity": 0.7321,
                    "method": "item_similarity"
                }
            ]
        """

        normalized_top_k = self._normalize_top_k(top_k)
        normalized_source_item_id = str(source_item_id).strip()

        matched_df = self.similarity_df[
            self.similarity_df["source_item_id"].astype(str) == normalized_source_item_id
        ].copy()

        if matched_df.empty:
            return []

        matched_df["similarity"] = matched_df["similarity"].astype(float)

        matched_df = matched_df.sort_values(
            by="similarity",
            ascending=False,
        ).head(normalized_top_k)

        results: list[dict[str, Any]] = []

        for rank, (_, row) in enumerate(matched_df.iterrows(), start=1):
            results.append(
                {
                    "rank": rank,
                    "source_item_id": str(row["source_item_id"]),
                    "source_model": str(row["source_model"]),
                    "item_id": str(row["similar_item_id"]),
                    "model": str(row["similar_model"]),
                    "similarity": round(float(row["similarity"]), 4),
                    "method": "item_similarity",
                }
            )

        return results

    def get_similar_item_ids(
        self,
        source_item_id: str,
        top_k: int = 3,
    ) -> list[str]:
        """
        คืนเฉพาะ similar item_id
        ใช้ตอนจะส่งต่อไป GraphRAG
        """

        candidates = self.get_similar_items(
            source_item_id=source_item_id,
            top_k=top_k,
        )

        return [candidate["item_id"] for candidate in candidates]

    def _normalize_top_k(self, top_k: int) -> int:
        """
        จำกัด top_k ให้เป็นค่าที่ระบบรองรับ
        """

        if top_k in [1, 3, 5]:
            return top_k

        return 3