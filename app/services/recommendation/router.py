from dataclasses import dataclass
from typing import Any, Literal

from app.services.recommendation.catalog_resolver import (
    CatalogResolver,
    CatalogResolveResult,
)
from app.services.recommendation.preference_extractor import PreferenceExtractorService
from app.services.recommendation.recommenders.item_based import ItemBasedRecommender
from app.services.recommendation.similarity_service import SimilarityService


RouteName = Literal[
    "recommendation",
    "info_lookup",
    "similar_to_model",
    "comparison",
    "price_lookup",
    "out_of_catalog",
]


@dataclass
class RecommendationRouteResult:
    """
    ผลลัพธ์จาก Router

    route:
        เส้นทางที่ระบบเลือก เช่น recommendation, info_lookup, similar_to_model,
        comparison, price_lookup, out_of_catalog

    user_message:
        ข้อความต้นฉบับของผู้ใช้

    preference:
        JSON preference จาก Extractor

    catalog_result:
        ผลจาก CatalogResolver ถ้า route นั้นต้องใช้

    candidates:
        รายการรถที่ถูกเลือกจาก Item-Based, Similarity หรือ Catalog Lookup

    graph_item_ids:
        item_id ของฝั่ง recommendation layer เช่น I001, I007
        ขั้นถัดไป graph_retriever.py จะ map item_id → model name → Neo4j VehicleModel เอง

        หมายเหตุ:
        - price_lookup จะเป็น [] เพราะตอบราคาจาก Items_Feature.csv ได้เลย
          ไม่จำเป็นต้องเข้า GraphRAG

    response_type:
        ใช้บอกขั้น response ว่าควรตอบแบบไหน
    """

    route: RouteName
    user_message: str
    preference: dict[str, Any]
    catalog_result: CatalogResolveResult | None
    candidates: list[dict[str, Any]]
    graph_item_ids: list[str]
    response_type: str
    message: str | None = None


class RecommendationRouter:
    """
    Router หลักของ MotorAiAgent

    หลักการ:
    - 1 user message = Extractor 1 ครั้ง
    - Router เลือก route เดียวเท่านั้น
    - recommendation ปกติ → Item-Based Top-K
    - info_lookup → Catalog Resolver → GraphRAG item_id
    - comparison → Catalog Resolver หลายรุ่น → GraphRAG item_ids
    - price_lookup → Catalog Resolver → ตอบราคาจาก Items_Feature.csv ไม่ query GraphRAG
    - similar_to_model → Catalog Resolver → Similarity Top-K
    - รุ่นนอก catalog → out_of_catalog ทันที ไม่ query GraphRAG
    """

    def __init__(
        self,
        extractor: PreferenceExtractorService | None = None,
        catalog_resolver: CatalogResolver | None = None,
        item_based_recommender: ItemBasedRecommender | None = None,
        similarity_service: SimilarityService | None = None,
    ):
        self.extractor = extractor or PreferenceExtractorService()
        self.catalog_resolver = catalog_resolver or CatalogResolver()
        self.item_based_recommender = item_based_recommender or ItemBasedRecommender()
        self.similarity_service = similarity_service or SimilarityService()

    def route(
        self,
        user_message: str,
        top_k: int = 3,
    ) -> RecommendationRouteResult:
        """
        Entry point หลักของ Router

        ยังไม่เรียก GraphRAG จริง
        คืน graph_item_ids ให้ graph_retriever.py ใช้ในขั้นถัดไป

        ยกเว้น:
        - price_lookup จะ graph_item_ids = []
          เพราะตอบราคาจาก Items_Feature.csv ได้เลย
        - out_of_catalog จะ graph_item_ids = []
          เพราะไม่มีรุ่นในฐานข้อมูล
        """

        extract_result = self.extractor.extract(
            user_message=user_message,
            schema_type="item_based",
        )

        preference = extract_result.preference

        intent = preference.get("intent", "recommendation")
        mentioned_model_raw = preference.get("mentioned_model_raw", "unknown")
        mentioned_models_raw = preference.get("mentioned_models_raw", [])

        # -------------------------
        # Case 0: comparison
        # เช่น "เวฟ125ดีกว่าคลิกยังไง"
        # หรือ "PCX 160 กับ Forza 350 ต่างกันยังไง"
        # -------------------------
        if intent == "comparison" or len(mentioned_models_raw) >= 2:
            return self._handle_comparison(
                user_message=user_message,
                preference=preference,
                mentioned_models_raw=mentioned_models_raw,
            )

        # -------------------------
        # Case 1: similar_to_model
        # เช่น "แนะนำรถคล้าย PCX 160"
        #
        # ต้องมาก่อน price_lookup/info_lookup
        # เพราะถ้าข้อความมีคำว่า "ราคา" ในอนาคต แต่ intent เป็น similar_to_model
        # ก็ยังควรเข้าทาง similarity ก่อน
        # -------------------------
        if intent == "similar_to_model":
            return self._handle_similar_to_model(
                user_message=user_message,
                preference=preference,
                mentioned_model_raw=mentioned_model_raw,
                top_k=top_k,
            )

        # -------------------------
        # Case 2: ผู้ใช้พูดชื่อรุ่นตรง ๆ
        # เช่น "ขอข้อมูล PCX 160"
        # หรือ "Honda Click 160 ดีไหม"
        # หรือ "N-MAX ราคาเท่าไหร่"
        #
        # ถึง intent จะหลุดเป็น recommendation แต่ถ้ามี mentioned_model_raw
        # ให้ไป Catalog Resolver ก่อน ไม่ให้เข้า Item-Based
        # -------------------------
        if mentioned_model_raw not in [None, "", "unknown"]:
            if self._is_price_lookup_question(user_message):
                return self._handle_price_lookup(
                    user_message=user_message,
                    preference=preference,
                    mentioned_model_raw=mentioned_model_raw,
                )

            return self._handle_info_lookup(
                user_message=user_message,
                preference=preference,
                mentioned_model_raw=mentioned_model_raw,
            )

        # -------------------------
        # Case 3: recommendation ปกติ
        # เช่น "อยากได้รถขี่ในเมือง ประหยัดน้ำมัน"
        # หรือ "แนะนำรถราคาไม่เกิน 100000 ตัวแรงทรงเท่"
        #
        # หมายเหตุ:
        # ถ้าไม่มีชื่อรุ่น แม้มีคำว่าราคา/งบ ก็ยังเป็น recommendation
        # เพราะต้องให้ Item-Based หา candidate ตาม preference
        # -------------------------
        return self._handle_recommendation(
            user_message=user_message,
            preference=preference,
            top_k=top_k,
        )

    def _is_price_lookup_question(self, user_message: str) -> bool:
        """
        ตรวจว่าผู้ใช้ถามราคาของรุ่นเฉพาะหรือไม่

        ใช้เฉพาะกรณีที่ Router ตรวจเจอ mentioned_model_raw แล้ว
        เช่น:
        - n max ราคาเท่าไหร่
        - PCX 160 กี่บาท
        - Click 160 ราคา
        """

        text = user_message.lower().strip()

        price_keywords = [
            "ราคา",
            "ราคาเท่าไหร่",
            "เท่าไหร่",
            "กี่บาท",
            "กี่ตัง",
            "บาท",
        ]

        return any(keyword in text for keyword in price_keywords)

    def _handle_recommendation(
        self,
        user_message: str,
        preference: dict[str, Any],
        top_k: int,
    ) -> RecommendationRouteResult:
        candidates = self.item_based_recommender.recommend(
            preference=preference,
            top_k=top_k,
        )

        graph_item_ids = [
            candidate["item_id"]
            for candidate in candidates
        ]

        return RecommendationRouteResult(
            route="recommendation",
            user_message=user_message,
            preference=preference,
            catalog_result=None,
            candidates=candidates,
            graph_item_ids=graph_item_ids,
            response_type="recommendation_with_candidates",
            message=None,
        )

    def _handle_comparison(
        self,
        user_message: str,
        preference: dict[str, Any],
        mentioned_models_raw: list[str],
    ) -> RecommendationRouteResult:
        """
        ใช้กับคำถามเปรียบเทียบรถ 2 รุ่นขึ้นไป

        ตัวอย่าง:
        - เวฟ125ดีกว่าคลิกยังไง
        - PCX 160 กับ Forza 350 ต่างกันยังไง
        """

        resolved_results: list[CatalogResolveResult] = []
        missing_models: list[str] = []

        for raw_model in mentioned_models_raw:
            resolved = self.catalog_resolver.resolve(raw_model)

            if resolved.found:
                resolved_results.append(resolved)
            else:
                missing_models.append(raw_model)

        if missing_models:
            missing_name = ", ".join(missing_models)
            missing_result = self.catalog_resolver.resolve(missing_models[0])

            return self._out_of_catalog_result(
                route="out_of_catalog",
                user_message=user_message,
                preference=preference,
                catalog_result=missing_result,
                model_name=missing_name,
                response_type="out_of_catalog_comparison",
            )

        if len(resolved_results) < 2:
            fallback_model = mentioned_models_raw[0] if mentioned_models_raw else "รุ่นที่ระบุ"
            fallback_result = self.catalog_resolver.resolve(fallback_model)

            return self._out_of_catalog_result(
                route="out_of_catalog",
                user_message=user_message,
                preference=preference,
                catalog_result=fallback_result,
                model_name=fallback_model,
                response_type="out_of_catalog_comparison",
            )

        candidates = []
        graph_item_ids = []

        for index, resolved in enumerate(resolved_results, start=1):
            candidates.append(
                {
                    "rank": index,
                    "item_id": resolved.item_id,
                    "brand": resolved.brand,
                    "model": resolved.model,
                    "method": "comparison_lookup",
                }
            )

            graph_item_ids.append(resolved.item_id)

        return RecommendationRouteResult(
            route="comparison",
            user_message=user_message,
            preference=preference,
            catalog_result=None,
            candidates=candidates,
            graph_item_ids=graph_item_ids,
            response_type="comparison",
            message=None,
        )

    def _handle_price_lookup(
        self,
        user_message: str,
        preference: dict[str, Any],
        mentioned_model_raw: str,
    ) -> RecommendationRouteResult:
        """
        ใช้กับคำถามราคาของรุ่นเฉพาะ

        ตัวอย่าง:
        - n max ราคาเท่าไหร่
        - PCX 160 กี่บาท
        - Click 160 ราคา

        Flow:
        mentioned_model_raw
        → CatalogResolver
        → ถ้าเจอรุ่น ให้ response_generator ไปดึง price_est_thb จาก Items_Feature.csv
        → ไม่เข้า GraphRAG
        """

        resolved = self.catalog_resolver.resolve(mentioned_model_raw)

        if not resolved.found:
            return self._out_of_catalog_result(
                route="out_of_catalog",
                user_message=user_message,
                preference=preference,
                catalog_result=resolved,
                model_name=mentioned_model_raw,
                response_type="out_of_catalog_price_lookup",
            )

        candidate = {
            "rank": 1,
            "item_id": resolved.item_id,
            "brand": resolved.brand,
            "model": resolved.model,
            "method": "price_lookup",
        }

        return RecommendationRouteResult(
            route="price_lookup",
            user_message=user_message,
            preference=preference,
            catalog_result=resolved,
            candidates=[candidate],
            graph_item_ids=[],
            response_type="price_lookup",
            message=None,
        )

    def _handle_info_lookup(
        self,
        user_message: str,
        preference: dict[str, Any],
        mentioned_model_raw: str,
    ) -> RecommendationRouteResult:
        resolved = self.catalog_resolver.resolve(mentioned_model_raw)

        if not resolved.found:
            return self._out_of_catalog_result(
                route="out_of_catalog",
                user_message=user_message,
                preference=preference,
                catalog_result=resolved,
                model_name=mentioned_model_raw,
                response_type="out_of_catalog_info_lookup",
            )

        candidate = {
            "rank": 1,
            "item_id": resolved.item_id,
            "brand": resolved.brand,
            "model": resolved.model,
            "method": "catalog_lookup",
        }

        return RecommendationRouteResult(
            route="info_lookup",
            user_message=user_message,
            preference=preference,
            catalog_result=resolved,
            candidates=[candidate],
            graph_item_ids=[resolved.item_id],
            response_type="info_lookup_by_item_id",
            message=None,
        )

    def _handle_similar_to_model(
        self,
        user_message: str,
        preference: dict[str, Any],
        mentioned_model_raw: str,
        top_k: int,
    ) -> RecommendationRouteResult:
        resolved = self.catalog_resolver.resolve(mentioned_model_raw)

        if not resolved.found:
            return self._out_of_catalog_result(
                route="out_of_catalog",
                user_message=user_message,
                preference=preference,
                catalog_result=resolved,
                model_name=mentioned_model_raw,
                response_type="out_of_catalog_similarity",
            )

        candidates = self.similarity_service.get_similar_items(
            source_item_id=resolved.item_id,
            top_k=top_k,
        )

        graph_item_ids = [
            candidate["item_id"]
            for candidate in candidates
        ]

        return RecommendationRouteResult(
            route="similar_to_model",
            user_message=user_message,
            preference=preference,
            catalog_result=resolved,
            candidates=candidates,
            graph_item_ids=graph_item_ids,
            response_type="similar_items",
            message=None,
        )

    def _out_of_catalog_result(
        self,
        route: RouteName,
        user_message: str,
        preference: dict[str, Any],
        catalog_result: CatalogResolveResult,
        model_name: str,
        response_type: str,
    ) -> RecommendationRouteResult:
        message = (
            f"ตอนนี้ระบบยังไม่มีข้อมูลรุ่น {model_name} ในฐานข้อมูล "
            "จึงยังไม่สามารถดึงข้อมูลหรือเทียบความคล้ายของรุ่นนี้ได้โดยตรง"
        )

        return RecommendationRouteResult(
            route=route,
            user_message=user_message,
            preference=preference,
            catalog_result=catalog_result,
            candidates=[],
            graph_item_ids=[],
            response_type=response_type,
            message=message,
        )