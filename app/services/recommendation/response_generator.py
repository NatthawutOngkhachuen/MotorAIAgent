import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.services.recommendation.data_loader import RecommendationDataLoader
from app.services.recommendation.router import RecommendationRouteResult

load_dotenv()


class ResponseGenerator:
    """
    Response Generator สำหรับ MotorAiAgent

    หน้าที่:
    - รับ user_message
    - รับ route_result จาก RecommendationRouter
    - รับ graph_evidence จาก GraphRetriever
    - ใช้ Typhoon2 8B สร้างคำตอบสุดท้ายภาษาไทย

    หลักสำคัญ:
    - ตอบจาก candidates + graph_evidence เท่านั้น
    - ห้ามแต่งรุ่นนอกระบบ
    - ห้ามแต่งข้อมูลที่ไม่มีใน evidence
    - out_of_catalog ใช้ template ทันที ไม่เรียก LLM เพื่อลด latency
    - price_lookup ใช้ template ทันที ไม่เรียก GraphRAG/LLM เพราะตอบจาก Items_Feature.csv ได้เลย
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.4,
        num_predict: int = 400,
    ):
        self.model_name = model_name or os.getenv("GENERATOR_MODEL", "typhoon2")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.temperature = temperature
        self.num_predict = num_predict

        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

    def generate(
        self,
        user_message: str,
        route_result: RecommendationRouteResult,
        graph_evidence: list[dict[str, Any]],
    ) -> str:
        """
        สร้างคำตอบสุดท้ายให้ user

        out_of_catalog:
            ตอบ template ทันที ไม่เรียก LLM

        price_lookup:
            ตอบราคาจาก Items_Feature.csv ทันที ไม่เรียก GraphRAG/LLM

        route อื่น:
            ใช้ Typhoon2 รวม context จาก candidates + graph_evidence
        """

        if route_result.route == "out_of_catalog":
            return self._generate_out_of_catalog_response(route_result)

        if route_result.response_type == "price_lookup":
            return self._generate_price_lookup_response(route_result)

        if not graph_evidence:
            return (
                "ตอนนี้ระบบยังไม่พบข้อมูลรถที่เกี่ยวข้องในฐานข้อมูลครับ "
                "จึงยังไม่สามารถสรุปคำแนะนำจาก GraphRAG ได้"
            )

        system_prompt = self._build_system_prompt()
        human_prompt = self._build_human_prompt(
            user_message=user_message,
            route_result=route_result,
            graph_evidence=graph_evidence,
        )

        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt),
            ]
        )

        return response.content.strip()

    def _build_system_prompt(self) -> str:
        return """
คุณคือ MotorAiAgent เซลล์แนะนำรถจักรยานยนต์ภาษาไทยที่เป็นกันเอง สุภาพ และคุยเหมือนให้คำปรึกษาหน้าร้าน

สไตล์การตอบ:
- ตอบเป็นภาษาไทย ใช้คำว่า "ครับ"
- เป็นกันเอง ดูพึ่งพาได้ ไม่แข็งเหมือนรายงาน
- อธิบายสั้น ชัด เหมือนเซลล์ช่วยลูกค้าตัดสินใจ
- แนะนำเป็นข้อ ๆ ตามลำดับ candidates
- แต่ละรุ่นให้เหตุผลสั้น ๆ จาก GRAPH_EVIDENCE เท่านั้น
- ถ้าเป็นการเปรียบเทียบ ให้เทียบแบบเป็นกลาง บอกว่าแต่ละรุ่นเด่นคนละด้าน ไม่ตอบเอียงโดยไม่มีข้อมูล
- ปิดท้ายด้วย "ถ้าให้ผมช่วยเลือกตัวเด่น..." แล้วเลือกรุ่นอันดับ 1 เฉพาะกรณี recommendation หรือ similar_to_model

กฎห้ามผิด:
- ใช้เฉพาะรุ่นใน CANDIDATES เท่านั้น
- ห้ามเพิ่มรุ่นอื่นเอง
- ห้ามแต่งสเปก ราคา หรือฟีเจอร์ที่ไม่มีในข้อมูล
- ถ้าข้อมูลไม่มี ไม่ต้องพูดถึง
- ตอบกระชับ ไม่เกิน 3 รุ่น
        """.strip()

    def _build_human_prompt(
        self,
        user_message: str,
        route_result: RecommendationRouteResult,
        graph_evidence: list[dict[str, Any]],
    ) -> str:
        compact_context = {
            "user_message": user_message,
            "route": route_result.route,
            "response_type": route_result.response_type,
            "preference": route_result.preference,
            "candidates": self._compact_candidates(route_result.candidates),
            "catalog_result": self._compact_catalog_result(route_result),
            "graph_evidence": self._compact_graph_evidence(graph_evidence),
        }

        route_instruction = self._get_route_instruction(route_result)

        return f"""
คำถามผู้ใช้:
{user_message}

ชนิดคำตอบที่ต้องสร้าง:
{route_instruction}

ข้อมูลสำหรับตอบ:
{json.dumps(compact_context, ensure_ascii=False)}

โปรดสร้างคำตอบสุดท้ายให้ผู้ใช้ โดยตอบแบบกระชับและอิงข้อมูลด้านบนเท่านั้น
        """.strip()

    def _get_route_instruction(
        self,
        route_result: RecommendationRouteResult,
    ) -> str:
        if route_result.route == "recommendation":
            return (
                "ผู้ใช้ต้องการคำแนะนำรถจากความต้องการทั่วไป "
                "ให้แนะนำตาม candidates เรียงตาม rank "
                "แต่ละรุ่นอธิบายไม่เกิน 1 ประโยค และคำตอบรวมไม่เกิน 10 บรรทัด"
            )

        if route_result.route == "info_lookup":
            return (
                "ผู้ใช้ถามข้อมูลรถรุ่นเฉพาะ "
                "ให้ตอบข้อมูลของรุ่นนั้นโดยตรงจาก graph_evidence"
            )

        if route_result.route == "similar_to_model":
            source_model = None
            if route_result.catalog_result:
                source_model = route_result.catalog_result.model

            return (
                f"ผู้ใช้ต้องการรถที่คล้ายกับ {source_model or 'รุ่นต้นทาง'} "
                "ให้แนะนำ candidates ที่ similarity service คืนมา พร้อมเหตุผลจาก graph_evidence"
            )

        if route_result.route == "comparison":
            return (
                "ผู้ใช้ต้องการเปรียบเทียบรถหลายรุ่น "
                "ให้เปรียบเทียบเฉพาะรุ่นใน candidates โดยใช้ graph_evidence ของแต่ละรุ่น "
                "บอกจุดเด่นของแต่ละรุ่น และสรุปว่าแต่ละรุ่นเหมาะกับใคร "
                "ห้ามตอบว่ารุ่นใดดีกว่าแบบฟันธงถ้า evidence ไม่พอ"
            )

        return "ตอบจากข้อมูลที่มีเท่านั้น"

    def _compact_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        ลดขนาด candidates ก่อนส่งเข้า LLM
        ไม่ส่ง score_detail เพราะยาวเกินและไม่จำเป็นต่อการตอบ
        """

        compact = []

        for candidate in candidates:
            item = {
                "rank": candidate.get("rank"),
                "item_id": candidate.get("item_id"),
                "brand": candidate.get("brand"),
                "model": candidate.get("model"),
                "method": candidate.get("method"),
            }

            if "price_est_thb" in candidate:
                item["price_est_thb"] = candidate.get("price_est_thb")

            if "score" in candidate:
                item["score"] = candidate.get("score")

            if "similarity" in candidate:
                item["similarity"] = candidate.get("similarity")
                item["source_model"] = candidate.get("source_model")

            compact.append(item)

        return compact

    def _compact_catalog_result(
        self,
        route_result: RecommendationRouteResult,
    ) -> dict[str, Any] | None:
        if route_result.catalog_result is None:
            return None

        catalog = route_result.catalog_result

        return {
            "found": catalog.found,
            "item_id": catalog.item_id,
            "brand": catalog.brand,
            "model": catalog.model,
            "raw_query": catalog.raw_query,
            "status": catalog.status,
        }

    def _compact_graph_evidence(
        self,
        graph_evidence: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        ลด graph evidence ให้พอดีกับ prompt
        ส่งเฉพาะหมวดสำคัญ + summary_text
        """

        compact = []

        for item in graph_evidence:
            evidence = item.get("evidence", {})

            compact.append(
                {
                    "item_id": item.get("item_id"),
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "found": item.get("found"),
                    "summary_text": item.get("summary_text"),
                    "evidence": {
                        "engine": self._limit_list(evidence.get("engine", []), 3),
                        "use_case": self._limit_list(evidence.get("use_case", []), 4),
                        "style": self._limit_list(evidence.get("style", []), 3),
                        "comfort": self._limit_list(evidence.get("comfort", []), 3),
                        "storage": self._limit_list(evidence.get("storage", []), 3),
                        "performance": self._limit_list(evidence.get("performance", []), 4),
                        "safety": self._limit_list(evidence.get("safety", []), 3),
                        "decision_factor": self._limit_list(
                            evidence.get("decision_factor", []),
                            4,
                        ),
                        "efficiency": self._limit_list(evidence.get("efficiency", []), 2),
                        "technology": self._limit_list(evidence.get("technology", []), 3),
                        "convenience": self._limit_list(evidence.get("convenience", []), 3),
                    },
                }
            )

        return compact

    def _limit_list(
        self,
        values: list[Any],
        limit: int,
    ) -> list[Any]:
        if not values:
            return []

        return values[:limit]

    def _generate_price_lookup_response(
        self,
        route_result: RecommendationRouteResult,
    ) -> str:
        """
        price_lookup ไม่เรียก LLM
        ใช้ราคาจาก Items_Feature.csv โดยตรง

        ใช้กับคำถามเช่น:
        - N-MAX ราคาเท่าไหร่
        - PCX 160 กี่บาท
        - Click 160 ราคา
        """

        if not route_result.catalog_result:
            return "ตอนนี้ยังไม่พบข้อมูลรุ่นนี้ในฐานข้อมูลครับ"

        item_id = route_result.catalog_result.item_id
        brand = route_result.catalog_result.brand
        model = route_result.catalog_result.model

        if not item_id:
            return f"ตอนนี้ยังไม่พบข้อมูลราคาของ {brand} {model} ในฐานข้อมูลครับ"

        loader = RecommendationDataLoader()
        item = loader.get_item_by_id(item_id)

        if item is None:
            return f"ตอนนี้ยังไม่พบข้อมูลราคาของ {brand} {model} ในฐานข้อมูลครับ"

        price = item.get("price_est_thb", "unknown")

        if price in [None, "", "unknown"]:
            return f"ตอนนี้ยังไม่มีข้อมูลราคาของ {brand} {model} ในฐานข้อมูลครับ"

        try:
            price_text = f"{int(price):,}"
        except (ValueError, TypeError):
            price_text = str(price)

        return (
            f"{brand} {model} ราคาประมาณ {price_text} บาทครับ\n"
            "ราคานี้เป็นข้อมูลประมาณจากฐานข้อมูลของระบบนะครับ"
        )

    def _generate_out_of_catalog_response(
        self,
        route_result: RecommendationRouteResult,
    ) -> str:
        """
        out_of_catalog ไม่เรียก LLM เพื่อประหยัดเวลาและกัน hallucination
        """

        model_name = "รุ่นนี้"

        if route_result.catalog_result and route_result.catalog_result.raw_query:
            model_name = route_result.catalog_result.raw_query

        if route_result.response_type == "out_of_catalog_similarity":
            return (
                f"ตอนนี้ระบบยังไม่มีข้อมูลรุ่น {model_name} ในฐานข้อมูลครับ "
                "จึงยังไม่สามารถเทียบความคล้ายจากรุ่นนี้ได้โดยตรง\n\n"
                "ระบบสามารถแนะนำได้จากรุ่นที่มีอยู่ในฐานข้อมูลเท่านั้น "
                "ถ้าคุณบอกลักษณะที่ต้องการ เช่น ขี่ในเมือง ประหยัดน้ำมัน สปอร์ต "
                "หรือเดินทางไกล ผมสามารถแนะนำรุ่นที่เหมาะจากฐานข้อมูลให้ได้ครับ"
            )

        if route_result.response_type == "out_of_catalog_price_lookup":
            return (
                f"ตอนนี้ระบบยังไม่มีข้อมูลรุ่น {model_name} ในฐานข้อมูลครับ "
                "จึงยังไม่สามารถบอกราคาของรุ่นนี้ได้โดยตรง\n\n"
                "ระบบสามารถบอกราคาได้เฉพาะรุ่นที่อยู่ในฐานข้อมูลเท่านั้นครับ"
            )

        if route_result.response_type == "out_of_catalog_comparison":
            return (
                f"ตอนนี้ระบบยังไม่มีข้อมูลครบสำหรับรุ่น {model_name} ในฐานข้อมูลครับ "
                "จึงยังไม่สามารถเปรียบเทียบรุ่นนี้ได้โดยตรง\n\n"
                "ระบบสามารถเปรียบเทียบได้เฉพาะรุ่นที่อยู่ในฐานข้อมูลเท่านั้นครับ"
            )

        return (
            f"ตอนนี้ระบบยังไม่มีข้อมูลรุ่น {model_name} ในฐานข้อมูลครับ "
            "จึงยังไม่สามารถดึงข้อมูลของรุ่นนี้ได้โดยตรง\n\n"
            "ระบบสามารถให้ข้อมูลและแนะนำได้เฉพาะรุ่นที่อยู่ในฐานข้อมูลเท่านั้นครับ"
        )