from typing import Any

from app.db.neo4j import run_query
from app.services.recommendation.data_loader import RecommendationDataLoader


class GraphRetriever:
    """
    GraphRetriever ตัวจริงสำหรับ MotorAiAgent

    หน้าที่:
    - รับ item_id จาก Router เช่น ["I004", "I008", "I009"]
    - map item_id -> model name จาก Items_Feature.csv
    - query Neo4j ด้วย (:VehicleModel {name: model_name})
    - ดึง evidence รอบ ๆ VehicleModel แบบ 2 ชั้น
    - group evidence เป็นหมวดเพื่อส่งต่อให้ ResponseGenerator / Typhoon2

    ไม่ใช้ LangChain ในชั้นนี้
    เพราะชั้นนี้เป็น deterministic retrieval logic
    """

    def __init__(self, data_loader: RecommendationDataLoader | None = None):
        self.data_loader = data_loader or RecommendationDataLoader()

    def retrieve_by_item_ids(
        self,
        item_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        ดึง Graph evidence จากรายการ item_id

        Parameters:
            item_ids:
                item_id จาก recommendation layer เช่น ["I001", "I007"]

        Returns:
            list[dict]:
                Evidence ที่ group ตามรถแต่ละรุ่น
        """

        if not item_ids:
            return []

        item_mappings = self._map_item_ids_to_models(item_ids)

        valid_mappings = [
            item
            for item in item_mappings
            if item.get("found") is True and item.get("model")
        ]

        missing_mappings = [
            item
            for item in item_mappings
            if item.get("found") is False
        ]

        if not valid_mappings:
            return [
                self._build_missing_item_evidence(item)
                for item in missing_mappings
            ]

        model_names = [
            item["model"]
            for item in valid_mappings
        ]

        rows = self._query_vehicle_graph(model_names=model_names)

        grouped_evidence = self._group_graph_rows(
            rows=rows,
            item_mappings=valid_mappings,
        )

        missing_evidence = [
            self._build_missing_item_evidence(item)
            for item in missing_mappings
        ]

        # รักษาลำดับตาม item_ids เดิมจาก Router
        evidence_by_item_id = {
            item["item_id"]: item
            for item in grouped_evidence + missing_evidence
        }

        ordered_evidence = []
        for item_id in item_ids:
            if item_id in evidence_by_item_id:
                ordered_evidence.append(evidence_by_item_id[item_id])

        return ordered_evidence

    def _map_item_ids_to_models(
        self,
        item_ids: list[str],
    ) -> list[dict[str, Any]]:
        """
        map item_id ของ recommender เป็น model name จาก Items_Feature.csv

        ตัวอย่าง:
            I002 -> ADV 160
            I007 -> PCX 160
        """

        mappings: list[dict[str, Any]] = []

        for item_id in item_ids:
            item = self.data_loader.get_item_by_id(item_id)

            if item is None:
                mappings.append(
                    {
                        "found": False,
                        "item_id": str(item_id),
                        "brand": None,
                        "model": None,
                    }
                )
                continue

            mappings.append(
                {
                    "found": True,
                    "item_id": str(item.get("item_id")),
                    "brand": str(item.get("brand")),
                    "model": str(item.get("model")),
                    "csv_features": item,
                }
            )

        return mappings

    def _query_vehicle_graph(
        self,
        model_names: list[str],
    ) -> list[dict[str, Any]]:
        """
        Query Neo4j ด้วย VehicleModel.name

        ใช้ 2-hop pattern:
        VehicleModel -> entity -> semantic

        เพราะ Graph มีทั้ง feature โดยตรง และ semantic node ต่ออีกชั้น
        """

        cypher = """
            MATCH (m:VehicleModel)
            WHERE m.name IN $model_names
            OPTIONAL MATCH (m)-[r1]->(entity)
            OPTIONAL MATCH (entity)-[r2]->(semantic)
            RETURN
                m.name AS model,
                properties(m) AS model_props,

                type(r1) AS rel1,
                labels(entity) AS entity_labels,
                properties(entity) AS entity_props,
                entity.name AS entity_name,

                type(r2) AS rel2,
                labels(semantic) AS semantic_labels,
                properties(semantic) AS semantic_props,
                semantic.name AS semantic_name
            ORDER BY m.name, rel1, entity_name, rel2, semantic_name
        """

        return run_query(
            cypher=cypher,
            params={"model_names": model_names},
        )

    def _group_graph_rows(
        self,
        rows: list[dict[str, Any]],
        item_mappings: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        group Neo4j rows ให้เป็น evidence ต่อรุ่น
        """

        mapping_by_model = {
            item["model"]: item
            for item in item_mappings
        }

        evidence_by_model: dict[str, dict[str, Any]] = {}

        for item in item_mappings:
            model = item["model"]

            evidence_by_model[model] = {
                "item_id": item["item_id"],
                "found": True,
                "brand": item.get("brand"),
                "model": model,
                "graph_node_label": "VehicleModel",
                "graph_lookup_key": model,
                "evidence": self._empty_evidence(),
                "csv_features": item.get("csv_features", {}),
                "raw_graph_rows_count": 0,
            }

        for row in rows:
            model = row.get("model")

            if model not in evidence_by_model:
                continue

            evidence_by_model[model]["raw_graph_rows_count"] += 1

            rel1 = row.get("rel1")
            entity_name = row.get("entity_name")
            entity_labels = row.get("entity_labels") or []

            rel2 = row.get("rel2")
            semantic_name = row.get("semantic_name")
            semantic_labels = row.get("semantic_labels") or []

            if not entity_name:
                continue

            category = self._map_relationship_to_category(rel1)

            self._append_unique(
                evidence_by_model[model]["evidence"][category],
                entity_name,
            )

            # เก็บ semantic link เพิ่ม เพื่อให้ Typhoon2 มีเหตุผลเชิงความหมาย
            if semantic_name:
                semantic_entry = {
                    "from": entity_name,
                    "relationship": rel2,
                    "to": semantic_name,
                    "from_labels": entity_labels,
                    "to_labels": semantic_labels,
                }

                self._append_unique_dict(
                    evidence_by_model[model]["evidence"]["semantic_links"],
                    semantic_entry,
                )

        results = []

        for item in item_mappings:
            model = item["model"]
            evidence = evidence_by_model.get(model)

            if evidence is None:
                continue

            evidence["summary_text"] = self._build_summary_text(evidence)
            results.append(evidence)

        return results

    def _empty_evidence(self) -> dict[str, Any]:
        """
        evidence schema ที่ ResponseGenerator จะใช้ต่อ
        """

        return {
            "brand": [],
            "engine": [],
            "safety": [],
            "color": [],
            "style": [],
            "comfort": [],
            "storage": [],
            "performance": [],
            "use_case": [],
            "decision_factor": [],
            "efficiency": [],
            "target_user": [],
            "technology": [],
            "convenience": [],
            "maintenance": [],
            "durability": [],
            "brand_perception": [],
            "other_features": [],
            "semantic_links": [],
        }

    def _map_relationship_to_category(
        self,
        rel_type: str | None,
    ) -> str:
        """
        map relationship type จาก Graph เป็นหมวด evidence

        อ้างอิงจาก relationship จริงใน Neo4j เช่น:
        HAS_BRAND, HAS_ENGINE_CC, HAS_SAFETY_FEATURE, HAS_STYLE,
        HAS_COMFORT_FEATURE, HAS_STORAGE_FEATURE, HAS_PERFORMANCE,
        SUITABLE_FOR, DECISION_FACTOR ฯลฯ
        """

        rel = (rel_type or "").upper()

        if rel in ["HAS_BRAND", "HAS_MODEL"]:
            return "brand"

        if rel in ["HAS_ENGINE_CC", "HAS_ENGINE_SPEC"]:
            return "engine"

        if rel == "HAS_SAFETY_FEATURE":
            return "safety"

        if rel == "HAS_COLOR":
            return "color"

        if rel in ["HAS_STYLE", "HAS_COLOR_STYLE"]:
            return "style"

        if rel == "HAS_COMFORT_FEATURE":
            return "comfort"

        if rel == "HAS_STORAGE_FEATURE":
            return "storage"

        if rel == "HAS_PERFORMANCE":
            return "performance"

        if rel in ["SUITABLE_FOR", "SUPPORTS_USE_CASE"]:
            return "use_case"

        if rel in ["DECISION_FACTOR", "SUPPORTS_DECISION_FOR"]:
            return "decision_factor"

        if rel == "HAS_EFFICIENCY":
            return "efficiency"

        if rel == "TARGET_USER":
            return "target_user"

        if rel == "HAS_TECHNOLOGY":
            return "technology"

        if rel == "HAS_CONVENIENCE_FEATURE":
            return "convenience"

        if rel in ["MAINTENANCE", "MAINTENANCE_POINT"]:
            return "maintenance"

        if rel == "HAS_DURABILITY":
            return "durability"

        if rel == "HAS_BRAND_PERCEPTION":
            return "brand_perception"

        return "other_features"

    def _build_missing_item_evidence(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "item_id": item.get("item_id"),
            "found": False,
            "brand": None,
            "model": None,
            "graph_node_label": "VehicleModel",
            "graph_lookup_key": None,
            "evidence": self._empty_evidence(),
            "csv_features": {},
            "raw_graph_rows_count": 0,
            "summary_text": f"ไม่พบ item_id {item.get('item_id')} ใน Items_Feature.csv",
        }

    def _build_summary_text(
        self,
        item_evidence: dict[str, Any],
    ) -> str:
        """
        สร้างข้อความสรุปสั้น ๆ จาก evidence
        ใช้เป็น context ให้ ResponseGenerator
        """

        model = item_evidence.get("model", "unknown")
        brand = item_evidence.get("brand", "unknown")
        evidence = item_evidence.get("evidence", {})

        parts = [
        f"รุ่น: {brand} {model}",
    ]

        csv_features = item_evidence.get("csv_features", {})
        price = csv_features.get("price_est_thb", "unknown")

        if price not in [None, "", "unknown"]:
            try:
                parts.append(f"ราคาประมาณ: {int(price):,} บาท")
            except (ValueError, TypeError):
                parts.append(f"ราคาประมาณ: {price} บาท")

        engine = evidence.get("engine", [])
        if engine:
            parts.append(f"เครื่องยนต์: {', '.join(engine[:3])}")

        use_case = evidence.get("use_case", [])
        if use_case:
            parts.append(f"เหมาะกับ: {', '.join(use_case[:6])}")

        style = evidence.get("style", [])
        if style:
            parts.append(f"สไตล์: {', '.join(style[:5])}")

        comfort = evidence.get("comfort", [])
        if comfort:
            parts.append(f"ความสบาย: {', '.join(comfort[:5])}")

        storage = evidence.get("storage", [])
        if storage:
            parts.append(f"พื้นที่เก็บของ: {', '.join(storage[:4])}")

        performance = evidence.get("performance", [])
        if performance:
            parts.append(f"สมรรถนะ/การขับขี่: {', '.join(performance[:6])}")

        safety = evidence.get("safety", [])
        if safety:
            parts.append(f"ความปลอดภัย: {', '.join(safety[:5])}")

        decision_factor = evidence.get("decision_factor", [])
        if decision_factor:
            parts.append(f"ปัจจัยช่วยตัดสินใจ: {', '.join(decision_factor[:5])}")

        efficiency = evidence.get("efficiency", [])
        if efficiency:
            parts.append(f"ความประหยัด/ประสิทธิภาพ: {', '.join(efficiency[:5])}")

        technology = evidence.get("technology", [])
        if technology:
            parts.append(f"เทคโนโลยี: {', '.join(technology[:5])}")

        return "\n".join(parts)

    def _append_unique(
        self,
        target_list: list[str],
        value: str,
    ) -> None:
        if value and value not in target_list:
            target_list.append(value)

    def _append_unique_dict(
        self,
        target_list: list[dict[str, Any]],
        value: dict[str, Any],
    ) -> None:
        if value not in target_list:
            target_list.append(value)