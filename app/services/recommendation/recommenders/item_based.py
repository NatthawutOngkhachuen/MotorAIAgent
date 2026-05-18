import os
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class ItemBasedRecommender:
    _items_df_cache: pd.DataFrame | None = None

    def __init__(self, items_csv_path: str | None = None):
        self.items_csv_path = items_csv_path or os.getenv(
            "ITEMS_FEATURE_PATH",
            "data/Items_Feature.csv",
        )
        self.items_df = self._load_items_df()

    def _load_items_df(self) -> pd.DataFrame:
        if ItemBasedRecommender._items_df_cache is not None:
            return ItemBasedRecommender._items_df_cache.copy()

        if not os.path.exists(self.items_csv_path):
            raise FileNotFoundError(
                f"Items_Feature.csv not found at: {self.items_csv_path}"
            )

        df = pd.read_csv(self.items_csv_path)

        required_columns = {
            "item_id",
            "brand",
            "model",
            "cc",
            "price_est_thb",
            "type",
            "budget_level",
            "usage_fit",
            "style",
            "performance",
            "comfort",
            "easy_to_ride",
            "fuel_saving",
            "storage_need",
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Items_Feature.csv missing required columns: {missing_columns}"
            )

        ItemBasedRecommender._items_df_cache = df.copy()
        return df

    def recommend(
        self,
        preference: dict[str, Any],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:

        top_k = self._normalize_top_k(top_k)

        scored_items = []

        for _, item in self.items_df.iterrows():
            score_detail = self._score_item(
                preference=preference,
                item=item,
            )

            scored_items.append(
                {
                    "item_id": str(item["item_id"]),
                    "brand": str(item["brand"]),
                    "model": str(item["model"]),
                    "score": score_detail["score"],
                    "score_detail": score_detail["detail"],
                }
            )

        scored_items = sorted(
            scored_items,
            key=lambda x: x["score"],
            reverse=True,
        )

        top_items = scored_items[:top_k]

        results = []
        for index, item in enumerate(top_items, start=1):
            results.append(
                {
                    "rank": index,
                    "item_id": item["item_id"],
                    "model": item["model"],
                    "score": round(float(item["score"]), 4),
                    "method": "item_based",
                    "score_detail": item["score_detail"],
                }
            )

        return results

    def _normalize_top_k(self, top_k: int) -> int:
        
        if top_k in [1, 3, 5]:
            return top_k

        return 3

    def _score_item(
        self,
        preference: dict[str, Any],
        item: pd.Series,
    ) -> dict[str, Any]:
        """
        คิดคะแนนแบบ weighted feature matching

        หลักการ:
        - ถ้า preference เป็น unknown จะไม่นำ field นั้นมาคิดคะแนน
        - ถ้า match ได้ จะบวก matched_weight
        - final_score = matched_weight / total_weight
        """

        total_weight = 0.0
        matched_weight = 0.0
        detail = {}

        def add_score(
            field_name: str,
            pref_value: Any,
            item_value: Any,
            weight: float,
        ):
            nonlocal total_weight, matched_weight, detail

            if self._is_unknown(pref_value):
                detail[field_name] = {
                    "preference": pref_value,
                    "item": item_value,
                    "matched": "skipped_unknown",
                    "weight": weight,
                }
                return

            total_weight += weight

            is_matched = self._is_match(pref_value, item_value)

            if is_matched:
                matched_weight += weight

            detail[field_name] = {
                "preference": pref_value,
                "item": item_value,
                "matched": is_matched,
                "weight": weight,
            }

        # -------------------------
        # Brand preference
        # -------------------------
        add_score(
            field_name="brand_preference",
            pref_value=preference.get("brand_preference", "unknown"),
            item_value=item.get("brand", "unknown"),
            weight=0.50,
        )

        # -------------------------
        # Item-Based features
        # -------------------------
        add_score(
            field_name="budget_level",
            pref_value=preference.get("budget_level", "unknown"),
            item_value=item.get("budget_level", "unknown"),
            weight=1.20,
        )

        add_score(
            field_name="usage_fit",
            pref_value=preference.get("usage_fit", "unknown"),
            item_value=item.get("usage_fit", "unknown"),
            weight=1.50,
        )

        add_score(
            field_name="style",
            pref_value=preference.get("style", "unknown"),
            item_value=item.get("style", "unknown"),
            weight=0.90,
        )

        add_score(
            field_name="performance",
            pref_value=preference.get("performance", "unknown"),
            item_value=item.get("performance", "unknown"),
            weight=0.90,
        )

        add_score(
            field_name="comfort",
            pref_value=preference.get("comfort", "unknown"),
            item_value=item.get("comfort", "unknown"),
            weight=0.90,
        )

        add_score(
            field_name="easy_to_ride",
            pref_value=preference.get("easy_to_ride", "unknown"),
            item_value=item.get("easy_to_ride", "unknown"),
            weight=1.00,
        )

        add_score(
            field_name="fuel_saving",
            pref_value=preference.get("fuel_saving", "unknown"),
            item_value=item.get("fuel_saving", "unknown"),
            weight=1.20,
        )

        add_score(
            field_name="storage_need",
            pref_value=preference.get("storage_need", "unknown"),
            item_value=item.get("storage_need", "unknown"),
            weight=0.80,
        )

        add_score(
            field_name="type",
            pref_value=preference.get("type", "unknown"),
            item_value=item.get("type", "unknown"),
            weight=0.70,
        )

        add_score(
            field_name="cc",
            pref_value=preference.get("cc", "unknown"),
            item_value=item.get("cc", "unknown"),
            weight=0.60,
        )

        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = matched_weight / total_weight

        return {
            "score": final_score,
            "detail": detail,
        }

    def _is_unknown(self, value: Any) -> bool:
        return value in [None, "", "unknown"]

    def _is_match(self, pref_value: Any, item_value: Any) -> bool:
        
        if self._is_unknown(pref_value) or self._is_unknown(item_value):
            return False

        normalized_pref = self._normalize_value(pref_value)
        normalized_item = self._normalize_value(item_value)

        # match ตรง ๆ
        if normalized_pref == normalized_item:
            return True

        # match กรณี item feature มีหลายค่า เช่น city,daily
        item_values = self._split_multi_values(item_value)
        if normalized_pref in item_values:
            return True

        # match กรณี item feature เป็นข้อความยาว เช่น sport automatic scooter
        pref_text = str(pref_value).lower().strip()
        item_text = str(item_value).lower().strip()

        if pref_text and pref_text in item_text:
            return True

        return False

    def _normalize_value(self, value: Any) -> str:

        if isinstance(value, bool):
            return "true" if value else "false"

        if isinstance(value, (int, float)):
            numeric_value = float(value)

            if numeric_value == 1:
                return "true"

            if numeric_value == 0:
                return "false"

            if numeric_value.is_integer():
                return str(int(numeric_value))

            return str(numeric_value)

        text = str(value).lower().strip()

        if text in ["true", "1", "yes", "y"]:
            return "true"

        if text in ["false", "0", "no", "n"]:
            return "false"

        text = text.replace("-", "")
        text = text.replace("_", "")
        text = text.replace(" ", "")

        return text

    def _split_multi_values(self, value: Any) -> set[str]:
        if self._is_unknown(value):
            return set()

        raw_text = str(value).lower().strip()

        parts = []

        # แยกด้วย comma เช่น city,daily
        parts.extend(raw_text.split(","))

        # แยกด้วย space เช่น sport automatic scooter
        parts.extend(raw_text.split())

        normalized_parts = {
            self._normalize_value(part)
            for part in parts
            if part.strip()
        }

        return normalized_parts