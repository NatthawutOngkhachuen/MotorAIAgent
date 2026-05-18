import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class RecommendationDataLoader:
    """
    Data Loader สำหรับระบบ Recommendation

    หน้าที่:
    - โหลด Items_Feature.csv
    - โหลด item_based_similarity_baseline.csv
    - cache DataFrame ไว้ใน memory
    - service อื่นเรียกใช้ซ้ำได้ ไม่ต้องอ่านไฟล์ใหม่ทุก request
    """

    _items_feature_cache: Optional[pd.DataFrame] = None
    _similarity_cache: Optional[pd.DataFrame] = None

    def __init__(
        self,
        items_feature_path: str | None = None,
        similarity_path: str | None = None,
    ):
        self.items_feature_path = items_feature_path or os.getenv(
            "ITEMS_FEATURE_PATH",
            "data/Items_Feature.csv",
        )

        self.similarity_path = similarity_path or os.getenv(
            "ITEM_SIMILARITY_PATH",
            "data/item_based_similarity_baseline.csv",
        )

    def load_items_feature(self, force_reload: bool = False) -> pd.DataFrame:
        """
        โหลด Items_Feature.csv แบบ cache

        force_reload=True ใช้กรณีต้องการบังคับอ่านไฟล์ใหม่
        """

        if (
            RecommendationDataLoader._items_feature_cache is not None
            and not force_reload
        ):
            return RecommendationDataLoader._items_feature_cache.copy()

        file_path = self._resolve_path(self.items_feature_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Items_Feature.csv not found at: {file_path}")

        df = pd.read_csv(file_path)

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

        self._validate_columns(
            df=df,
            required_columns=required_columns,
            file_name="Items_Feature.csv",
        )

        RecommendationDataLoader._items_feature_cache = df.copy()
        return df.copy()

    def load_similarity_baseline(self, force_reload: bool = False) -> pd.DataFrame:
        """
        โหลด item_based_similarity_baseline.csv แบบ cache

        force_reload=True ใช้กรณีต้องการบังคับอ่านไฟล์ใหม่
        """

        if RecommendationDataLoader._similarity_cache is not None and not force_reload:
            return RecommendationDataLoader._similarity_cache.copy()

        file_path = self._resolve_path(self.similarity_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"item_based_similarity_baseline.csv not found at: {file_path}"
            )

        df = pd.read_csv(file_path)

        required_columns = {
            "source_item_id",
            "source_model",
            "similar_item_id",
            "similar_model",
            "similarity",
        }

        self._validate_columns(
            df=df,
            required_columns=required_columns,
            file_name="item_based_similarity_baseline.csv",
        )

        RecommendationDataLoader._similarity_cache = df.copy()
        return df.copy()

    def get_item_by_id(self, item_id: str) -> dict | None:
        """
        ดึงข้อมูล item จาก Items_Feature.csv ด้วย item_id
        """

        df = self.load_items_feature()

        matched = df[df["item_id"].astype(str) == str(item_id)]

        if matched.empty:
            return None

        return matched.iloc[0].to_dict()

    def get_all_items(self) -> list[dict]:
        """
        คืนรายการรถทั้งหมดใน catalog
        """

        df = self.load_items_feature()
        return df.to_dict(orient="records")

    def clear_cache(self) -> None:
        """
        ล้าง cache ทั้งหมด
        ใช้ตอน debug หรือหลังแก้ CSV แล้วอยากโหลดใหม่
        """

        RecommendationDataLoader._items_feature_cache = None
        RecommendationDataLoader._similarity_cache = None

    def _resolve_path(self, path: str) -> Path:
        """
        แปลง path ให้เป็น absolute path จาก root project

        รองรับทั้ง:
        - data/Items_Feature.csv
        - E:/.../Items_Feature.csv
        """

        file_path = Path(path)

        if file_path.is_absolute():
            return file_path

        project_root = Path.cwd()
        return project_root / file_path

    def _validate_columns(
        self,
        df: pd.DataFrame,
        required_columns: set[str],
        file_name: str,
    ) -> None:
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"{file_name} missing required columns: {missing_columns}"
            )