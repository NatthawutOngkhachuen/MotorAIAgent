import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CatalogResolveResult:
    """
    ผลลัพธ์จากการ resolve ชื่อรุ่นรถกับ catalog หลักของระบบ

    found:
        True  = รุ่นนี้มีอยู่ใน Items_Feature.csv
        False = รุ่นนี้ไม่มีใน catalog

    item_id:
        item_id ของรถในระบบ เช่น I001, I007

    model:
        ชื่อรุ่นมาตรฐานในระบบ เช่น PCX 160, N-MAX

    brand:
        แบรนด์ เช่น Honda, Yamaha

    raw_query:
        ข้อความรุ่นดิบที่ user พิมพ์ หรือที่ extractor ดึงมาได้

    status:
        in_catalog หรือ out_of_catalog
    """

    found: bool
    item_id: Optional[str]
    model: Optional[str]
    brand: Optional[str]
    raw_query: str
    status: str


class CatalogResolver:
    """
    Catalog Resolver สำหรับ MotorAiAgent

    หน้าที่:
    - โหลด Items_Feature.csv ครั้งเดียว
    - สร้าง index สำหรับค้นหาชื่อรุ่น
    - ตรวจว่า mentioned_model_raw อยู่ใน 9 รุ่นของระบบไหม
    - ห้ามเดาข้อมูลเอง ถ้าไม่เจอให้คืน out_of_catalog
    """

    _items_df_cache: Optional[pd.DataFrame] = None

    def __init__(self, items_csv_path: Optional[str] = None):
        self.items_csv_path = items_csv_path or os.getenv(
            "ITEMS_FEATURE_PATH",
            "data/Items_Feature.csv",
        )

        self.items_df = self._load_items_df()
        self.catalog_index = self._build_catalog_index()

    def _load_items_df(self) -> pd.DataFrame:
        """
        โหลด Items_Feature.csv แบบ cache
        """

        if CatalogResolver._items_df_cache is not None:
            return CatalogResolver._items_df_cache.copy()

        if not os.path.exists(self.items_csv_path):
            raise FileNotFoundError(
                f"Items_Feature.csv not found at: {self.items_csv_path}"
            )

        df = pd.read_csv(self.items_csv_path)

        required_columns = {"item_id", "brand", "model"}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            raise ValueError(
                f"Items_Feature.csv missing required columns: {missing_columns}"
            )

        CatalogResolver._items_df_cache = df.copy()
        return df

    def _normalize_text(self, text: str) -> str:
        """
        normalize ข้อความเพื่อใช้ match รุ่นรถ
        """

        if text is None:
            return ""

        text = str(text).lower().strip()

        # แปลงช่องว่างหลายตัวเป็นช่องว่างเดียว
        text = re.sub(r"\s+", " ", text)

        # ลบสัญลักษณ์บางตัวเพื่อช่วย match เช่น n-max กับ nmax
        text = text.replace("-", "")
        text = text.replace("_", "")
        text = text.replace("+", "plus")

        return text

    def _build_catalog_index(self) -> dict[str, dict]:
        """
        สร้าง index สำหรับ resolve ชื่อรุ่น

        key:
            alias ที่ normalize แล้ว เช่น pcx, pcx 160, nmax

        value:
            row data ของรถคันนั้น
        """

        index: dict[str, dict] = {}

        for _, row in self.items_df.iterrows():
            item_id = str(row["item_id"])
            brand = str(row["brand"])
            model = str(row["model"])

            row_data = {
                "item_id": item_id,
                "brand": brand,
                "model": model,
            }

            aliases = self._generate_aliases(model=model, brand=brand)

            for alias in aliases:
                normalized_alias = self._normalize_text(alias)
                if normalized_alias:
                    index[normalized_alias] = row_data

        return index

    def _generate_aliases(self, model: str, brand: str) -> list[str]:
        """
        สร้าง alias ของรุ่นที่มีอยู่ใน catalog

        ตรงนี้ยังเป็น rule-based เพื่อให้ deterministic
        ไม่ให้ LLM เป็นคนตัดสินว่ารุ่นอยู่ในระบบไหม
        """

        aliases = [
            model,
            f"{brand} {model}",
        ]

        model_lower = model.lower()

        if "click" in model_lower:
            aliases.extend(["click", "click 160", "honda click", "honda click 160", "คลิก"])

        elif "adv" in model_lower:
            aliases.extend(["adv", "adv 160", "honda adv", "honda adv 160"])

        elif "forza" in model_lower:
            aliases.extend(["forza", "forza 350", "honda forza", "honda forza 350", "ฟอร์ซ่า"])

        elif "giorno" in model_lower:
            aliases.extend(["giorno", "giorno+", "giorno plus", "honda giorno", "จอร์โน่"])

        elif "grand filano" in model_lower:
            aliases.extend(
                [
                    "grand filano",
                    "grand filano hybrid",
                    "filano",
                    "yamaha grand filano",
                    "ฟีลาโน่",
                ]
            )

        elif "n-max" in model_lower or "nmax" in model_lower:
            aliases.extend(
                [
                    "n-max",
                    "nmax",
                    "n max",
                    "yamaha n-max",
                    "yamaha nmax",
                    "เอ็นแม็กซ์",
                ]
            )

        elif "pcx" in model_lower:
            aliases.extend(["pcx", "pcx 160", "honda pcx", "honda pcx 160"])

        elif "scoopy" in model_lower:
            aliases.extend(["scoopy", "scoopy i", "honda scoopy", "สกู๊ปปี้"])

        elif "wave" in model_lower:
            aliases.extend(["wave", "wave 125i", "honda wave", "honda wave 125i", "เวฟ"])

        return aliases

    def resolve(self, mentioned_model_raw: str) -> CatalogResolveResult:
        """
        resolve ชื่อรุ่นดิบกับ catalog

        ถ้าเจอ:
            return found=True + item_id

        ถ้าไม่เจอ:
            return found=False + out_of_catalog
        """

        raw_query = mentioned_model_raw or "unknown"

        if raw_query in ["", "unknown", None]:
            return CatalogResolveResult(
                found=False,
                item_id=None,
                model=None,
                brand=None,
                raw_query="unknown",
                status="out_of_catalog",
            )

        normalized_query = self._normalize_text(raw_query)

        matched = self.catalog_index.get(normalized_query)

        if matched is None:
            return CatalogResolveResult(
                found=False,
                item_id=None,
                model=None,
                brand=None,
                raw_query=raw_query,
                status="out_of_catalog",
            )

        return CatalogResolveResult(
            found=True,
            item_id=matched["item_id"],
            model=matched["model"],
            brand=matched["brand"],
            raw_query=raw_query,
            status="in_catalog",
        )

    def resolve_from_preference(self, preference: dict) -> CatalogResolveResult:
        """
        ใช้กับ output จาก PreferenceExtractorService โดยตรง
        """

        mentioned_model_raw = preference.get("mentioned_model_raw", "unknown")
        return self.resolve(mentioned_model_raw)

    def get_all_catalog_items(self) -> list[dict]:
        """
        คืนรายการรถทั้งหมดใน catalog
        ใช้สำหรับ debug หรือ out-of-catalog response
        """

        return self.items_df[["item_id", "brand", "model"]].to_dict(orient="records")