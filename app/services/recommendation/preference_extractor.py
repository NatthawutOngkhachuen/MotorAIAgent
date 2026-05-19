import json
import os
from typing import Literal, Type

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.services.ollama_client import get_ollama_base_url, make_chat_ollama
from app.services.recommendation.schemas import (
    UserPreferenceSchema,
    ItemPreferenceSchema,
    ExtractPreferenceResult,
)

load_dotenv()


SchemaType = Literal["user_based", "ncf", "item_based"]


class PreferenceExtractorService:
    """
    Service สำหรับ Extract Preference JSON จากข้อความผู้ใช้

    Flow:
    user_message
      -> qwen ผ่าน Ollama + LangChain
      -> JSON ตาม Pydantic Schema
      -> Pydantic Validation
      -> Rule-based Normalization
      -> Prevent Over Guessing
      -> return preference dict
    """

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ):
        self.model_name = model_name or os.getenv("EXTRACTOR_MODEL", "qwen2.5:3b")
        self.base_url = base_url or get_ollama_base_url("EXTRACTOR")
        self.temperature = temperature

        self.llm = make_chat_ollama(
            model=self.model_name,
            prefix="EXTRACTOR",
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=600,
        )

    def extract(
        self,
        user_message: str,
        schema_type: SchemaType = "user_based",
        apply_rule_normalization: bool = True,
    ) -> ExtractPreferenceResult:
        """
        ใช้เรียกจากส่วนอื่นของระบบ

        schema_type:
        - user_based ใช้กับ User-Based CF
        - ncf ใช้ schema เดียวกับ user_based
        - item_based ใช้กับ Item-Based Recommender
        """

        schema_class = self._get_schema_class(schema_type)

        try:
            preference = self._extract_with_json_mode(
                user_message=user_message,
                schema_class=schema_class,
            )
        except Exception as e:
            print("[WARN] LLM extraction failed. Fallback to rule-based extraction.")
            print("[WARN]", e)

            preference = schema_class().model_dump()

        if apply_rule_normalization:
            preference = self._normalize_with_rules(
                user_message=user_message,
                preference=preference,
                schema_type=schema_type,
            )

            preference = self._prevent_over_guessing(
                user_message=user_message,
                preference=preference,
                schema_type=schema_type,
            )

        final_schema_type: Literal["user_based", "item_based"]
        final_schema_type = "item_based" if schema_type == "item_based" else "user_based"

        return ExtractPreferenceResult(
            model_name=self.model_name,
            schema_type=final_schema_type,
            raw_message=user_message,
            preference=preference,
        )

    def _extract_with_json_mode(
        self,
        user_message: str,
        schema_class: Type[BaseModel],
    ) -> dict:
        """
        Extract JSON ด้วย Ollama JSON Schema mode
        """

        schema_json = schema_class.model_json_schema()

        json_llm_kwargs = {
            "temperature": 0,
            "num_predict": 600,
        }
        if "ollama.com" not in self.base_url:
            json_llm_kwargs["format"] = schema_json

        json_llm = make_chat_ollama(
            model=self.model_name,
            prefix="EXTRACTOR",
            base_url=self.base_url,
            **json_llm_kwargs,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
คุณคือ Preference Extractor ของระบบแนะนำรถจักรยานยนต์ MotorAiAgent

ตอบเป็น JSON object เท่านั้น
ห้ามอธิบาย
ห้าม markdown
ห้ามใส่ข้อความอื่นนอก JSON
ถ้าไม่พบข้อมูล ให้ใช้ "unknown"
ห้ามเดา age_group หรือ gender ถ้าผู้ใช้ไม่ได้บอก
ห้ามเดา boolean เป็น false ถ้าผู้ใช้ไม่ได้บอกว่าไม่ต้องการสิ่งนั้น

Intent:
ถ้าผู้ใช้ขอให้แนะนำรถจากความต้องการ เช่น อยากได้/แนะนำ/หา/ต้องการ = recommendation
ถ้าผู้ใช้ถามข้อมูลรุ่นเฉพาะ เช่น ขอข้อมูล/รายละเอียด/สเปค/ดีไหม/ราคา/รีวิว = info_lookup
ถ้าผู้ใช้ขอรถคล้ายรุ่นใดรุ่นหนึ่ง เช่น คล้าย/เหมือน/ใกล้เคียง/เทียบกับ/แทน = similar_to_model
ถ้าผู้ใช้เปรียบเทียบรถหลายรุ่น เช่น ดีกว่า/ต่างกัน/เปรียบเทียบ/กับ/vs = comparison

mentioned_model_raw:
ให้ดึงชื่อรุ่นรถตัวแรกที่ผู้ใช้พูดถึง เช่น CBR, R7, PCX, N-MAX, Click, Wave
ถ้าไม่พบให้ใช้ unknown

mentioned_models_raw:
ให้ดึงรายชื่อรุ่นรถทั้งหมดที่ผู้ใช้พูดถึงเป็น list
เช่น "เวฟ125ดีกว่าคลิกยังไง" = ["Wave 125i", "Click 160"]
ถ้าไม่พบให้ใช้ []

Mapping:
Honda/ฮอนด้า = brand_preference Honda
Yamaha/ยามาฮ่า = brand_preference Yamaha

Click/Click 160/คลิก = mentioned_model Click 160
ADV/ADV 160 = mentioned_model ADV 160
Forza/Forza 350/ฟอร์ซ่า = mentioned_model Forza 350
Giorno/Giorno+/จอร์โน่ = mentioned_model Giorno+
Grand Filano/Grand Filano Hybrid/ฟีลาโน่ = mentioned_model Grand Filano Hybrid
N-MAX/NMAX/N MAX/เอ็นแม็กซ์ = mentioned_model N-MAX
PCX/PCX 160 = mentioned_model PCX 160
Scoopy/Scoopy i/สกู๊ปปี้ = mentioned_model Scoopy i
Wave/Wave 125i/เวฟ/เวฟ125 = mentioned_model Wave 125i

ขี่ในเมือง/รถติด/ซอกแซก = city
ใช้ทุกวัน/ไปทำงาน/ไปเรียน = daily
เดินทางไกล/ออกทริป/ต่างจังหวัด = long_distance
ส่งของ/เดลิเวอรี่/ไรเดอร์/grab = delivery

งบน้อย/งบไม่แรง/ราคาไม่แรง/ถูก/ไม่แพง = low
งบกลาง/กลางๆ/พอประมาณ = medium
งบสูง/แพงได้/ตัวท็อป/พรีเมียม = high

ประหยัดน้ำมัน/กินน้ำมันน้อย = fuel_saving true
มือใหม่/ขับง่าย/ขี่ง่าย/ควบคุมง่าย = easy_to_ride true
เก็บของ/ใส่ของ/ช่องเก็บของ = storage_need true

สปอร์ต/เท่/วัยรุ่น = style sporty
หรู/ดูดี/พรีเมียม = style premium
สวย/น่ารัก/แฟชั่น = style beauty

แรงนิด/แรงนิดนึง/แรงพอประมาณ = performance medium
แรงมาก/อัตราเร่งดี/เครื่องแรง = performance high

นั่งสบาย/ซ้อนสบาย/เบาะสบาย = comfort high
                    """,
                ),
                (
                    "human",
                    """
ข้อความผู้ใช้: {user_message}
                    """,
                ),
            ]
        )

        chain = prompt | json_llm

        response = chain.invoke(
            {
                "user_message": user_message,
            }
        )

        raw_text = response.content.strip()

        if not raw_text:
            print("========== EMPTY OLLAMA RESPONSE DEBUG ==========")
            print("model:", self.model_name)
            print("base_url:", self.base_url)
            print("response:", response)
            print("additional_kwargs:", getattr(response, "additional_kwargs", None))
            print("response_metadata:", getattr(response, "response_metadata", None))
            print("=================================================")

            raise ValueError(
                "Ollama returned empty response. "
                "Please check Ollama server, model name, or model output."
            )

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print("========== RAW LLM OUTPUT ==========")
            print(raw_text)
            print("====================================")
            raise ValueError(f"LLM did not return valid JSON: {e}") from e

        validated = schema_class.model_validate(parsed)

        return validated.model_dump()

    def _get_schema_class(
        self,
        schema_type: SchemaType,
    ) -> Type[BaseModel]:
        if schema_type in ["user_based", "ncf"]:
            return UserPreferenceSchema

        if schema_type == "item_based":
            return ItemPreferenceSchema

        raise ValueError(f"Unsupported schema_type: {schema_type}")

    def _detect_intent(
        self,
        text: str,
        mentioned_model_raw: str = "unknown",
        mentioned_models_raw: list[str] | None = None,
    ) -> str:
        mentioned_models_raw = mentioned_models_raw or []

        similar_keywords = [
            "คล้าย",
            "คล้ายๆ",
            "คล้าย ๆ",
            "เหมือน",
            "ใกล้เคียง",
            "เทียบกับ",
            "แทน",
            "similar",
        ]

        info_keywords = [
            "ขอข้อมูล",
            "ข้อมูล",
            "รายละเอียด",
            "สเปค",
            "spec",
            "ดีไหม",
            "ราคา",
            "ราคาเท่าไหร่",
            "เท่าไหร่",
            "กี่บาท",
            "บาท",
            "รีวิว",
            "เป็นยังไง",
        ]

        recommendation_keywords = [
            "อยากได้",
            "แนะนำ",
            "หา",
            "ต้องการ",
            "มีรถ",
            "รถที่",
            "เหมาะกับ",
            "ขอรถ",
            "มี",
            "มีแบบ",
        ]

        has_model = mentioned_model_raw not in [None, "", "unknown"]

        if len(mentioned_models_raw) >= 2:
            return "comparison"

        if any(word in text for word in similar_keywords):
            return "similar_to_model"

        if any(word in text for word in info_keywords) and has_model:
            return "info_lookup"

        if any(word in text for word in recommendation_keywords):
            return "recommendation"

        if has_model:
            return "info_lookup"

        return "recommendation"

    def _get_model_aliases(self) -> list[tuple[str, list[str]]]:
        return [
            ("Click 160", ["click 160", "click160", "click", "คลิก"]),
            ("ADV 160", ["adv 160", "adv160", "adv", "เอดีวี"]),
            ("Forza 350", ["forza 350", "forza350", "forza", "ฟอร์ซ่า"]),
            ("Giorno+", ["giorno+", "giorno plus", "giorno", "จอร์โน่", "จีออโน่"]),
            (
                "Grand Filano Hybrid",
                [
                    "grand filano hybrid",
                    "grand filano",
                    "filano",
                    "ฟีลาโน่",
                    "แกรนด์ฟีลาโน่",
                    "แกน ฟีลาโน่",
                ],
            ),
            ("N-MAX", ["n-max", "nmax", "n max", "เอ็นแม็กซ์", "เอนแมก"]),
            ("PCX 160", ["pcx 160", "pcx160", "pcx", "พีซีเอ็กซ์"]),
            ("Scoopy i", ["scoopy i", "scoopy", "สกู๊ปปี้", "สกู๊ปปี้ไอ"]),
            (
                "Wave 125i",
                [
                    "wave 125i",
                    "wave125i",
                    "wave 125",
                    "wave125",
                    "wave",
                    "เวฟ125i",
                    "เวฟ125",
                    "เวฟ 125",
                    "เวฟร้อยยิบห้า",
                    "เวฟร้อย",
                    "เวฟ",
                ],
            ),

            # รุ่นนอกระบบที่ผู้ใช้อาจถามถึง
            # ยังไม่ถือว่าอยู่ใน catalog จนกว่า CatalogResolver จะ resolve เจอ
            ("CBR", ["cbr"]),
            ("R7", ["r7"]),
            ("Aerox", ["aerox", "แอร็อกซ์"]),
            ("XMAX", ["xmax", "x-max"]),
            ("MT-15", ["mt-15", "mt15"]),
        ]

    def _detect_mentioned_models_raw(self, text: str) -> list[str]:
        """
        ดึงชื่อรุ่นรถทั้งหมดจากข้อความผู้ใช้
        โดยเรียงตามตำแหน่งที่เจอในประโยคจริง

        ตัวอย่าง:
        - เวฟ125ดีกว่าคลิกยังไง
          -> ["Wave 125i", "Click 160"]
        """

        found: list[tuple[int, str]] = []

        for canonical_name, aliases in self._get_model_aliases():
            best_position: int | None = None

            for alias in aliases:
                position = text.find(alias)

                if position == -1:
                    continue

                if best_position is None or position < best_position:
                    best_position = position

            if best_position is not None:
                found.append((best_position, canonical_name))

        found.sort(key=lambda item: item[0])

        ordered_models: list[str] = []
        for _, canonical_name in found:
            if canonical_name not in ordered_models:
                ordered_models.append(canonical_name)

        return ordered_models

    def _detect_mentioned_model_raw(self, text: str) -> str:
        """
        ดึงชื่อรุ่นรถดิบตัวแรกจากข้อความผู้ใช้
        ใช้เพื่อส่งต่อให้ CatalogResolver ตรวจ whitelist อีกที
        """

        models = self._detect_mentioned_models_raw(text)

        if models:
            return models[0]

        return "unknown"

    def _detect_brand_preference(self, text: str) -> str:
        if "honda" in text or "ฮอนด้า" in text:
            return "Honda"

        if "yamaha" in text or "ยามาฮ่า" in text:
            return "Yamaha"

        return "unknown"

    def _normalize_with_rules(
        self,
        user_message: str,
        preference: dict,
        schema_type: SchemaType,
    ) -> dict:
        """
        Rule-based normalization
        ใช้ช่วยแก้/เติม field ที่ชัดเจนจาก keyword ภาษาไทย
        """

        text = user_message.lower().strip()

        def set_if_unknown(key: str, value):
            if preference.get(key) in [None, "", "unknown"]:
                preference[key] = value

        def force_set(key: str, value):
            preference[key] = value

        usage_key = "usage_fit" if schema_type == "item_based" else "usage_type"

        # -------------------------
        # Intent / Raw Model
        # -------------------------
        detected_models_raw = self._detect_mentioned_models_raw(text)
        detected_raw_model = detected_models_raw[0] if detected_models_raw else "unknown"

        detected_intent = self._detect_intent(
            text=text,
            mentioned_model_raw=detected_raw_model,
            mentioned_models_raw=detected_models_raw,
        )

        if "intent" in preference:
            force_set("intent", detected_intent)

        if "mentioned_model_raw" in preference:
            force_set("mentioned_model_raw", detected_raw_model)

        if "mentioned_models_raw" in preference:
            force_set("mentioned_models_raw", detected_models_raw)

        # -------------------------
        # Brand / Mentioned Model
        # ใช้เฉพาะ item_based เพราะ UserPreferenceSchema ไม่มี field brand_preference/mentioned_model
        # -------------------------
        if schema_type == "item_based":
            detected_brand = self._detect_brand_preference(text)

            if detected_brand != "unknown":
                force_set("brand_preference", detected_brand)

            catalog_models = [
                "Click 160",
                "ADV 160",
                "Forza 350",
                "Giorno+",
                "Grand Filano Hybrid",
                "N-MAX",
                "PCX 160",
                "Scoopy i",
                "Wave 125i",
            ]

            if detected_raw_model in catalog_models:
                force_set("mentioned_model", detected_raw_model)

        # -------------------------
        # Usage
        # -------------------------
        city_keywords = ["ในเมือง", "ขี่เมือง", "ขับในเมือง", "รถติด", "ซอกแซก"]
        daily_keywords = ["ใช้ทุกวัน", "ขี่ทุกวัน", "ไปทำงาน", "ไปเรียน", "ใช้งานประจำ"]
        long_distance_keywords = ["เดินทางไกล", "ออกทริป", "ต่างจังหวัด", "ทางไกล", "ไกลๆ", "ไกล ๆ"]
        delivery_keywords = ["ส่งของ", "เดลิเวอรี่", "delivery", "แกร็บ", "grab", "ไรเดอร์"]

        if any(word in text for word in delivery_keywords):
            force_set(usage_key, "delivery")
        elif any(word in text for word in long_distance_keywords):
            force_set(usage_key, "long_distance")
        elif any(word in text for word in daily_keywords):
            set_if_unknown(usage_key, "daily")
        elif any(word in text for word in city_keywords):
            set_if_unknown(usage_key, "city")

        # -------------------------
        # Budget
        # -------------------------
        low_budget_keywords = [
            "งบน้อย",
            "งบไม่เยอะ",
            "งบไม่แรง",
            "ราคาไม่แรง",
            "ถูก",
            "ประหยัดงบ",
            "ไม่แพง",
        ]
        medium_budget_keywords = [
            "งบกลาง",
            "กลางๆ",
            "กลาง ๆ",
            "พอประมาณ",
            "ไม่ถูกไม่แพง",
        ]
        high_budget_keywords = [
            "งบสูง",
            "แพงได้",
            "ตัวท็อป",
            "พรีเมียม",
            "premium",
        ]

        if any(word in text for word in high_budget_keywords):
            force_set("budget_level", "high")
        elif any(word in text for word in medium_budget_keywords):
            force_set("budget_level", "medium")
        elif any(word in text for word in low_budget_keywords):
            force_set("budget_level", "low")

        # -------------------------
        # Boolean needs
        # -------------------------
        fuel_keywords = [
            "ประหยัดน้ำมัน",
            "กินน้ำมันน้อย",
            "น้ำมันประหยัด",
            "ไม่กินน้ำมัน",
        ]
        easy_keywords = [
            "มือใหม่",
            "ขับง่าย",
            "ขี่ง่าย",
            "ควบคุมง่าย",
            "ตัวเล็ก",
        ]
        storage_keywords = [
            "เก็บของ",
            "ใส่ของ",
            "ช่องเก็บของ",
            "บรรทุกของ",
            "มีที่เก็บ",
        ]

        if any(word in text for word in fuel_keywords):
            force_set("fuel_saving", True)

        if any(word in text for word in easy_keywords):
            force_set("easy_to_ride", True)

        if any(word in text for word in storage_keywords):
            force_set("storage_need", True)

        # -------------------------
        # Style / Performance / Comfort
        # -------------------------
        sporty_keywords = ["สปอร์ต", "เท่", "วัยรุ่น", "ทรงสปอร์ต"]
        premium_keywords = ["พรีเมียม", "หรู", "ดูดี", "ผู้ใหญ่", "premium"]
        beauty_keywords = ["สวย", "น่ารัก", "แฟชั่น", "ดีไซน์สวย"]

        medium_performance_keywords = [
            "แรงนิด",
            "แรงนิดนึง",
            "แรงพอประมาณ",
        ]
        high_performance_keywords = [
            "แรงมาก",
            "อัตราเร่งดี",
            "เร่งดี",
            "เครื่องแรง",
        ]

        comfort_keywords = [
            "นั่งสบาย",
            "ซ้อนสบาย",
            "ขับสบาย",
            "เบาะสบาย",
        ]

        if any(word in text for word in sporty_keywords):
            set_if_unknown("style", "sporty")

        if any(word in text for word in premium_keywords):
            set_if_unknown("style", "premium")

        if any(word in text for word in beauty_keywords):
            set_if_unknown("style", "beauty")

        if any(word in text for word in medium_performance_keywords):
            force_set("performance", "medium")
        elif any(word in text for word in high_performance_keywords):
            force_set("performance", "high")

        if any(word in text for word in comfort_keywords):
            set_if_unknown("comfort", "high")

        # -------------------------
        # Item-based specific
        # -------------------------
        if schema_type == "item_based":
            if "สกู๊ตเตอร์" in text or "scooter" in text:
                set_if_unknown("type", "scooter")

            if "ออโต้" in text or "automatic" in text:
                set_if_unknown("type", "automatic")

            if "ไฟฟ้า" in text or "ev" in text:
                set_if_unknown("type", "ev")

            if "125" in text:
                set_if_unknown("cc", "125")
            elif "150" in text:
                set_if_unknown("cc", "150")
            elif "155" in text:
                set_if_unknown("cc", "155")
            elif "160" in text:
                set_if_unknown("cc", "160")
            elif "350" in text:
                set_if_unknown("cc", "350")

        return preference

    def _prevent_over_guessing(
        self,
        user_message: str,
        preference: dict,
        schema_type: SchemaType,
    ) -> dict:
        """
        กัน LLM เดาค่าเกินจากข้อความผู้ใช้

        หลักการ:
        - ถ้าผู้ใช้ไม่ได้พูดถึง field นั้นโดยตรง ให้กลับเป็น unknown
        - ค่า False ควรเกิดเฉพาะเมื่อผู้ใช้บอกว่า "ไม่ต้องการ" ชัดเจน
        - ใช้หลัง _normalize_with_rules()
        """

        text = user_message.lower().strip()

        # -------------------------
        # Intent / Raw Model guard
        # -------------------------
        detected_models_raw = self._detect_mentioned_models_raw(text)
        detected_raw_model = detected_models_raw[0] if detected_models_raw else "unknown"

        detected_intent = self._detect_intent(
            text=text,
            mentioned_model_raw=detected_raw_model,
            mentioned_models_raw=detected_models_raw,
        )

        if "intent" in preference:
            preference["intent"] = detected_intent

        if "mentioned_model_raw" in preference:
            preference["mentioned_model_raw"] = detected_raw_model

        if "mentioned_models_raw" in preference:
            preference["mentioned_models_raw"] = detected_models_raw

        # -------------------------
        # Brand / Mentioned Model guard
        # ใช้เฉพาะ item_based
        # -------------------------
        if schema_type == "item_based":
            detected_brand = self._detect_brand_preference(text)
            detected_raw_model = self._detect_mentioned_model_raw(text)

            if detected_brand == "unknown":
                preference["brand_preference"] = "unknown"
            else:
                preference["brand_preference"] = detected_brand

            catalog_models = [
                "Click 160",
                "ADV 160",
                "Forza 350",
                "Giorno+",
                "Grand Filano Hybrid",
                "N-MAX",
                "PCX 160",
                "Scoopy i",
                "Wave 125i",
            ]

            if detected_raw_model in catalog_models:
                preference["mentioned_model"] = detected_raw_model
            else:
                preference["mentioned_model"] = "unknown"

        # -------------------------
        # Boolean guard
        # -------------------------
        storage_keywords = [
            "เก็บของ",
            "ใส่ของ",
            "ช่องเก็บของ",
            "บรรทุกของ",
            "มีที่เก็บ",
        ]
        no_storage_keywords = [
            "ไม่ต้องเก็บของ",
            "ไม่เอาที่เก็บของ",
            "ไม่จำเป็นต้องมีที่เก็บของ",
        ]

        if any(word in text for word in no_storage_keywords):
            preference["storage_need"] = False
        elif not any(word in text for word in storage_keywords):
            preference["storage_need"] = "unknown"

        fuel_keywords = [
            "ประหยัดน้ำมัน",
            "กินน้ำมันน้อย",
            "น้ำมันประหยัด",
            "ไม่กินน้ำมัน",
        ]
        no_fuel_keywords = [
            "ไม่เน้นประหยัดน้ำมัน",
            "ไม่สนประหยัดน้ำมัน",
        ]

        if any(word in text for word in no_fuel_keywords):
            preference["fuel_saving"] = False
        elif not any(word in text for word in fuel_keywords):
            preference["fuel_saving"] = "unknown"

        easy_keywords = [
            "มือใหม่",
            "ขับง่าย",
            "ขี่ง่าย",
            "ควบคุมง่าย",
            "ตัวเล็ก",
        ]
        no_easy_keywords = [
            "ไม่ต้องขับง่าย",
            "ขี่คล่องอยู่แล้ว",
        ]

        if any(word in text for word in no_easy_keywords):
            preference["easy_to_ride"] = False
        elif not any(word in text for word in easy_keywords):
            preference["easy_to_ride"] = "unknown"

        # -------------------------
        # Budget guard
        # -------------------------
        budget_keywords = [
            "งบน้อย",
            "งบไม่เยอะ",
            "งบไม่แรง",
            "ราคาไม่แรง",
            "ถูก",
            "ประหยัดงบ",
            "ไม่แพง",
            "งบกลาง",
            "กลางๆ",
            "กลาง ๆ",
            "พอประมาณ",
            "ไม่ถูกไม่แพง",
            "งบสูง",
            "แพงได้",
            "ตัวท็อป",
            "พรีเมียม",
            "premium",
        ]

        if not any(word in text for word in budget_keywords):
            preference["budget_level"] = "unknown"

        # -------------------------
        # Style guard
        # -------------------------
        style_keywords = [
            "สปอร์ต",
            "เท่",
            "วัยรุ่น",
            "ทรงสปอร์ต",
            "พรีเมียม",
            "หรู",
            "ดูดี",
            "ผู้ใหญ่",
            "premium",
            "สวย",
            "น่ารัก",
            "แฟชั่น",
            "ดีไซน์สวย",
        ]

        if not any(word in text for word in style_keywords):
            preference["style"] = "unknown"

        # -------------------------
        # Performance guard
        # -------------------------
        performance_keywords = [
            "แรงนิด",
            "แรงนิดนึง",
            "แรงพอประมาณ",
            "แรงมาก",
            "อัตราเร่ง",
            "เร่งดี",
            "เครื่องแรง",
        ]

        if not any(word in text for word in performance_keywords):
            preference["performance"] = "unknown"

        # -------------------------
        # Comfort guard
        # -------------------------
        comfort_keywords = [
            "นั่งสบาย",
            "ซ้อนสบาย",
            "ขับสบาย",
            "เบาะสบาย",
        ]

        if not any(word in text for word in comfort_keywords):
            preference["comfort"] = "unknown"

        return preference
