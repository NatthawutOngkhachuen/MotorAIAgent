from typing import Literal, Union
from pydantic import BaseModel, Field, ConfigDict


BoolUnknown = Union[bool, Literal["unknown"]]

IntentType = Literal[
    "recommendation",
    "info_lookup",
    "similar_to_model",
    "comparison",
    "unknown",
]


class UserPreferenceSchema(BaseModel):
    """
    Schema สำหรับ User-Based และ NCF
    ใช้กรณีที่ recommender ต้องอิงข้อมูลผู้ใช้ร่วมกับ preference
    """

    model_config = ConfigDict(extra="ignore")

    intent: IntentType = Field(
        default="unknown",
        description="เจตนาของผู้ใช้ เช่น recommendation, info_lookup, similar_to_model หรือ unknown",
    )
    mentioned_model_raw: str = Field(
        default="unknown",
        description="ชื่อรุ่นรถดิบที่ผู้ใช้พิมพ์ เช่น CBR, R7, PCX, N-MAX หรือ unknown",
    )
    mentioned_models_raw: list[str] = Field(
        default_factory=list,
        description="รายชื่อรุ่นรถทั้งหมดที่ผู้ใช้พูดถึง เช่น ['Wave 125i', 'Click 160']",
    )
    age_group: Literal["teen", "young_adult", "adult", "unknown"] = Field(
        default="unknown",
        description="ช่วงอายุของผู้ใช้ เช่น teen, young_adult, adult หรือ unknown ถ้าไม่ทราบ",
    )
    gender: Literal["male", "female", "unknown"] = Field(
        default="unknown",
        description="เพศของผู้ใช้ ถ้าไม่พบให้เป็น unknown",
    )
    budget_level: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับงบประมาณ low=งบน้อย, medium=งบกลาง, high=งบสูง",
    )
    usage_type: Literal[
        "city",
        "long_distance",
        "daily",
        "delivery",
        "unknown",
    ] = Field(
        default="unknown",
        description="ลักษณะการใช้งาน เช่น city, daily, long_distance, delivery",
    )
    style: Literal["beauty", "sporty", "premium", "unknown"] = Field(
        default="unknown",
        description="สไตล์รถที่ชอบ เช่น beauty, sporty, premium",
    )
    performance: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับความแรงหรือ performance ที่ต้องการ",
    )
    comfort: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับความสบายที่ต้องการ",
    )
    easy_to_ride: BoolUnknown = Field(
        default="unknown",
        description="ต้องการรถขับง่ายหรือเหมาะกับมือใหม่หรือไม่",
    )
    fuel_saving: BoolUnknown = Field(
        default="unknown",
        description="ต้องการรถประหยัดน้ำมันหรือไม่",
    )
    storage_need: BoolUnknown = Field(
        default="unknown",
        description="ต้องการพื้นที่เก็บของหรือไม่",
    )


class ItemPreferenceSchema(BaseModel):
    """
    Schema สำหรับ Item-Based
    ใช้กรณี Unknown User หรือไม่ต้องอิงข้อมูลส่วนตัวผู้ใช้
    """

    model_config = ConfigDict(extra="ignore")

    intent: IntentType = Field(
        default="unknown",
        description="เจตนาของผู้ใช้ เช่น recommendation, info_lookup, similar_to_model หรือ unknown",
    )
    mentioned_model_raw: str = Field(
        default="unknown",
        description="ชื่อรุ่นรถดิบที่ผู้ใช้พิมพ์ เช่น CBR, R7, PCX, N-MAX หรือ unknown",
    )
    mentioned_models_raw: list[str] = Field(
        default_factory=list,
        description="รายชื่อรุ่นรถทั้งหมดที่ผู้ใช้พูดถึง เช่น ['Wave 125i', 'Click 160']",
    )

    brand_preference: str = Field(
        default="unknown",
        description="แบรนด์รถที่ผู้ใช้พูดถึงหรือสนใจ เช่น Honda, Yamaha หรือ unknown",
    )
    mentioned_model: str = Field(
        default="unknown",
        description="ชื่อรุ่นรถที่ผู้ใช้พูดถึงและอยู่ใน catalog เช่น Click 160, PCX 160, N-MAX หรือ unknown",
    )
    cc: str = Field(
        default="unknown",
        description="ขนาดเครื่องยนต์ที่ต้องการ เช่น 110, 125, 150, 160 หรือ unknown",
    )
    price_est_thb: str = Field(
        default="unknown",
        description="ราคาประมาณเป็นเงินบาท เช่น 50000, 90000 หรือ unknown",
    )
    type: Literal[
        "scooter",
        "automatic",
        "manual",
        "sport",
        "bigbike",
        "ev",
        "unknown",
    ] = Field(
        default="unknown",
        description="ประเภทรถจักรยานยนต์",
    )
    budget_level: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับงบประมาณ",
    )
    usage_fit: Literal[
        "city",
        "long_distance",
        "daily",
        "delivery",
        "unknown",
    ] = Field(
        default="unknown",
        description="ลักษณะการใช้งานที่เหมาะสม",
    )
    style: Literal["beauty", "sporty", "premium", "unknown"] = Field(
        default="unknown",
        description="สไตล์รถที่ชอบ",
    )
    performance: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับความแรงหรือ performance ที่ต้องการ",
    )
    comfort: Literal["low", "medium", "high", "unknown"] = Field(
        default="unknown",
        description="ระดับความสบายที่ต้องการ",
    )
    easy_to_ride: BoolUnknown = Field(
        default="unknown",
        description="ต้องการรถขับง่ายหรือไม่",
    )
    fuel_saving: BoolUnknown = Field(
        default="unknown",
        description="ต้องการรถประหยัดน้ำมันหรือไม่",
    )
    storage_need: BoolUnknown = Field(
        default="unknown",
        description="ต้องการพื้นที่เก็บของหรือไม่",
    )


class ExtractPreferenceResult(BaseModel):
    model_name: str
    schema_type: Literal["user_based", "item_based"]
    raw_message: str
    preference: dict
