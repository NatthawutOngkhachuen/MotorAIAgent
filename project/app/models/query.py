from pydantic import BaseModel
from typing import Literal

class SearchRequest(BaseModel):
    keyword: str

class ChatRequest(BaseModel):
    question: str
    language: Literal["th", "en"] = "th"