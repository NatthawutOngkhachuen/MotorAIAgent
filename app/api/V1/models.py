from pydantic import BaseModel
from typing import Literal, Optional

class SearchRequest(BaseModel):
    keyword: str

class ChatRequest(BaseModel):
    question: str
    language: Literal["th", "en"] = "th"
    session_id: Optional[str] = None
    user_id: Optional[str] = None