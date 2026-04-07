from pydantic import BaseModel

class SearchRequest(BaseModel):
    keyword: str

class ChatRequest(BaseModel):
    question: str