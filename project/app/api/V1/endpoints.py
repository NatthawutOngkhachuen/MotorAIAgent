from fastapi import APIRouter
from app.models.query import SearchRequest, ChatRequest
from app.services.query_service import search_by_keyword
from app.services.chat_service import answer_question

router = APIRouter()

@router.post("/search")
async def search(req: SearchRequest):
    return search_by_keyword(req.keyword)

@router.post("/chat")
async def chat(req: ChatRequest):
    return answer_question(req.question, req.language)