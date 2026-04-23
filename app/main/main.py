from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.V1.endpoints import router

app = FastAPI(title="Motor AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["search"])

@app.get("/graph")
async def graph_page():
    return FileResponse("app/static/graph.html")

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")