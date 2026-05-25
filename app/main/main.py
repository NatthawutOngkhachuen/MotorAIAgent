from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.V1.endpoints import router
from app.api.V1.auth_route import router as auth_router
from app.api.V1.recommendation import router as recommendation_chat_router

app = FastAPI(title="Motor AI Agent")


FIELD_LABELS = {
    "username": "ชื่อผู้ใช้",
    "password": "รหัสผ่าน",
    "name": "ชื่อ",
    "age": "อายุ",
    "gender": "เพศ",
}


def _friendly_validation_message(exc: RequestValidationError) -> str:
    messages = []
    for error in exc.errors():
        field = str(error.get("loc", [""])[-1])
        label = FIELD_LABELS.get(field, field or "ข้อมูล")
        error_type = str(error.get("type", ""))
        ctx = error.get("ctx") or {}

        if error_type == "string_too_short":
            messages.append(f"กรุณากรอก{label}อย่างน้อย {ctx.get('min_length')} ตัวอักษร")
        elif error_type == "string_too_long":
            messages.append(f"{label}ยาวเกินไป กรุณากรอกไม่เกิน {ctx.get('max_length')} ตัวอักษร")
        elif error_type in {"greater_than_equal", "less_than_equal"} and field == "gender":
            messages.append("กรุณาเลือกเพศให้ถูกต้อง")
        elif error_type == "missing":
            messages.append(f"กรุณากรอก{label}")
        else:
            messages.append(f"กรุณาตรวจสอบ{label}ให้ถูกต้อง")

    return " / ".join(messages) if messages else "กรุณาตรวจสอบข้อมูลที่กรอกให้ถูกต้อง"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": _friendly_validation_message(exc)},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["chat"])
app.include_router(auth_router, prefix="/api/v1")
app.include_router(recommendation_chat_router, prefix="/api/v1")


@app.on_event("startup")
def warm_database_pool():
    try:
        from app.db.postgresql import get_connection, release_connection

        conn = get_connection()
        release_connection(conn)
    except Exception as exc:
        print(f"[WARN] PostgreSQL pool warmup failed: {exc}")


@app.get("/health")
def health():
    return {"status": "ok"}
