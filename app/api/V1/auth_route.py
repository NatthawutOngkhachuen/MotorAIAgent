import logging

from fastapi import APIRouter, HTTPException, Query, status

from app.schemas.auth_schema import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse
)
from app.services.auth_service import register_user, login_user


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)


def _mask_token(token: str) -> str:
    if len(token) <= 16:
        return token

    return f"{token[:12]}...{token[-8:]}"


def _login_with_logging(username: str, password: str):
    result = login_user(username=username, password=password)
    logger.info(
        "Login success username=%s user_id=%s token=%s expires_at=%s",
        result["username"],
        result["user_id"],
        _mask_token(result["access_token"]),
        result["expires_at"],
    )
    return result


@router.post("/register", response_model=RegisterResponse)
def register(request: RegisterRequest):
    try:
        return register_user(
            username=request.username,
            password=request.password,
            name=request.name,
            age=request.age,
            gender=request.gender,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/login", response_model=LoginResponse)
def login_get(
    username: str = Query(..., min_length=3, max_length=100),
    password: str = Query(..., min_length=1),
):
    try:
        return _login_with_logging(
            username=username,
            password=password,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
