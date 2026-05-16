from fastapi import APIRouter, HTTPException, status

from app.schemas.auth_schema import (
    RegisterRequest,
    RegisterResponse,
    LoginRequest,
    LoginResponse
)
from app.services.auth_service import register_user, login_user


router = APIRouter(
    prefix="/auth",
    tags=["Auth"]
)


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


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest):
    try:
        return login_user(
            username=request.username,
            password=request.password,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )