from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6)
    name: str = Field(..., min_length=1, max_length=255)
    age: Optional[int] = None
    gender: int = Field(..., ge=1, le=2)


class RegisterResponse(BaseModel):
    user_id: UUID
    username: str
    name: str
    age: Optional[int] = None
    gender: Optional[int] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    message: str
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    expires_at: str
    user_id: UUID
    username: str
    name: str
    age: Optional[int] = None
    gender: Optional[int] = None
