import bcrypt

from app.repositories.auth_repository import find_auth_by_username
from app.repositories.user_repository import find_user_by_id
from app.repositories.auth_transaction_repository import create_user_with_auth
from app.services.jwt_service import JWT_EXPIRES_IN_SECONDS, create_access_token


def hash_password(password: str) -> str:
    password_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(password_bytes, bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    password_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(password_bytes, hashed_password.encode("utf-8"))


def register_user(username: str, password: str, name: str, age: int = None, gender: int = None):
    password_hash = hash_password(password)

    return create_user_with_auth(
        username=username,
        password_hash=password_hash,
        name=name,
        age=age,
        gender=gender,
    )


def login_user(username: str, password: str):
    account = find_auth_by_username(username)

    if not account:
        raise ValueError("Invalid username or password")

    if not verify_password(password, account["password"]):
        raise ValueError("Invalid username or password")

    user = find_user_by_id(account["user_id"])

    if not user:
        raise ValueError("User profile not found")

    access_token, expires_at = create_access_token(user["id"])

    return {
        "message": "Login successful",
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRES_IN_SECONDS,
        "expires_at": expires_at.isoformat(),
        "user_id": user["id"],
        "auth_account_id": account["auth_account_id"],
        "username": account["username"],
        "name": user["name"],
        "age": user["age"],
        "gender": user["gender"],
    }
