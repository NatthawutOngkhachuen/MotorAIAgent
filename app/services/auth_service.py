from passlib.context import CryptContext

from app.db.auth_repository import find_auth_by_username
from app.db.user_repository import find_user_by_id
from app.db.auth_transaction_repository import create_user_with_auth

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


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

    return {
        "message": "Login successful",
        "user_id": user["id"],
        "auth_account_id": account["auth_account_id"],
        "username": account["username"],
        "name": user["name"],
        "age": user["age"],
        "gender": user["gender"],
    }