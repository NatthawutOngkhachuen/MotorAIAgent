import base64
import hashlib
import hmac
import json
import os
from datetime import datetime, timedelta, timezone


JWT_ALGORITHM = "HS256"
JWT_EXPIRES_IN_SECONDS = 60 * 60 * 24


class TokenExpiredError(ValueError):
    pass


class InvalidTokenError(ValueError):
    pass


def _get_secret_key() -> bytes:
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY") or "change-me-in-env"
    return secret.encode("utf-8")


def _base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _base64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _json_dumps(data: dict) -> bytes:
    return json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")


def create_access_token(user_id: str) -> tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=JWT_EXPIRES_IN_SECONDS)
    header = {
        "alg": JWT_ALGORITHM,
        "typ": "JWT",
    }
    payload = {
        "sub": str(user_id),
        "user_id": str(user_id),
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "exp": int(expires_at.timestamp()),
    }

    signing_input = ".".join(
        [
            _base64url_encode(_json_dumps(header)),
            _base64url_encode(_json_dumps(payload)),
        ]
    )
    signature = hmac.new(
        _get_secret_key(),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    token = f"{signing_input}.{_base64url_encode(signature)}"

    return token, expires_at


def decode_access_token(token: str) -> dict:
    try:
        header_segment, payload_segment, signature_segment = token.split(".")
    except ValueError as exc:
        raise InvalidTokenError("Invalid token format") from exc

    signing_input = f"{header_segment}.{payload_segment}"
    expected_signature = hmac.new(
        _get_secret_key(),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()

    try:
        actual_signature = _base64url_decode(signature_segment)
    except Exception as exc:
        raise InvalidTokenError("Invalid token signature") from exc

    if not hmac.compare_digest(expected_signature, actual_signature):
        raise InvalidTokenError("Invalid token signature")

    try:
        header = json.loads(_base64url_decode(header_segment))
        payload = json.loads(_base64url_decode(payload_segment))
    except Exception as exc:
        raise InvalidTokenError("Invalid token payload") from exc

    if header.get("alg") != JWT_ALGORITHM:
        raise InvalidTokenError("Invalid token algorithm")

    expires_at = payload.get("exp")
    if not isinstance(expires_at, int):
        raise InvalidTokenError("Token missing expiration")

    if datetime.now(timezone.utc).timestamp() >= expires_at:
        raise TokenExpiredError("Token expired")

    user_id = payload.get("user_id") or payload.get("sub")
    if not user_id:
        raise InvalidTokenError("Token missing user id")

    return payload
