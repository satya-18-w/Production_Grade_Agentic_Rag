"""
JWT bearer tokens for the auth system in app/auth/service.py.

Stateless by design — no server-side session table. A token embeds the
user's id and username; app_users/app_threads (service.py) remain the only
source of truth for who exists and who owns what.
"""

from datetime import datetime, timedelta, timezone

import jwt

from app.config import settings

ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    pass


def create_access_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(minutes=settings.AUTH_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.AUTH_SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Returns {"user_id": int, "username": str}. Raises InvalidTokenError
    on any expiry/signature/format problem — callers map this to a 401."""
    try:
        payload = jwt.decode(token, settings.AUTH_SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as e:
        raise InvalidTokenError(str(e))

    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError, TypeError):
        raise InvalidTokenError("Token missing a valid subject claim.")

    return {"user_id": user_id, "username": payload.get("username")}
