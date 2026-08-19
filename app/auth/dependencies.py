"""FastAPI dependency that authenticates a request via a JWT bearer token."""

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.auth.tokens import InvalidTokenError, decode_access_token

_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> dict:
    """Returns {"user_id": int, "username": str}. Raises 401 on any
    missing/invalid/expired token — used as a Depends() on any protected
    route."""
    if credentials is None:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    try:
        return decode_access_token(credentials.credentials)
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")
