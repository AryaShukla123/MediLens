"""
Auth helper functions:
  - password hashing/verification (passlib + bcrypt)
  - JWT creation/decoding for the session cookie
  - get_current_user_optional: a FastAPI dependency other routes use
    to check who's logged in (returns None instead of raising if
    no valid session — used for pages that behave differently for
    logged-in vs anonymous users, like the nav bar)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Request, Depends
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.auth.models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT algorithm — kept as a local constant since config.py doesn't
# define one (there's no need to make this configurable per-environment).
ALGORITHM = "HS256"
COOKIE_NAME = "access_token"


# --- Password hashing ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# --- JWT session tokens ---

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


# --- Current-user dependency ---

def get_current_user_optional(
    request: Request, db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Extracts the current user from the cookie token if present and valid.
    Returns None if there's no token or it's invalid (does not raise —
    use this for pages that should still render for anonymous visitors).
    """
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None

    user_id = payload.get("sub")
    if user_id is None:
        return None

    return db.query(User).filter(User.id == int(user_id)).first()
