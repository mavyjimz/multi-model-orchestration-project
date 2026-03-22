"""Authentication API router."""

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from src.auth.dependencies import get_current_user, require_scope
from src.auth.jwt_utils import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_password_hash,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


class Token(BaseModel):
    """Token response model."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token payload model."""

    username: str
    scopes: list[str] | None = None


class LoginRequest(BaseModel):
    """Login request model."""

    username: str
    password: str


# Demo user store (replace with database in production)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),  # Change in production
        "scopes": ["admin", "read", "write"],
    }
}


@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """Authenticate user and return JWT token."""
    user = USERS_DB.get(request.username)

    if not user or not verify_password(request.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "scopes": user["scopes"]},
        expires_delta=access_token_expires,
    )

    return Token(access_token=access_token)


@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info."""
    return current_user


@router.get("/protected")
async def protected_route(user: dict = Depends(require_scope("read"))):
    """Example protected route requiring 'read' scope."""
    return {"message": f"Hello, {user['sub']}! You have access."}
