from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pydantic import ValidationError

from app.database import get_db
from app.auth.models import User
from app.auth.schemas import UserCreate
from app.auth.utils import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user_optional,
    COOKIE_NAME,
)

router = APIRouter(tags=["auth"])
templates = Jinja2Templates(directory="app/templates")


# --- Register ---

@router.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="users/register.html",
        context={"error": None}
    )


@router.post("/register", response_class=HTMLResponse)
def register_submit(
    request: Request,
    full_name: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    # Validate input shape (email format, password length) via Pydantic schema
    try:
        UserCreate(email=email, password=password, full_name=full_name)
    except ValidationError as e:
        return templates.TemplateResponse(
            request=request,
            name="users/register.html",
            context={"error": e.errors()[0]["msg"]},
            status_code=400,
        )

    existing = db.query(User).filter(User.email == email).first()
    if existing:
        return templates.TemplateResponse(
            request=request,
            name="users/register.html",
            context={"error": "An account with that email already exists."},
            status_code=400,
        )

    new_user = User(
        email=email,
        full_name=full_name,
        hashed_password=hash_password(password),
    )
    db.add(new_user)
    db.commit()

    return RedirectResponse(url="/login", status_code=303)


# --- Login ---

@router.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="users/login.html", 
        context={"error": None}
    )


@router.post("/login", response_class=HTMLResponse)
def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(
            request=request,
            name="users/login.html",
            context={"error": "Incorrect email or password."},
            status_code=401,
        )

    # Note: Using str(user.id) so get_current_user_optional can cleanly parse integer IDs
    token = create_access_token(data={"sub": str(user.id)})

    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,       # JS can't read this cookie — reduces XSS risk
        samesite="lax",
       
    )
    return response


# --- Logout ---

@router.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(COOKIE_NAME)
    return response