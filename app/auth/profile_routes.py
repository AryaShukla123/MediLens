"""
Profile page routes (Stage 8).

  GET  /profile            - account details, usage stats and the three forms
  POST /profile/name       - change the display name
  POST /profile/password   - change the password (needs the current one)
  POST /profile/delete     - permanently delete the account and all its data
                             (needs the password)

All routes need a logged-in user. After a successful change the browser is
redirected back to /profile with ?updated=... so a refresh doesn't resubmit
the form; a failed change re-renders the page with the error shown inside the
card it belongs to.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth.models import User
from app.auth.schemas import PasswordChange
from app.auth.utils import (
    COOKIE_NAME,
    get_current_user_optional,
    hash_password,
    verify_password,
)
from app.chatbot.context_builder import MODULE_DISPLAY_NAMES
from app.chatbot.models import ChatMessage
from app.database import get_db
from app.history.models import Prediction
from app.history.routes import MODULE_ORDER, delete_gradcam_files

logger = logging.getLogger(__name__)

router = APIRouter(tags=["profile"])
templates = Jinja2Templates(directory="app/templates")

MAX_NAME_LENGTH = 100

# Messages for the ?updated=... redirect. Only these keys are ever shown, so
# nothing from the URL is echoed into the page.
NOTICES = {
    "name": "Your name has been updated.",
    "password": "Your password has been changed.",
}


# --- helpers ---------------------------------------------------------------

def _initials(user: User) -> str:
    source = (user.full_name or "").strip() or user.email.split("@")[0]
    parts = [p for p in source.replace(".", " ").replace("_", " ").split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def _stats(db: Session, user: User) -> dict:
    counts = dict(
        db.query(Prediction.module, func.count(Prediction.id))
        .filter(Prediction.user_id == user.id)
        .group_by(Prediction.module)
        .all()
    )
    last = (
        db.query(func.max(Prediction.created_at))
        .filter(Prediction.user_id == user.id)
        .scalar()
    )
    chats = (
        db.query(func.count(ChatMessage.id))
        .filter(ChatMessage.user_id == user.id, ChatMessage.role == "user")
        .scalar()
    )
    return {
        "screenings": sum(counts.values()),
        "chats": chats or 0,
        "last_screening": last.strftime("%d %b %Y") if last else None,
        "by_module": [
            {"slug": slug, "name": MODULE_DISPLAY_NAMES[slug], "count": counts[slug]}
            for slug in MODULE_ORDER
            if counts.get(slug)
        ],
    }


def _render(
    request: Request,
    db: Session,
    user: User,
    *,
    notice: Optional[str] = None,
    error: Optional[str] = None,
    error_section: Optional[str] = None,
    name_value: Optional[str] = None,
    status_code: int = 200,
):
    return templates.TemplateResponse(
        "users/profile.html",
        {
            "request": request,
            "current_user": user,
            "initials": _initials(user),
            "member_since": user.created_at.strftime("%d %b %Y") if user.created_at else None,
            "stats": _stats(db, user),
            "notice": notice,
            "error": error,
            "error_section": error_section,
            "name_value": (user.full_name or "") if name_value is None else name_value,
            "max_name": MAX_NAME_LENGTH,
        },
        status_code=status_code,
    )


def _login_redirect() -> RedirectResponse:
    return RedirectResponse(url="/login", status_code=303)


# --- page ------------------------------------------------------------------

@router.get("/profile", response_class=HTMLResponse)
def profile_page(
    request: Request,
    updated: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _login_redirect()
    return _render(request, db, current_user, notice=NOTICES.get(updated))


# --- change name -----------------------------------------------------------

@router.post("/profile/name", response_class=HTMLResponse)
def change_name(
    request: Request,
    full_name: str = Form(""),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _login_redirect()

    name = " ".join(full_name.split())  # trims and collapses repeated spaces
    error = None
    if not name:
        error = "Please enter your name."
    elif len(name) > MAX_NAME_LENGTH:
        error = f"Your name must be {MAX_NAME_LENGTH} characters or fewer."

    if error:
        return _render(
            request, db, current_user,
            error=error, error_section="name", name_value=full_name, status_code=400,
        )

    current_user.full_name = name
    db.commit()
    return RedirectResponse(url="/profile?updated=name", status_code=303)


# --- change password -------------------------------------------------------

@router.post("/profile/password", response_class=HTMLResponse)
def change_password(
    request: Request,
    current_password: str = Form(""),
    new_password: str = Form(""),
    confirm_password: str = Form(""),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _login_redirect()

    def fail(message: str):
        return _render(
            request, db, current_user,
            error=message, error_section="password", status_code=400,
        )

    if not verify_password(current_password, current_user.hashed_password):
        return fail("Your current password is incorrect.")

    try:
        PasswordChange(
            current_password=current_password,
            new_password=new_password,
            confirm_password=confirm_password,
        )
    except ValidationError as exc:
        message = exc.errors()[0]["msg"].removeprefix("Value error, ")
        return fail(message)

    current_user.hashed_password = hash_password(new_password)
    db.commit()
    return RedirectResponse(url="/profile?updated=password", status_code=303)


# --- delete account --------------------------------------------------------

@router.post("/profile/delete", response_class=HTMLResponse)
def delete_account(
    request: Request,
    password: str = Form(""),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _login_redirect()

    if not verify_password(password, current_user.hashed_password):
        return _render(
            request, db, current_user,
            error="That password is incorrect, so nothing was deleted.",
            error_section="delete", status_code=400,
        )

    # Grad-CAM images live on disk, so remove them before the rows go.
    for prediction in db.query(Prediction).filter(
        Prediction.user_id == current_user.id, Prediction.module == "eye"
    ):
        delete_gradcam_files(prediction)

    # Predictions and chat messages are removed by the User relationship cascades.
    db.delete(current_user)
    db.commit()

    response = _login_redirect()
    response.delete_cookie(COOKIE_NAME)
    return response
