"""
Chatbot routes.

  GET  /chat                    - the chat page. Optional ?prediction_id=N opens
                                  a conversation about one specific result;
                                  without it you get the general conversation.
  POST /api/chat                - JSON in, JSON out. Sends one message to Gemini
                                  and returns the assistant's reply.
  POST /api/chat/clear          - deletes the messages of one conversation.

Conversations ("threads") are not a separate table. A thread is simply all of a
user's ChatMessage rows with the same prediction_id (NULL = the general thread),
so no migration is needed.

Everything requires a logged-in user, and a user can only ever read or discuss
their own predictions.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.auth.utils import get_current_user_optional
from app.chatbot.context_builder import build_system_prompt, describe_prediction
from app.chatbot.gemini_client import ChatbotError, generate_reply
from app.chatbot.models import ChatMessage
from app.database import get_db
from app.history.models import Prediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chatbot"])
templates = Jinja2Templates(directory="app/templates")

MAX_MESSAGE_CHARS = 1000        # per user message (also enforced in the textarea)
HISTORY_MESSAGES_SENT = 12      # how many past messages go to Gemini as memory
HISTORY_MESSAGES_SHOWN = 100    # how many past messages the page loads
RECENT_PREDICTIONS_LIMIT = 15   # entries in the "Discussing" dropdown
RATE_LIMIT_MESSAGES = 10        # max messages per user ...
RATE_LIMIT_WINDOW_SECONDS = 60  # ... per this many seconds


# user_id -> timestamps of their recent chat requests. In-memory on purpose: it's
# only a guard against runaway clicking / burning the Gemini quota, so it doesn't
# need to survive restarts (and each server worker keeps its own count).
_recent_requests: dict[int, deque] = defaultdict(deque)
_rate_lock = threading.Lock()


def _rate_limited(user_id: int) -> bool:
    now = time.monotonic()
    with _rate_lock:
        stamps = _recent_requests[user_id]
        while stamps and now - stamps[0] > RATE_LIMIT_WINDOW_SECONDS:
            stamps.popleft()
        if len(stamps) >= RATE_LIMIT_MESSAGES:
            return True
        stamps.append(now)
        return False


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    prediction_id: Optional[int] = None


class ClearRequest(BaseModel):
    prediction_id: Optional[int] = None


# --- helpers ---------------------------------------------------------------

def _json_error(message: str, status_code: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status_code)


def _get_own_prediction(db: Session, user_id: int, prediction_id: Optional[int]) -> Optional[Prediction]:
    if prediction_id is None:
        return None
    return (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user_id)
        .first()
    )


def _thread_query(db: Session, user_id: int, prediction_id: Optional[int]):
    query = db.query(ChatMessage).filter(ChatMessage.user_id == user_id)
    if prediction_id is None:
        return query.filter(ChatMessage.prediction_id.is_(None))
    return query.filter(ChatMessage.prediction_id == prediction_id)


def _recent_predictions(db: Session, user_id: int, limit: int) -> list[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(desc(Prediction.created_at), desc(Prediction.id))
        .limit(limit)
        .all()
    )


# --- chat page -------------------------------------------------------------

@router.get("/chat", response_class=HTMLResponse)
def chat_page(
    request: Request,
    prediction_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return RedirectResponse(url="/login")

    selected = _get_own_prediction(db, current_user.id, prediction_id)
    if prediction_id is not None and selected is None:
        # unknown id, or somebody else's result: fall back to the general chat
        return RedirectResponse(url="/chat")

    recent = _recent_predictions(db, current_user.id, RECENT_PREDICTIONS_LIMIT)
    # make sure the selected result is in the dropdown even if it's an older one
    options = [describe_prediction(p) for p in recent]
    if selected and all(o["id"] != selected.id for o in options):
        options.append(describe_prediction(selected))

    messages = (
        _thread_query(db, current_user.id, selected.id if selected else None)
        .order_by(desc(ChatMessage.id))
        .limit(HISTORY_MESSAGES_SHOWN)
        .all()
    )
    messages.reverse()  # oldest first

    return templates.TemplateResponse(
        "chat.html",
        {
            "request": request,
            "current_user": current_user,
            "page_class": "chat-theme",
            "selected": describe_prediction(selected) if selected else None,
            "prediction_options": options,
            "messages": messages,
            "max_chars": MAX_MESSAGE_CHARS,
        },
    )


# --- chat API --------------------------------------------------------------

@router.post("/api/chat")
def chat_send(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _json_error("Your session has expired. Please log in again.", 401)

    message = payload.message.strip()
    if not message:
        return _json_error("Please type a message first.", 400)
    if len(message) > MAX_MESSAGE_CHARS:
        return _json_error(f"Please keep messages under {MAX_MESSAGE_CHARS} characters.", 400)

    prediction = _get_own_prediction(db, current_user.id, payload.prediction_id)
    if payload.prediction_id is not None and prediction is None:
        return _json_error("That screening result wasn't found.", 404)

    if _rate_limited(current_user.id):
        return _json_error("You're sending messages quickly. Please wait a moment and try again.", 429)

    # Conversation memory: the last few messages of this thread, oldest first.
    thread = _thread_query(db, current_user.id, prediction.id if prediction else None)
    past = thread.order_by(desc(ChatMessage.id)).limit(HISTORY_MESSAGES_SENT).all()
    history = [{"role": m.role, "content": m.content} for m in reversed(past)]

    system_prompt = build_system_prompt(
        prediction,
        recent=[] if prediction else _recent_predictions(db, current_user.id, 5),
    )

    try:
        reply = generate_reply(system_prompt, history, message)
    except ChatbotError as exc:
        # Nothing is saved, so a failed attempt leaves no half-conversation behind.
        return _json_error(exc.user_message, exc.status_code)

    pid = prediction.id if prediction else None
    db.add(ChatMessage(user_id=current_user.id, prediction_id=pid, role="user", content=message))
    db.add(ChatMessage(user_id=current_user.id, prediction_id=pid, role="assistant", content=reply))
    db.commit()

    return {"reply": reply}


@router.post("/api/chat/clear")
def chat_clear(
    payload: ClearRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return _json_error("Your session has expired. Please log in again.", 401)

    _thread_query(db, current_user.id, payload.prediction_id).delete(synchronize_session=False)
    db.commit()
    return {"ok": True}
