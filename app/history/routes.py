"""
History routes (Stage 7).

  GET  /history                    - the user's past screenings, newest first,
                                     with a per-module filter and pagination
  GET  /history/{id}               - one saved result, shown with the same
                                     templates as when it was first run
  POST /history/{id}/delete        - deletes a result (its chat messages go with
                                     it, via the Prediction.chat_messages cascade)

Everything requires a logged-in user and only ever touches that user's own rows.
"""

import logging
import math
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.auth.utils import get_current_user_optional
from app.chatbot.context_builder import MODULE_DISPLAY_NAMES, feature_labels_for
from app.database import get_db
from app.history.models import Prediction

logger = logging.getLogger(__name__)

router = APIRouter(tags=["history"])
templates = Jinja2Templates(directory="app/templates")

PAGE_SIZE = 10

# The modules the app currently has, in display order. Only rows from these
# modules are ever shown, so a stale row from a removed module can't break a page.
# Each module stores the probability of the HIGH-RISK class in `confidence`.
MODULE_ORDER = list(MODULE_DISPLAY_NAMES)


# --- helpers ---------------------------------------------------------------

def _is_high(p: Prediction) -> bool:
    return p.result.startswith("High")


def _row(p: Prediction) -> dict:
    percent = round(p.confidence * 100) if p.confidence is not None else None
    return {
        "id": p.id,
        "module": p.module,
        "module_name": MODULE_DISPLAY_NAMES.get(p.module, p.module.title()),
        "result": p.result,
        "is_high": _is_high(p),
        "percent": percent,
        "percent_label": "Estimated risk",
        "date": p.created_at.strftime("%d %b %Y") if p.created_at else "",
    }


def _get_own(db: Session, user_id: int, prediction_id: int) -> Optional[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user_id)
        .first()
    )


# --- list ------------------------------------------------------------------

@router.get("/history", response_class=HTMLResponse)
def history_page(
    request: Request,
    module: Optional[str] = None,
    page: int = 1,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return RedirectResponse(url="/login")

    if module not in MODULE_DISPLAY_NAMES:
        module = None

    base = db.query(Prediction).filter(
        Prediction.user_id == current_user.id, Prediction.module.in_(MODULE_ORDER)
    )

    counts = dict(
        db.query(Prediction.module, func.count(Prediction.id))
        .filter(Prediction.user_id == current_user.id, Prediction.module.in_(MODULE_ORDER))
        .group_by(Prediction.module)
        .all()
    )
    total_all = sum(counts.values())

    query = base.filter(Prediction.module == module) if module else base
    total = query.count()
    pages = max(1, math.ceil(total / PAGE_SIZE))
    page = min(max(page, 1), pages)

    items = (
        query.order_by(desc(Prediction.created_at), desc(Prediction.id))
        .offset((page - 1) * PAGE_SIZE)
        .limit(PAGE_SIZE)
        .all()
    )

    filters = [
        {"slug": slug, "name": MODULE_DISPLAY_NAMES[slug], "count": counts.get(slug, 0)}
        for slug in MODULE_ORDER
        if counts.get(slug, 0) > 0
    ]

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "current_user": current_user,
            "rows": [_row(p) for p in items],
            "filters": filters,
            "active_module": module,
            "active_module_name": MODULE_DISPLAY_NAMES.get(module) if module else None,
            "total_all": total_all,
            "total": total,
            "page": page,
            "pages": pages,
        },
    )


# --- detail ----------------------------------------------------------------

@router.get("/history/{prediction_id}", response_class=HTMLResponse)
def history_detail(
    prediction_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return RedirectResponse(url="/login")

    p = _get_own(db, current_user.id, prediction_id)
    if p is None or p.module not in MODULE_ORDER:
        return RedirectResponse(url="/history")

    saved_on = p.created_at.strftime("%d %b %Y") if p.created_at else ""

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "current_user": current_user,
            "module": p.module,
            "module_display_name": MODULE_DISPLAY_NAMES.get(p.module, p.module.title()),
            "result": p.result,
            "is_positive": _is_high(p),
            "confidence": p.confidence or 0.0,
            "prediction_id": p.id,
            "shap_data": p.shap_summary,
            "raw_input": p.input_data,
            "feature_labels": feature_labels_for(p.module),
            "saved_on": saved_on,
        },
    )


# --- delete ----------------------------------------------------------------

@router.post("/history/{prediction_id}/delete")
def history_delete(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return RedirectResponse(url="/login", status_code=303)

    p = _get_own(db, current_user.id, prediction_id)
    if p is not None:
        db.delete(p)  # cascades to this result's chat messages
        db.commit()

    return RedirectResponse(url="/history", status_code=303)
