"""
History routes (Stage 7).

  GET  /history                    - the user's past screenings, newest first,
                                     with a per-module filter and pagination
  GET  /history/{id}               - one saved result, shown with the same
                                     templates as when it was first run
  POST /history/{id}/delete        - deletes a result (its chat messages go with
                                     it, via the Prediction.chat_messages cascade,
                                     and so do its Grad-CAM image files)

Everything requires a logged-in user and only ever touches that user's own rows.
"""

import logging
import math
import os
import re
from pathlib import Path
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

# Same folder / URL prefix the retinopathy module writes its images to.
GRADCAM_DIR = Path(__file__).resolve().parents[1] / "static" / "gradcam"
GRADCAM_URL_PREFIX = "/static/gradcam"
_GRADCAM_FILE = re.compile(r"^([0-9a-f]{32})_(overlay|original|heatmap)\.png$")

# Tabular modules store the probability of the HIGH-RISK class in `confidence`;
# the eye module stores the probability of the predicted grade.
MODULE_ORDER = ["heart", "diabetes", "kidney", "stroke", "eye"]


# --- helpers ---------------------------------------------------------------

def _is_high(p: Prediction) -> bool:
    if p.module == "eye":
        grade = (p.shap_summary or {}).get("predicted_grade")
        if grade is not None:
            return grade >= 1
        return not p.result.startswith("No ")
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
        "percent_label": "Grade probability" if p.module == "eye" else "Estimated risk",
        "date": p.created_at.strftime("%d %b %Y") if p.created_at else "",
    }


def _get_own(db: Session, user_id: int, prediction_id: int) -> Optional[Prediction]:
    return (
        db.query(Prediction)
        .filter(Prediction.id == prediction_id, Prediction.user_id == user_id)
        .first()
    )


def _stored_gradcam(p: Prediction) -> Optional[dict]:
    """
    Rebuilds the {"urls": {...}} dict predict_eye.html expects from the overlay
    path stored on the row (the original and heatmap sit beside it). Returns
    None if the images are gone, e.g. the static/gradcam folder was cleared
    after a redeploy, so the page degrades instead of showing broken images.
    """
    path = p.gradcam_image_path or ""
    suffix = "_overlay.png"
    if not path.startswith(GRADCAM_URL_PREFIX + "/") or not path.endswith(suffix):
        return None
    base = path[: -len(suffix)]
    urls = {"overlay": path, "original": base + "_original.png", "heatmap": base + "_heatmap.png"}
    if not all((GRADCAM_DIR / Path(u).name).is_file() for u in urls.values()):
        return None
    return {"urls": urls}


def delete_gradcam_files(p: Prediction) -> None:
    path = p.gradcam_image_path or ""
    match = _GRADCAM_FILE.match(os.path.basename(path))
    if not match or not path.startswith(GRADCAM_URL_PREFIX + "/"):
        return
    stem = match.group(1)
    for kind in ("overlay", "original", "heatmap"):
        try:
            (GRADCAM_DIR / f"{stem}_{kind}.png").unlink(missing_ok=True)
        except OSError:
            logger.warning("Could not delete Grad-CAM file %s_%s.png", stem, kind)


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

    base = db.query(Prediction).filter(Prediction.user_id == current_user.id)

    counts = dict(
        db.query(Prediction.module, func.count(Prediction.id))
        .filter(Prediction.user_id == current_user.id)
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
    if p is None:
        return RedirectResponse(url="/history")

    saved_on = p.created_at.strftime("%d %b %Y") if p.created_at else ""

    if p.module == "eye":
        return _render_eye(request, current_user, p, saved_on)

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


def _render_eye(request: Request, current_user, p: Prediction, saved_on: str):
    # Imported here so the history page doesn't load the CNN code just to list rows.
    from app.modules.retinopathy.routes import GRADE_INFO, LOW_CONFIDENCE_THRESHOLD

    info = p.shap_summary or {}
    grade = info.get("predicted_grade")
    if grade not in GRADE_INFO:
        logger.warning("Prediction %s has no usable retinopathy grade", p.id)
        return RedirectResponse(url="/history")

    probabilities = [
        {"name": name, "percent": round(prob * 100, 1), "is_predicted": i == grade}
        for i, (name, prob) in enumerate((info.get("probabilities") or {}).items())
    ]
    confidence = p.confidence or 0.0

    return templates.TemplateResponse(
        "predict_eye.html",
        {
            "request": request,
            "current_user": current_user,
            "module": "eye",
            "result": p.result,
            "is_positive": grade >= 1,
            "confidence": confidence,
            "low_confidence": confidence < LOW_CONFIDENCE_THRESHOLD,
            "probabilities": probabilities,
            "grade_info": GRADE_INFO[grade],
            "gradcam": _stored_gradcam(p),
            "attention": info.get("attention"),
            "prediction_id": p.id,
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
        delete_gradcam_files(p)
        db.delete(p)  # cascades to this result's chat messages
        db.commit()

    return RedirectResponse(url="/history", status_code=303)
