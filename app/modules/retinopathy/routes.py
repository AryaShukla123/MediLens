"""
Diabetic Retinopathy (imaging) module routes.

  GET  /predict/eye   - renders the image upload form
  POST /predict/eye   - validates the upload, runs the CNN, builds the
                         Grad-CAM heat map, saves a Prediction row, and
                         renders the result on the same template

Same shape as the tabular modules (login required, one Prediction row per
run), with Grad-CAM taking the place of SHAP.

Where things are stored on the Prediction row:
  - gradcam_image_path : web path of the blended overlay image. The original
                         and heatmap files sit beside it with the same name
                         stem (..._original.png / ..._heatmap.png).
  - shap_summary       : holds the explanation payload for this module too
                         (method, grade probabilities, Grad-CAM attention
                         summary). The column name is a tabular-era leftover,
                         but reusing it avoids a migration and means the
                         chatbot's context_builder reads one place for both
                         kinds of module.
The uploaded photo itself is not kept; only the cropped copy used for the
Grad-CAM display is saved.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Request, Depends, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.auth.utils import get_current_user_optional
from app.database import get_db
from app.history.models import Prediction
from app.modules.retinopathy.gradcam import generate_gradcam
from app.modules.retinopathy.preprocessing import predict_retinopathy

logger = logging.getLogger(__name__)

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODULE_NAME = "eye"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
LOW_CONFIDENCE_THRESHOLD = 0.5

# Plain-language text per grade (International Clinical DR severity scale,
# which is what the APTOS labels follow). Deliberately general: the page is
# a screening aid, not advice for an individual case.
GRADE_INFO = {
    0: {
        "title": "No Diabetic Retinopathy Detected",
        "description": "No visible signs of diabetic retinopathy were found in this image.",
        "next_step": "If you have diabetes, regular dilated eye exams are still recommended. "
                     "This tool can't replace one.",
    },
    1: {
        "title": "Mild Diabetic Retinopathy",
        "description": "Early changes, typically tiny swellings in the retina's smallest blood "
                       "vessels. It often causes no symptoms.",
        "next_step": "Mention this to an eye specialist at your next check-up. Keeping blood sugar "
                     "and blood pressure well controlled helps slow progression.",
    },
    2: {
        "title": "Moderate Diabetic Retinopathy",
        "description": "More widespread changes than the mild stage, such as small bleeds and "
                       "fluid leaking from blood vessels.",
        "next_step": "Book an exam with an ophthalmologist or optometrist soon rather than "
                     "waiting for your next routine check.",
    },
    3: {
        "title": "Severe Diabetic Retinopathy",
        "description": "Many of the retina's blood vessels are blocked, and the retina is starting "
                       "to signal for new vessels to grow. The risk of progression is high.",
        "next_step": "See an eye specialist promptly. This stage usually needs close monitoring "
                     "or treatment.",
    },
    4: {
        "title": "Proliferative Diabetic Retinopathy",
        "description": "An advanced stage where fragile new blood vessels grow on the retina and "
                       "can bleed, which threatens vision.",
        "next_step": "Seek an urgent appointment with an ophthalmologist. Without treatment this "
                     "stage can cause serious vision loss.",
    },
}


def _render_form(request: Request, current_user, error: Optional[str] = None, status_code: int = 200):
    return templates.TemplateResponse(
        "predict_eye.html",
        {"request": request, "current_user": current_user, "error": error},
        status_code=status_code,
    )


@router.get("/predict/eye", response_class=HTMLResponse)
def get_eye_form(request: Request, current_user=Depends(get_current_user_optional)):
    if not current_user:
        return RedirectResponse(url="/login")
    return _render_form(request, current_user)


@router.post("/predict/eye", response_class=HTMLResponse)
def post_eye_predict(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
    image: Optional[UploadFile] = File(None),
):
    if not current_user:
        return RedirectResponse(url="/login")

    # --- validate the upload ---
    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1) if image else b""
    if not image_bytes:
        return _render_form(request, current_user, "Please choose a retinal image to upload.", 400)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return _render_form(
            request, current_user,
            f"That file is too large. Please upload an image under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
            400,
        )

    # --- predict ---
    try:
        pred = predict_retinopathy(image_bytes)
    except ValueError as exc:  # not a readable image
        return _render_form(request, current_user, str(exc), 400)
    except FileNotFoundError:
        logger.exception("Retinopathy model file is missing")
        return _render_form(
            request, current_user,
            "The retinopathy model isn't available on this server yet. Please try again later.",
            503,
        )

    grade = pred["prediction"]
    grade_info = GRADE_INFO[grade]

    # --- Grad-CAM ---
    gradcam = None
    try:
        gradcam = generate_gradcam(pred["input_tensor"], pred["display_rgb"], grade)
    except Exception:
        logger.exception("Grad-CAM generation failed")

    attention = gradcam["attention"] if gradcam else None

    # --- save to history ---
    filename = os.path.basename(image.filename or "upload")[:120]
    new_prediction = Prediction(
        user_id=current_user.id,
        module=MODULE_NAME,
        input_data={"filename": filename, "size_kb": round(len(image_bytes) / 1024, 1)},
        result=grade_info["title"],
        confidence=pred["confidence"],
        shap_summary={
            "method": "grad-cam",
            "predicted_grade": grade,
            "probabilities": pred["probabilities"],
            "attention": attention,
        },
        gradcam_image_path=gradcam["urls"]["overlay"] if gradcam else None,
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    probabilities = [
        {"name": name, "percent": round(p * 100, 1), "is_predicted": i == grade}
        for i, (name, p) in enumerate(pred["probabilities"].items())
    ]

    return templates.TemplateResponse(
        "predict_eye.html",
        {
            "request": request,
            "current_user": current_user,
            "module": MODULE_NAME,
            "result": grade_info["title"],
            "is_positive": pred["is_positive"],
            "confidence": pred["confidence"],
            "low_confidence": pred["confidence"] < LOW_CONFIDENCE_THRESHOLD,
            "probabilities": probabilities,
            "grade_info": grade_info,
            "gradcam": gradcam,
            "attention": attention,
            "prediction_id": new_prediction.id,
        },
    )
