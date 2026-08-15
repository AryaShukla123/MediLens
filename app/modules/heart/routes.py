from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.utils import get_current_user_optional
from app.history.models import Prediction
from app.modules.heart.preprocessing import predict_heart_disease, model
from app.shared.shap_utils import generate_shap_explanation

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODULE_NAME = "heart"

# Human-readable labels for the SHAP chart and the "your submitted
# values" summary on the result page.
HEART_FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate Achieved",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels (Fluoroscopy)",
    "thal": "Thalassemia",
}


@router.get("/predict/heart", response_class=HTMLResponse)
def get_heart_form(
    request: Request, current_user=Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        "predict_heart.html",
        {"request": request, "current_user": current_user},
    )


@router.post("/predict/heart", response_class=HTMLResponse)
def post_heart_predict(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...),
):
    if not current_user:
        return RedirectResponse(url="/login")

    raw_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    prediction_output = predict_heart_disease(raw_data)

    is_positive = prediction_output["prediction"] == 1
    result_label = "High Risk of Heart Disease" if is_positive else "Low Risk of Heart Disease"

    shap_data = generate_shap_explanation(
        model=model,
        processed_features=prediction_output["processed_features"],
        raw_input=raw_data,
        feature_labels=HEART_FEATURE_LABELS,
        top_n=6,
    )

    new_prediction = Prediction(
        user_id=current_user.id,
        module=MODULE_NAME,
        input_data=raw_data,
        result=result_label,
        confidence=prediction_output["probability"],
        shap_summary=shap_data,
        gradcam_image_path=None,
    )
    db.add(new_prediction)
    db.commit()
    db.refresh(new_prediction)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "current_user": current_user,
            "module": MODULE_NAME,
            "module_display_name": "Heart Disease",
            "result": result_label,
            "is_positive": is_positive,
            "confidence": prediction_output["probability"],
            "prediction_id": new_prediction.id,
            "shap_data": shap_data,
            "raw_input": raw_data,
            "feature_labels": HEART_FEATURE_LABELS,
        },
    )
