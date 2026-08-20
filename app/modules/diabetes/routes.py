from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.utils import get_current_user_optional
from app.history.models import Prediction
from app.modules.diabetes.preprocessing import predict_diabetes, model
from app.shared.shap_utils import generate_shap_explanation

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODULE_NAME = "diabetes"

DIABETES_FEATURE_LABELS = {
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Plasma Glucose",
    "BloodPressure": "Diastolic Blood Pressure",
    "SkinThickness": "Triceps Skin Fold Thickness",
    "Insulin": "2-Hour Serum Insulin",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
}


@router.get("/predict/diabetes", response_class=HTMLResponse)
def get_diabetes_form(
    request: Request, current_user=Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        "predict_diabetes.html",
        {"request": request, "current_user": current_user},
    )


@router.post("/predict/diabetes", response_class=HTMLResponse)
def post_diabetes_predict(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
    Pregnancies: int = Form(...),
    Glucose: float = Form(...),
    BloodPressure: float = Form(...),
    SkinThickness: float = Form(...),
    Insulin: float = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...),
):
    if not current_user:
        return RedirectResponse(url="/login")

    raw_data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age,
    }

    prediction_output = predict_diabetes(raw_data)

    is_positive = prediction_output["prediction"] == 1
    result_label = "High Risk of Diabetes" if is_positive else "Low Risk of Diabetes"

    shap_data = generate_shap_explanation(
        model=model,
        processed_features=prediction_output["processed_features"],
        raw_input=raw_data,
        feature_labels=DIABETES_FEATURE_LABELS,
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
            "module_display_name": "Diabetes",
            "result": result_label,
            "is_positive": is_positive,
            "confidence": prediction_output["probability"],
            "prediction_id": new_prediction.id,
            "shap_data": shap_data,
            "raw_input": raw_data,
            "feature_labels": DIABETES_FEATURE_LABELS,
        },
    )
