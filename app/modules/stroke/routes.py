"""
Stroke Risk module routes.

  GET  /predict/stroke   - renders the input form
  POST /predict/stroke   - validates form input, runs the model, saves a
                            Prediction row, renders the shared result page

Same pattern as heart/diabetes/kidney routes.py: login required, SHAP
explanation via the shared app/shared/shap_utils.py.

NOTE on encoding: train.py builds its feature set with
pd.get_dummies(df, columns=[...], drop_first=True), so the model expects
one-hot dummy columns, not raw category strings. The form below collects
plain category choices (gender, work type, etc.) and this route converts
each one into the matching 0/1 dummy column before scaling — the dummy
column names and the dropped reference category per field are exactly
what train.py's pd.get_dummies() produced from this dataset:
  - gender:         reference "Female" -> dummy gender_Male
  - ever_married:   reference "No"     -> dummy ever_married_Yes
  - work_type:      reference "Govt_job" -> dummies for Never_worked,
                     Private, Self-employed, children
  - Residence_type: reference "Rural"  -> dummy Residence_type_Urban
  - smoking_status: reference "Unknown" -> dummies for "formerly smoked",
                     "never smoked", "smokes"
"""

from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.utils import get_current_user_optional
from app.history.models import Prediction
from app.modules.stroke.preprocessing import predict_stroke_risk, model
from app.shared.shap_utils import generate_shap_explanation

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODULE_NAME = "stroke"

STROKE_FEATURE_LABELS = {
    "age": "Age",
    "hypertension": "Hypertension",
    "heart_disease": "Heart Disease",
    "avg_glucose_level": "Average Glucose Level",
    "bmi": "BMI",
    "gender_Male": "Gender: Male",
    "ever_married_Yes": "Ever Married",
    "work_type_Never_worked": "Work Type: Never Worked",
    "work_type_Private": "Work Type: Private",
    "work_type_Self-employed": "Work Type: Self-employed",
    "work_type_children": "Work Type: Child",
    "Residence_type_Urban": "Residence: Urban",
    "smoking_status_formerly smoked": "Smoking: Formerly Smoked",
    "smoking_status_never smoked": "Smoking: Never Smoked",
    "smoking_status_smokes": "Smoking: Currently Smokes",
}


@router.get("/predict/stroke", response_class=HTMLResponse)
def get_stroke_form(
    request: Request, current_user=Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        "predict_stroke.html",
        {"request": request, "current_user": current_user},
    )


@router.post("/predict/stroke", response_class=HTMLResponse)
def post_stroke_predict(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
    age: float = Form(...),
    hypertension: int = Form(...),
    heart_disease: int = Form(...),
    avg_glucose_level: float = Form(...),
    bmi: float = Form(...),
    gender: str = Form(...),
    ever_married: str = Form(...),
    work_type: str = Form(...),
    residence_type: str = Form(..., alias="residence_type"),
    smoking_status: str = Form(...),
):
    if not current_user:
        return RedirectResponse(url="/login")

    # Convert plain category choices into the exact one-hot dummy
    # columns the model was trained on (see module docstring above).
    raw_data = {
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "avg_glucose_level": avg_glucose_level,
        "bmi": bmi,
        "gender_Male": 1 if gender == "Male" else 0,
        "ever_married_Yes": 1 if ever_married == "Yes" else 0,
        "work_type_Never_worked": 1 if work_type == "Never_worked" else 0,
        "work_type_Private": 1 if work_type == "Private" else 0,
        "work_type_Self-employed": 1 if work_type == "Self-employed" else 0,
        "work_type_children": 1 if work_type == "children" else 0,
        "Residence_type_Urban": 1 if residence_type == "Urban" else 0,
        "smoking_status_formerly smoked": 1 if smoking_status == "formerly smoked" else 0,
        "smoking_status_never smoked": 1 if smoking_status == "never smoked" else 0,
        "smoking_status_smokes": 1 if smoking_status == "smokes" else 0,
    }

    prediction_output = predict_stroke_risk(raw_data)

    is_positive = prediction_output["prediction"] == 1
    result_label = "High Risk of Stroke" if is_positive else "Low Risk of Stroke"

    shap_data = generate_shap_explanation(
        model=model,
        processed_features=prediction_output["processed_features"],
        raw_input=raw_data,
        feature_labels=STROKE_FEATURE_LABELS,
        top_n=6,
    )

    new_prediction = Prediction(
        user_id=current_user.id,
        module=MODULE_NAME,
        input_data=raw_data,
        result=result_label,
        confidence=prediction_output["probability"],
        shap_summary=shap_data,
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
            "module_display_name": "Stroke Risk",
            "result": result_label,
            "is_positive": is_positive,
            "confidence": prediction_output["probability"],
            "prediction_id": new_prediction.id,
            "shap_data": shap_data,
            "raw_input": raw_data,
            "feature_labels": STROKE_FEATURE_LABELS,
        },
    )
