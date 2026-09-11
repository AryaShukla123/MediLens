from fastapi import APIRouter, Request, Depends, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.auth.utils import get_current_user_optional
from app.history.models import Prediction
from app.modules.kidney.preprocessing import predict_kidney_disease, model
from app.shared.shap_utils import generate_shap_explanation

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODULE_NAME = "kidney"

KIDNEY_FEATURE_LABELS = {
    "bp (Diastolic)": "Diastolic Blood Pressure (elevated)",
    "bp limit": "Blood Pressure Category",
    "sg": "Urine Specific Gravity",
    "al": "Albumin Level",
    "rbc": "Red Blood Cells (in urine)",
    "su": "Sugar Level",
    "pc": "Pus Cells",
    "pcc": "Pus Cell Clumps",
    "ba": "Bacteria",
    "bgr": "Blood Glucose (Random)",
    "bu": "Blood Urea",
    "sod": "Sodium",
    "sc": "Serum Creatinine",
    "pot": "Potassium",
    "hemo": "Hemoglobin",
    "pcv": "Packed Cell Volume",
    "rbcc": "Red Blood Cell Count",
    "wbcc": "White Blood Cell Count",
    "htn": "Hypertension",
    "dm": "Diabetes Mellitus",
    "cad": "Coronary Artery Disease",
    "appet": "Appetite",
    "pe": "Pedal Edema",
    "ane": "Anemia",
    "grf": "Glomerular Filtration Rate (GFR)",
    "age": "Age",
}


@router.get("/predict/kidney", response_class=HTMLResponse)
def get_kidney_form(
    request: Request, current_user=Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/login")

    return templates.TemplateResponse(
        "predict_kidney.html",
        {"request": request, "current_user": current_user},
    )


@router.post("/predict/kidney", response_class=HTMLResponse)
def post_kidney_predict(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
    bp_diastolic: int = Form(..., alias="bp_diastolic"),
    bp_limit: int = Form(...),
    sg: float = Form(...),
    al: int = Form(...),
    rbc: int = Form(...),
    su: int = Form(...),
    pc: int = Form(...),
    pcc: int = Form(...),
    ba: int = Form(...),
    bgr: float = Form(...),
    bu: float = Form(...),
    sod: float = Form(...),
    sc: float = Form(...),
    pot: float = Form(...),
    hemo: float = Form(...),
    pcv: float = Form(...),
    rbcc: float = Form(...),
    wbcc: float = Form(...),
    htn: int = Form(...),
    dm: int = Form(...),
    cad: int = Form(...),
    appet: int = Form(...),
    pe: int = Form(...),
    ane: int = Form(...),
    grf: float = Form(...),
    age: float = Form(...),
):
    if not current_user:
        return RedirectResponse(url="/login")

    raw_data = {
        "bp (Diastolic)": bp_diastolic,
        "bp limit": bp_limit,
        "sg": sg,
        "al": al,
        "rbc": rbc,
        "su": su,
        "pc": pc,
        "pcc": pcc,
        "ba": ba,
        "bgr": bgr,
        "bu": bu,
        "sod": sod,
        "sc": sc,
        "pot": pot,
        "hemo": hemo,
        "pcv": pcv,
        "rbcc": rbcc,
        "wbcc": wbcc,
        "htn": htn,
        "dm": dm,
        "cad": cad,
        "appet": appet,
        "pe": pe,
        "ane": ane,
        "grf": grf,
        "age": age,
    }

    prediction_output = predict_kidney_disease(raw_data)

    is_positive = prediction_output["prediction"] == 1
    result_label = "High Risk of Chronic Kidney Disease" if is_positive else "Low Risk of Chronic Kidney Disease"

    shap_data = generate_shap_explanation(
        model=model,
        processed_features=prediction_output["processed_features"],
        raw_input=raw_data,
        feature_labels=KIDNEY_FEATURE_LABELS,
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
            "module_display_name": "Chronic Kidney Disease",
            "result": result_label,
            "is_positive": is_positive,
            "confidence": prediction_output["probability"],
            "prediction_id": new_prediction.id,
            "shap_data": shap_data,
            "raw_input": raw_data,
            "feature_labels": KIDNEY_FEATURE_LABELS,
        },
    )
