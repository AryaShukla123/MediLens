from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.routes import router as auth_router
from app.modules.heart.routes import router as heart_router
from app.modules.diabetes.routes import router as diabetes_router
from app.modules.kidney.routes import router as kidney_router
from app.modules.stroke.routes import router as stroke_router
from app.auth.utils import get_current_user_optional

from app.auth.models import User
from app.history.models import Prediction
from app.chatbot.models import ChatMessage

from app.database import engine, Base, get_db
from sqlalchemy.orm import Session
from sqlalchemy import desc

app = FastAPI(title="MediLens")

# Serve /static/* (css, js, images) from app/static/
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# --- Routers ---
app.include_router(auth_router)
app.include_router(heart_router)
app.include_router(diabetes_router)
app.include_router(kidney_router)
app.include_router(stroke_router)


# --- Root ---

@app.get("/", response_class=HTMLResponse)
def root(request: Request, current_user=Depends(get_current_user_optional)):
    if current_user:
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")


# --- Dashboard: module selection hub ---

AVAILABLE_MODULES = [
    {
        "slug": "heart",
        "name": "Heart Disease",
        "description": "Cardiovascular risk from vitals and bloodwork.",
        "icon": "heart",
    },
    {
        "slug": "diabetes",
        "name": "Diabetes",
        "description": "Type 2 diabetes risk from clinical measurements.",
        "icon": "droplet",
    },
    {
        "slug": "kidney",
        "name": "Chronic Kidney Disease",
        "description": "Renal function risk from urinalysis and blood panel.",
        "icon": "filter",
    },
    {
        "slug": "stroke",
        "name": "Stroke Risk",
        "description": "Stroke likelihood from personal and lifestyle factors.",
        "icon": "zap",
    },
]

FEATURED_MODULE = {
    "name": "Diabetic Retinopathy",
    "description": "Retinal imaging analysis using a CNN with Grad-CAM visual explanations -- our most advanced module.",
    "tag": "Deep Learning · CNN + Grad-CAM",
    "icon": "eye",
}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user_optional),
):
    if not current_user:
        return RedirectResponse(url="/login")

    recent_predictions = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user.id)
        .order_by(desc(Prediction.created_at))
        .limit(5)
        .all()
    )
    total_predictions = (
        db.query(Prediction).filter(Prediction.user_id == current_user.id).count()
    )

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "current_user": current_user,
            "page_class": "dashboard-theme",
            "modules": AVAILABLE_MODULES,
            "featured_module": FEATURED_MODULE,
            "recent_predictions": recent_predictions,
            "total_predictions": total_predictions,
        },
    )

