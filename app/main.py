from fastapi import FastAPI, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse

from app.auth.routes import router as auth_router
from app.auth.utils import get_current_user_optional

# --- IMPORTANT ---
# Every SQLAlchemy model must be imported somewhere before the app
# starts handling requests, otherwise relationships that reference a
# model by string name (e.g. User.chat_messages -> "ChatMessage")
# can't be resolved and every DB query crashes with a 500 error.
from app.auth.models import User
from app.history.models import Prediction
from app.chatbot.models import ChatMessage

from app.database import engine, Base

app = FastAPI(title="MediLens")

# Serve /static/* (css, js, images) from app/static/
app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

# --- Routers ---
app.include_router(auth_router)


# --- Root ---

@app.get("/", response_class=HTMLResponse)
def root(request: Request, current_user=Depends(get_current_user_optional)):
    if current_user:
        return RedirectResponse(url="/dashboard")
    return RedirectResponse(url="/login")


# --- TEMPORARY placeholder ---

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_placeholder(
    request: Request, current_user=Depends(get_current_user_optional)
):
    if not current_user:
        return RedirectResponse(url="/login")
    return HTMLResponse(
        f"<h1>Welcome, {current_user.full_name or current_user.email}</h1>"
        f"<p>Dashboard placeholder — real version comes in Stage 4.</p>"
        f'<p><a href="/logout">Logout</a></p>'
    )

