# MediLens

> A full-stack AI health-screening platform engineered for fast, explainable risk assessment across multiple diseases. It gives users an integrated environment to run screenings, understand *why* a model produced a result, review their history, and discuss findings with an AI assistant.

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat&logo=sqlalchemy&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat&logo=postgresql&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_API-8E75B2?style=flat&logo=googlegemini&logoColor=white)
![status](https://img.shields.io/badge/status-active-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)

MediLens is a secure, modern medical-screening web application built to estimate risk for **heart disease, diabetes, chronic kidney disease, and stroke** from clinical inputs. The architecture pairs a lightweight FastAPI backend with server-rendered Jinja2 templates and a dependency-free vanilla JavaScript frontend. Four independently trained machine-learning modules (scikit-learn and XGBoost) sit behind a shared prediction pipeline, and every result is explained with **SHAP** feature contributions. Results are stored per user, and a **Gemini-powered chatbot** reads the stored result so it can explain exactly what the user saw. Secure cookie-based JWT authentication, Alembic-managed migrations, and a one-file Render blueprint make the project ready for real deployment.

> ⚠️ **Medical disclaimer:** MediLens is an educational and portfolio project. It is a screening aid, **not** a diagnostic tool, and it does not replace professional medical advice, diagnosis, or treatment.

---

🔗 **Live Demo:** [medilens.onrender.com](https://medilens-cwnz.onrender.com)

> ⏳ Hosted on Render's free tier: the first load may take ~1 minute while the server wakes up.

### 🎥 Demo Video

https://github.com/user-attachments/assets/a51e8dd8-13ee-4c98-a9c5-46e4eecf6b22

## Example Screenshot

### Dashboard & Module Hub

![MediLens Dashboard](https://github.com/user-attachments/assets/198be147-bc67-4c4b-beb6-0de969b5cc35
)

![MediLens Heart Module](https://github.com/user-attachments/assets/13525def-a9a3-4744-a3e3-76d595edd89f
)

![MediLens AI Assistant](https://github.com/user-attachments/assets/3e1ea8fb-6794-4fd2-b24c-d7da332f5878
)

![MediLens Profile](https://github.com/user-attachments/assets/190fa08d-4ca2-4049-a389-6b624e3aed93
)

---

## Table of Contents

- [Illustrations](#illustrations)
- [Scope of Functionalities](#scope-of-functionalities)
- [Installation](#installation)
  - [Windows](#windows)
  - [OS X & Linux](#os-x--linux)
- [Development Setup](#development-setup)
- [Usage Example](#usage-example)
- [Benefits](#benefits)
- [Project Status](#project-status)
- [Release History](#release-history)
- [Sources](#sources)
- [Other Information](#other-information)

---

## Illustrations

Below is the architectural diagram illustrating the request-response workflow between the browser, the FastAPI application, the ML modules, and the database:

```text
┌──────────────────────────────────────────────────────────┐
│                  Browser (Jinja2 + Vanilla JS)           │
│        Dashboard · Predict forms · Results · Chat        │
└────────────────────────────┬─────────────────────────────┘
                             │
                             │ HTTP(S) Requests (JWT in HttpOnly cookie)
                             ▼
┌──────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│            (Uvicorn Server running on :8000)             │
├──────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ │
│  │ Auth Router│ │  Module    │ │  History   │ │Chatbot │ │
│  │ (JWT login,│ │  Routers   │ │  Router    │ │ Router │ │
│  │  profile)  │ │ (4 models) │ │ (view/del) │ │(Gemini)│ │
│  └────────────┘ └─────┬──────┘ └────────────┘ └───┬────┘ │
└───────────────────────┼─────────────────────────┬─┴──────┘
                        │                         │
            ┌───────────▼──────────┐              │
            │   ML Pipeline        │              │
            │ preprocessing → model│              │
            │  → SHAP explanation  │              │
            └───────────┬──────────┘              │
                        │                         │
                        ▼                         ▼
          ┌──────────────────────────┐   ┌─────────────────┐
          │ SQLAlchemy ORM + Alembic │   │   Gemini API    │
          │ SQLite (dev) / Postgres  │   │ (google-genai)  │
          └──────────────────────────┘   └─────────────────┘
```

---

## Scope of Functionalities

### 🛠️ Current Implemented Features

- **Secure Authentication Engine**: Registration, login, and logout with server-side validation via Pydantic (including `EmailStr`), password hashing through Passlib and Bcrypt, and PyJWT session tokens stored in an `HttpOnly`, `SameSite=Lax` cookie (`Secure` in production).
- **Four Disease-Screening Modules**: Dedicated prediction forms and models for **Heart Disease**, **Diabetes**, **Chronic Kidney Disease**, and **Stroke Risk**, each with attribute-driven client-side form validation.
- **Explainable Predictions (SHAP)**: A shared TreeExplainer helper produces the top contributing factors for every result, rendered as a horizontal bar chart so users see *why* a risk level was returned.
- **Prediction History**: Every screening is saved per user, with a history page to review past results in detail or delete them.
- **AI Health Assistant**: A floating Gemini-powered chatbot that can discuss a specific saved result (inputs, model output, and SHAP factors) or give general guidance. It is instructed to explain, not diagnose, and to escalate emergencies. Includes per-user rate limiting and a message-length cap.
- **User Profile Management**: Update display name, change password, and permanently delete the account together with all stored data.
- **Database Migrations**: Schema versioned with Alembic; runs on SQLite locally and PostgreSQL in production.
- **One-Click Deployment**: `start.sh`, `render.yaml`, and a `Dockerfile` for deploying to Render or any host that runs a Python app.

### 🧠 Machine-Learning Modules

| Module | Model | Notes |
|--------|-------|-------|
| ❤️ Heart Disease | RandomForest vs. XGBoost (best selected) | Winner chosen by recall on the positive class, ROC-AUC as tiebreaker |
| 🩸 Diabetes | XGBoost | ~89% accuracy, 0.94 ROC-AUC, 0.85 recall |
| 🫘 Chronic Kidney Disease | RandomForest | Raw dataset cleaned in `train.py` (junk header rows, target-leakage columns, Excel date corruption, pre-binned range values) |
| 🧬 Stroke Risk | RandomForest + SMOTE | Handles severe class imbalance (~4.9% positive); SMOTE applied to the training split only |

### 🚀 Upcoming Feature Roadmap

- **Downloadable Reports**: Export a screening result and its SHAP explanation as a shareable PDF.
- **Trend Tracking**: Visualize how a user's risk scores change across repeated screenings.
- **Additional Modules**: Extend the shared pipeline to further conditions.

---

## Installation

### Windows

1. Clone the project codebase repository:

   ```bash
   git clone https://github.com/AryaShukla123/MediLens.git
   cd MediLens
   ```

2. Initialize and provision the environment:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

> Python **3.11** is recommended (`numpy==1.26.4` has no Python 3.13 build).

### OS X & Linux

1. Clone the project codebase repository:

   ```bash
   git clone https://github.com/AryaShukla123/MediLens.git
   cd MediLens
   ```

2. Initialize and provision the environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Development Setup

### 1. Environment Configuration

Copy `.env.example` to `.env` in the project root and fill in your values:

```env
SECRET_KEY="your-cryptographically-secure-64-character-hex-string"
ENVIRONMENT="development"
ACCESS_TOKEN_EXPIRE_MINUTES=60
DATABASE_URL="sqlite:///./medilens.db"
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.5-flash"
```

💡 To generate a compliant security key on the fly, run:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

> For PostgreSQL, set `DATABASE_URL` to your connection string (e.g. a free Neon database). `postgres://` URLs are automatically normalised to `postgresql://`.

### 2. Train the Models

Trained model artifacts (`model.pkl`, `scaler.pkl`, `feature_order.json`) are loaded when the app starts. Datasets are not committed to the repository (`data/raw/` is git-ignored), so place each CSV in `data/raw/` (`heart.csv`, `diabetes.csv`, `kidney.csv`, `stroke.csv`) and run the training script for any module whose artifacts are missing:

```bash
python -m app.modules.heart.train
python -m app.modules.diabetes.train
python -m app.modules.kidney.train
python -m app.modules.stroke.train
```

### 3. Launching the Application

Apply database migrations, then start the server from the project root:

```bash
alembic upgrade head
uvicorn app.main:app --reload
```

On Windows PowerShell, if `alembic` cannot import the `app` package, set the path first: `$env:PYTHONPATH = (Get-Location).Path`.

Application layer runs directly at: `http://localhost:8000`

Interactive API documentation runs at: `http://localhost:8000/docs`

### 4. Deployment (Render)

Push the repository to GitHub, then in the Render dashboard choose **New → Blueprint**. `render.yaml` provisions the web service and asks for `DATABASE_URL` and `GEMINI_API_KEY`. `start.sh` applies migrations and launches Uvicorn automatically.

A `Dockerfile` is also included:

```bash
docker build -t medilens .
docker run --rm -p 8000:8000 --env-file .env medilens
```

---

## Usage Example

### Creating an Account & Running a Screening

1. Direct the browser to `http://localhost:8000/register`.
2. Fill in your name, a unique email, and a matching password in the registration form.
3. After submission, sign in at `http://localhost:8000/login`, which sets a secure session cookie and redirects you to the dashboard.
4. Choose a module card (for example **Heart Disease**) and enter the clinical values in the form.
5. Review the result page: the predicted risk level, the model's confidence, and a SHAP chart of the factors that influenced it.
6. Open the floating **AI assistant** to ask questions about your result in plain language.
7. Visit `/history` to revisit or delete past screenings, and `/profile` to manage your account.

### Application Routes

| Route | Purpose |
|-------|---------|
| `/register`, `/login`, `/logout` | Account creation and session handling |
| `/dashboard` | Module selection hub and recent activity |
| `/predict/heart` · `/predict/diabetes` · `/predict/kidney` · `/predict/stroke` | Screening forms and results |
| `/history`, `/history/{id}` | Saved results list and detail view |
| `/chat`, `/api/chat` | AI assistant page and message endpoint |
| `/profile` | Name, password, and account deletion |

---

## Benefits

- **Explainable by Design**: SHAP contributions accompany every prediction, so results are transparent instead of a black-box score.
- **Recall-First Modelling**: Model selection prioritises catching true positives, which matters most in a screening context.
- **Imbalance-Aware Training**: The stroke module uses SMOTE on the training split only, avoiding inflated accuracy on a 4.9%-positive dataset.
- **Lightweight Frontend**: Server-rendered templates with vanilla JS keep the app fast with zero frontend build step.
- **Resilient Integrations**: The Gemini client is isolated in one file and the model name is an environment variable, so a retired model needs a config change, not a code change.
- **Deployment-Ready**: Connection pre-ping and recycling keep hosted Postgres stable across idle pauses, and the app fits in Render's free 512 MB instance.

---

## Project Status

**Under Active Development**

Phase 1 (Authentication, Database & Shared Infrastructure) and Phase 2 (all four tabular ML modules with SHAP explanations) are completely finalized and stable. The AI assistant, prediction history, profile management, and deployment configuration are complete. Current work focuses on polish, documentation, and the roadmap features above.

---

## Release History

| Version | Date | Description |
|---------|------|-------------|
| v1.0.0 | Current | Core application complete: JWT auth, heart/diabetes/kidney/stroke modules with SHAP, Gemini assistant, history and profile pages, Alembic migrations, Render deployment config. |
| v1.1.0 (Beta) | Upcoming | PDF report export, risk-trend tracking, and additional screening modules. |

---

## 📊 System Information Summary

| Metric | Details |
|--------|---------|
| Built For | Advanced Full-Stack Medical AI Portfolio System Showcase |
| Frontend Stack | Jinja2 Templates, HTML, CSS, Vanilla JavaScript, Chart.js |
| Backend Stack | FastAPI, Python, SQLAlchemy, Alembic, Pydantic, Passlib, PyJWT, Uvicorn |
| ML Stack | scikit-learn, XGBoost, SHAP, imbalanced-learn, pandas, NumPy |
| AI Assistant | Google Gemini API (`google-genai` SDK) |
| Database Engine | SQLite (development) / PostgreSQL (production) |
| Deployment | Render (native Python runtime) or Docker |
| License Type | MIT Open-Source Permissive License |

---

## Sources

- [FastAPI Documentation](https://fastapi.tiangolo.com/) - High-performance core routing guidelines.
- [SHAP Documentation](https://shap.readthedocs.io/) - Model explainability and TreeExplainer reference.
- [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/) - Tabular model training and evaluation.
- [Alembic Documentation](https://alembic.sqlalchemy.org/) - Database migration tooling.
- [Google Gemini API](https://ai.google.dev/gemini-api/docs) - Conversational assistant integration.

---

## Other Information

### 👤 Author Information

- **Lead Architect**: Arya Shukla
- **Academic Specialization**: Computer Science (Artificial Intelligence & Machine Learning)
