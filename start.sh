#!/usr/bin/env bash
# MediLens startup script: apply database migrations, then start the web server.
#
# Used as the start command on Render (see render.yaml) and works the same on
# any host that runs a plain Python app (Railway, a VPS, ...):
#     bash start.sh
#
# Reads $PORT if the platform sets one, otherwise listens on 8000.
# Needs the usual environment variables: SECRET_KEY, DATABASE_URL, GEMINI_API_KEY.

set -euo pipefail

# Run from the folder this script lives in (the project root).
cd "$(dirname "${BASH_SOURCE[0]}")"

# The `alembic` command does not add the current folder to Python's import path,
# and migrations/env.py imports from app/, so add it ourselves.
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"

echo "==> Applying database migrations"
alembic upgrade head

echo "==> Starting MediLens on port ${PORT:-8000}"
# exec: uvicorn replaces this shell, so it receives the platform's stop signals.
# --proxy-headers: trust X-Forwarded-* headers from the platform's load balancer.
# One worker on purpose: each worker would load PyTorch and every model into memory.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
