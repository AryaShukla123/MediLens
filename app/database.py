"""
Database connection setup.

- `engine`      : the SQLAlchemy connection to your DB (SQLite or PostgreSQL,
                   controlled entirely by DATABASE_URL in .env)
- `SessionLocal`: factory for creating DB sessions per-request
- `Base`        : declarative base class — every SQLAlchemy model
                   (User, Prediction, ChatMessage, etc.) inherits from this
- `get_db()`    : FastAPI dependency — use it in routes like:
                   def my_route(db: Session = Depends(get_db)): ...
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

# connect_args is only needed for SQLite (allows use across FastAPI's
# multiple threads). It's ignored/unnecessary for PostgreSQL.
connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

engine = create_engine(settings.database_url, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Yields a DB session for the duration of a single request, then closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()