from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings

connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)

# pool_pre_ping: test each pooled connection before using it and silently reconnect
# if it has gone stale. Hosted Postgres pauses after a few
# idle minutes and drops open connections; without this the first request after
# a pause would fail with an "SSL connection has been closed" error.
# pool_recycle: also retire connections older than 5 minutes.
engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
    pool_recycle=300,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Yields a DB session for the duration of a single request, then closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()