"""
Alembic's environment script.

This is what actually runs when you type `alembic upgrade head` or
`alembic revision --autogenerate`. It's wired to:
  1. Pull the real DATABASE_URL from app.config (so alembic.ini can stay blank/safe)
  2. Import Base.metadata from app.database, PLUS every model file, so
     autogenerate can "see" your tables (User, Prediction, ChatMessage, etc.)

NOTE: this file was missing from the project — without it, `alembic upgrade`
and `alembic revision --autogenerate` fail with "Can't find Python file
migrations/env.py". Your existing migrations/versions/*.py file still works
fine once this is back in place.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

from app.config import settings
from app.database import Base

# --- IMPORTANT ---
# Import every SQLAlchemy model module here so Alembic's autogenerate
# can detect their tables. Add a line for each new module as you build it.
from app.auth import models as auth_models          # noqa: F401  (User)
from app.history import models as history_models    # noqa: F401  (Prediction)
from app.chatbot import models as chatbot_models     # noqa: F401  (ChatMessage)

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
