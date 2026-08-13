"""
User table — one row per registered account.
Every Prediction and ChatMessage links back to a User via user_id.
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # --- Relationships ---
    
    predictions = relationship(
        "Prediction", back_populates="user", cascade="all, delete-orphan"
    )
    chat_messages = relationship(
        "ChatMessage", back_populates="user", cascade="all, delete-orphan"
    )