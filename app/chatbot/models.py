"""
ChatMessage table — one row per message in the chatbot conversation
(both the user's question and the assistant's reply are stored as
separate rows, distinguished by `role`).
"""

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Optional link to the prediction this conversation is about.
    # Nullable because a user might ask a general question with no
    # specific prediction attached.
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    role = Column(String, nullable=False)     # "user" or "assistant"
    content = Column(Text, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # --- Relationships ---
    user = relationship("User", back_populates="chat_messages")
    prediction = relationship("Prediction", back_populates="chat_messages")