from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    module = Column(String, nullable=False, index=True)

    input_data = Column(JSON, nullable=False)

    result = Column(String, nullable=False)          
    confidence = Column(Float, nullable=True)         # model's confidence score, 0-1

    # Top SHAP feature contributions behind the result
    shap_summary = Column(JSON, nullable=True)

    # Unused since the retinal-imaging module was removed. The column is kept (always
    # null) so no database migration is needed just to drop it.
    gradcam_image_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # --- Relationships ---
    user = relationship("User", back_populates="predictions")
    chat_messages = relationship(
        "ChatMessage", back_populates="prediction", cascade="all, delete-orphan"
    )