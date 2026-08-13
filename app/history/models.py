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

    # Top SHAP feature contributions (tabular modules only, null for eye module)
    shap_summary = Column(JSON, nullable=True)

    # Path to the saved Grad-CAM heatmap image (eye module only, null otherwise)
    gradcam_image_path = Column(String, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # --- Relationships ---
    user = relationship("User", back_populates="predictions")
    chat_messages = relationship(
        "ChatMessage", back_populates="prediction", cascade="all, delete-orphan"
    )