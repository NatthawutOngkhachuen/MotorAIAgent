import uuid
from sqlalchemy import Column, String, DateTime, Integer, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func

from app.db import Base


class User(Base):
    __tablename__ = "users"

    __table_args__ = (
        CheckConstraint("gender IN (1, 2)", name="chk_users_gender_male_female"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    age = Column(Integer, nullable=True)
    gender = Column(Integer, nullable=True)