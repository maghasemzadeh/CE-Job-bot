from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, ForeignKey, Text, UniqueConstraint, DateTime, JSON
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(primary_key=True)  # Telegram user id
    username: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(default=True)

class Preference(Base):
    __tablename__ = "preferences"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True)
    text: Mapped[str] = mapped_column(Text)

class Delivery(Base):
    __tablename__ = "deliveries"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(index=True)
    channel_msg_id: Mapped[int] = mapped_column(index=True)
    __table_args__ = (UniqueConstraint("user_id", "channel_msg_id", name="uniq_delivery"),)

class ChannelPost(Base):
    __tablename__ = "channel_posts"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    channel_msg_id: Mapped[int] = mapped_column(unique=True, index=True)
    text: Mapped[str] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[datetime] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # Classification fields (individual fields for indexing and querying)
    employment_type: Mapped[str | None] = mapped_column(String(50), index=True)
    job_function: Mapped[str | None] = mapped_column(String(100), index=True)
    industry: Mapped[str | None] = mapped_column(String(50), index=True)
    seniority_level: Mapped[str | None] = mapped_column(String(50), index=True)
    work_location: Mapped[str | None] = mapped_column(String(50), index=True)
    job_specialization: Mapped[str | None] = mapped_column(String(100), index=True)
    skills_technologies: Mapped[str | None] = mapped_column(Text)
    bonuses: Mapped[bool | None] = mapped_column(Boolean)
    health_insurance: Mapped[bool | None] = mapped_column(Boolean)
    stock_options: Mapped[bool | None] = mapped_column(Boolean)
    work_schedule: Mapped[str | None] = mapped_column(String(50))
    company_size: Mapped[str | None] = mapped_column(String(50))
    company_name: Mapped[str | None] = mapped_column(String(200))
    is_classified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    
    # Complete Pydantic classification data as JSON
    classification_data: Mapped[dict | None] = mapped_column(JSON)
