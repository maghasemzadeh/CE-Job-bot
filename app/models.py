from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import String, Integer, Boolean, ForeignKey, Text, UniqueConstraint, DateTime, JSON
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id: Mapped[int] = mapped_column(primary_key=True)  # Telegram user id
    username: Mapped[str | None] = mapped_column(String(64))
    enabled: Mapped[bool] = mapped_column(default=True)
    # Telegram chat id (usually same as user_id for private chats, but stored explicitly)
    chat_id: Mapped[int | None] = mapped_column(Integer, index=True)
    # Telegram document file id for the latest uploaded resume (if any)
    resume_file_id: Mapped[str | None] = mapped_column(String(255))
    # Optional path on disk where the resume was saved
    resume_file_path: Mapped[str | None] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

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
    # Telegram channel chat id (needed to forward/copy original message)
    channel_chat_id: Mapped[int | None] = mapped_column(Integer, index=True)
    text: Mapped[str] = mapped_column(Text)
    caption: Mapped[str | None] = mapped_column(Text)
    posted_at: Mapped[datetime] = mapped_column(index=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    
    # Classification fields (individual fields for indexing and querying)
    employment_type: Mapped[str | None] = mapped_column(String(50), index=True)
    position: Mapped[str | None] = mapped_column(String(100), index=True)
    industry: Mapped[str | None] = mapped_column(String(50), index=True)
    seniority_level: Mapped[str | None] = mapped_column(String(50), index=True)
    work_location: Mapped[str | None] = mapped_column(String(50), index=True)
    # Minimum years of experience required, if mentioned
    years_experience: Mapped[int | None] = mapped_column(Integer, index=True)
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


class PreferredJobPosition(Base):
    __tablename__ = "preferred_job_positions"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.user_id"), index=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

    # Mirror fields from JobPositionClassification (excluding company_name)
    employment_type: Mapped[str | None] = mapped_column(String(50), index=True)
    position: Mapped[str | None] = mapped_column(String(100), index=True)
    industry: Mapped[str | None] = mapped_column(String(50), index=True)
    seniority_level: Mapped[str | None] = mapped_column(String(50), index=True)
    years_experience: Mapped[int | None] = mapped_column(Integer, index=True)
    work_location: Mapped[str | None] = mapped_column(String(50), index=True)
    skills_technologies: Mapped[str | None] = mapped_column(Text)
    bonuses: Mapped[bool | None] = mapped_column(Boolean)
    health_insurance: Mapped[bool | None] = mapped_column(Boolean)
    stock_options: Mapped[bool | None] = mapped_column(Boolean)
    work_schedule: Mapped[str | None] = mapped_column(String(50))
    company_size: Mapped[str | None] = mapped_column(String(50))
    # Accumulated free-form user text corrections/preferences (separated by '|||')
    preferred_position_text: Mapped[str | None] = mapped_column(Text)
    # Store the raw classification JSON for reference/debugging
    classification_data: Mapped[dict | None] = mapped_column(JSON)
    # Whether the user wants to receive new job notifications (active by default)
    active: Mapped[bool] = mapped_column(Boolean, default=True, index=True)
