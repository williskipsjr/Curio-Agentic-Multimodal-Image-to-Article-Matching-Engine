"""SQLAlchemy ORM models for Curio."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class Image(Base):
    """An image in the live index."""

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False, default="unsplash")
    url: Mapped[str] = mapped_column(Text, nullable=False)
    local_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Vision tags (from Gemini Flash)
    subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    attributes: Mapped[dict] = mapped_column(JSONB, default=list)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    people_present: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    orientation: Mapped[str | None] = mapped_column(String, nullable=True)
    vision_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    vision_tagged: Mapped[bool] = mapped_column(Boolean, default=False)
    vision_flagged: Mapped[bool] = mapped_column(Boolean, default=False)

    # Embeddings
    caption_embedding = mapped_column(Vector(384), nullable=True)
    clip_embedding = mapped_column(Vector(512), nullable=True)

    # Full-text search
    search_tsv = mapped_column(TSVECTOR, nullable=True)

    # Deduplication
    content_hash: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    perceptual_hash: Mapped[str | None] = mapped_column(String, nullable=True)

    # Soft delete
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    searchable_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Relationships
    suggestions: Mapped[list[Suggestion]] = relationship(back_populates="image")
    labels: Mapped[list[Label]] = relationship(back_populates="image")

    __table_args__ = (
        Index("idx_images_category", "category"),
        Index("idx_images_content_hash", "content_hash"),
        Index("idx_images_perceptual_hash", "perceptual_hash"),
        Index("idx_images_search", "search_tsv", postgresql_using="gin"),
    )


class Article(Base):
    """An article that needs an image."""

    __tablename__ = "articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Structured brief (from LLM)
    primary_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    must_have: Mapped[dict] = mapped_column(JSONB, default=list)
    must_not_have: Mapped[dict] = mapped_column(JSONB, default=list)
    mood: Mapped[str | None] = mapped_column(String, nullable=True)
    orientation: Mapped[str | None] = mapped_column(String, nullable=True)

    # Embedding
    brief_embedding = mapped_column(Vector(384), nullable=True)

    # Metadata
    is_no_match: Mapped[bool] = mapped_column(Boolean, default=False)
    split: Mapped[str | None] = mapped_column(String, nullable=True)  # 'dev' | 'test'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    suggestions: Mapped[list[Suggestion]] = relationship(back_populates="article")
    labels: Mapped[list[Label]] = relationship(back_populates="article")


class Editor(Base):
    """An editor who reviews image-article pairings."""

    __tablename__ = "editors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    publication: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    preferences: Mapped[list[EditorPreferenceRecord]] = relationship(
        back_populates="editor"
    )
    suggestions: Mapped[list[Suggestion]] = relationship(back_populates="editor")


class EditorPreferenceRecord(Base):
    """A preference fact stored for an editor."""

    __tablename__ = "editor_preferences"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    editor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("editors.id", ondelete="CASCADE"), nullable=False
    )
    preference: Mapped[str] = mapped_column(Text, nullable=False)
    polarity: Mapped[str] = mapped_column(String, nullable=False)  # positive | negative
    source: Mapped[str] = mapped_column(String, nullable=False)  # explicit | inferred
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    confirmed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    editor: Mapped[Editor] = relationship(back_populates="preferences")

    __table_args__ = (
        Index("idx_pref_editor", "editor_id"),
    )


class Suggestion(Base):
    """A proposed image-article pairing awaiting human review."""

    __tablename__ = "suggestions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    editor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("editors.id", ondelete="CASCADE"), nullable=False
    )

    # Scores
    retrieval_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    reranker_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    guard_result: Mapped[str] = mapped_column(String, nullable=False)
    guard_reasons: Mapped[dict] = mapped_column(JSONB, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Explanation
    reasons: Mapped[dict] = mapped_column(JSONB, default=list)
    alternatives: Mapped[dict] = mapped_column(JSONB, default=list)

    # Human review
    status: Mapped[str] = mapped_column(String, default="pending")
    review_comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    article: Mapped[Article] = relationship(back_populates="suggestions")
    image: Mapped[Image] = relationship(back_populates="suggestions")
    editor: Mapped[Editor] = relationship(back_populates="suggestions")


class CostLedger(Base):
    """A ledger entry for every model API call."""

    __tablename__ = "cost_ledger"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model: Mapped[str] = mapped_column(String, nullable=False)
    operation: Mapped[str] = mapped_column(String, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_cost_date", "created_at"),
    )


class Job(Base):
    """A job in the Postgres-backed work queue."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    queue: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String, unique=True, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_jobs_queue_status", "queue", "status"),
    )


class Label(Base):
    """A relevance label for an article-image pair (for evaluation)."""

    __tablename__ = "labels"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    article_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False
    )
    grade: Mapped[int] = mapped_column(Integer, nullable=False)  # 0, 1, or 2
    labeler: Mapped[str] = mapped_column(String, nullable=False)  # human | llm_judge
    split: Mapped[str] = mapped_column(String, nullable=False)  # dev | test
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    article: Mapped[Article] = relationship(back_populates="labels")
    image: Mapped[Image] = relationship(back_populates="labels")

    __table_args__ = (
        UniqueConstraint("article_id", "image_id", "labeler", name="uq_label"),
    )
