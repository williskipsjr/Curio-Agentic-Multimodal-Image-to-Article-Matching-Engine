"""Pydantic schemas for the Curio API."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class VisionTags(BaseModel):
    """Structured tags produced by the vision model for a single image."""

    subject: str
    category: str
    attributes: list[str]
    caption: str
    people_present: bool
    orientation: Literal["landscape", "portrait", "square"]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Article Brief
# ---------------------------------------------------------------------------


class ArticleBrief(BaseModel):
    """Structured image requirements derived from an article."""

    primary_subject: str
    must_have: list[str]
    must_not_have: list[str]
    mood: str
    orientation: Literal["landscape", "portrait", "square", "any"]


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------


class GuardResult(BaseModel):
    """Mismatch guard decision for an image-article pair."""

    decision: Literal["ACCEPT", "REJECT", "NO_CONFIDENT_MATCH"]
    reasons: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Final response from the curator agent."""

    image_id: UUID | None = None
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    alternatives: list[UUID] = Field(default_factory=list)
    guard_result: GuardResult | None = None


# ---------------------------------------------------------------------------
# Preference
# ---------------------------------------------------------------------------


class EditorPreference(BaseModel):
    """A single preference fact for an editor."""

    preference: str
    polarity: Literal["positive", "negative"]
    source: Literal["explicit", "inferred"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Request / Response helpers
# ---------------------------------------------------------------------------


class ImageUploadRequest(BaseModel):
    """Request body for image upload."""

    url: str
    source: Literal["unsplash", "upload"] = "upload"
    source_id: str | None = None


class ArticleCreateRequest(BaseModel):
    """Request body for creating an article."""

    title: str
    body: str
    source_url: str | None = None
    source_id: str | None = None


class MatchResult(BaseModel):
    """A single image match in the ranked results."""

    image_id: UUID
    rank: int
    scores: dict[str, float]
    guard: GuardResult
    reasons: list[str] = Field(default_factory=list)


class AgentChatRequest(BaseModel):
    """Request body for multi-turn curator agent chat."""

    article_id: UUID
    editor_id: UUID
    message: str
    session_id: str | None = None


class AgentChatResponse(BaseModel):
    """Response from the curator agent chat endpoint."""

    message: str
    decision: str | None = None
    image_id: UUID | None = None
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    alternatives: list[UUID] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class SuggestionReviewRequest(BaseModel):
    """Request body for approving or rejecting a suggestion."""

    comment: str | None = None


class SuggestionResponse(BaseModel):
    """Suggestion representation."""

    id: UUID
    article_id: UUID
    image_id: UUID
    editor_id: UUID
    status: str
    guard_result: str
    guard_reasons: list[str] = Field(default_factory=list)
    confidence: float | None = None
    reasons: list[str] = Field(default_factory=list)
    review_comment: str | None = None


class CostSummaryResponse(BaseModel):
    """Cost ledger and budget summary response."""

    daily_budget_usd: float
    today_spent_usd: float
    budget_exceeded: bool
    total_spent_usd: float
    breakdown: list[dict[str, Any]] = Field(default_factory=list)

