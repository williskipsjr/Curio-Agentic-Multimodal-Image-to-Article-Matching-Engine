"""Pydantic schemas for the Curio API."""

from __future__ import annotations

from typing import Literal
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
    """Mismatch guard decision for an image–article pair."""

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
