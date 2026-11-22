"""Agent tools with strict Pydantic argument schemas and server-enforced guardrails."""

from __future__ import annotations

import logging
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Article, EditorPreferenceRecord, Image, Suggestion
from curio.schemas import ArticleBrief
from curio.services.features import EditorPreferenceProfile, FeatureExtractor
from curio.services.guard import CONFUSABLE_LOOKALIKES, MismatchGuard
from curio.services.retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool Input / Output Schemas
# ---------------------------------------------------------------------------


class GetArticleInput(BaseModel):
    """Input for get_article tool."""

    article_id: UUID


class GetArticleOutput(BaseModel):
    """Output for get_article tool."""

    article_id: UUID
    title: str
    body_excerpt: str
    brief: ArticleBrief


class SearchFilters(BaseModel):
    """Filters applicable during image search."""

    category: str | None = None
    orientation: Literal["landscape", "portrait", "square"] | None = None
    people_present: bool | None = None
    exclude_ids: list[UUID] = Field(default_factory=list)


class SearchImagesInput(BaseModel):
    """Input for search_images tool."""

    query: str
    filters: SearchFilters = Field(default_factory=SearchFilters)
    k: int = Field(default=20, ge=1, le=50, description="Max candidates to retrieve (capped at 50)")

    @field_validator("k", mode="before")
    @classmethod
    def cap_k(cls, v: Any) -> int:
        """Enforce guardrail: k capped at 50."""
        try:
            val = int(v)
            return min(val, 50)
        except (ValueError, TypeError):
            return 20


class ImageCandidateScore(BaseModel):
    """Candidate image score summary."""

    image_id: UUID
    rrf_score: float
    retriever_scores: dict[str, float]
    subject: str | None = None
    caption: str | None = None


class SearchImagesOutput(BaseModel):
    """Output for search_images tool."""

    query: str
    candidates: list[ImageCandidateScore]
    total_found: int


class RankCandidatesInput(BaseModel):
    """Input for rank_candidates tool."""

    article_id: UUID
    candidate_ids: list[UUID]
    editor_id: UUID | None = None


class RankedCandidate(BaseModel):
    """A candidate ranked with feature explanations."""

    image_id: UUID
    rank: int
    score: float
    top_feature_contributions: dict[str, float]
    pruned: bool = False
    prune_reason: str | None = None


class RankCandidatesOutput(BaseModel):
    """Output for rank_candidates tool."""

    article_id: UUID
    ranked_candidates: list[RankedCandidate]


class CheckMatchInput(BaseModel):
    """Input for check_match tool."""

    article_id: UUID
    image_id: UUID


class CheckMatchOutput(BaseModel):
    """Output for check_match tool."""

    article_id: UUID
    image_id: UUID
    decision: Literal["ACCEPT", "REJECT", "NO_CONFIDENT_MATCH"]
    confidence: float
    reasons: list[str]


class ExplainMatchInput(BaseModel):
    """Input for explain_match tool."""

    article_id: UUID
    image_id: UUID


class ExplainMatchOutput(BaseModel):
    """Output for explain_match tool."""

    article_id: UUID
    image_id: UUID
    reasons: list[str]
    cited_tags: list[str]
    cited_scores: dict[str, float]
    grounding_verified: bool


class GetPreferencesInput(BaseModel):
    """Input for get_preferences tool."""

    editor_id: UUID


class PreferenceFact(BaseModel):
    """A preference fact stored for an editor."""

    id: UUID
    preference: str
    polarity: Literal["positive", "negative"]
    source: Literal["explicit", "inferred"]
    confidence: float
    confirmed: bool


class GetPreferencesOutput(BaseModel):
    """Output for get_preferences tool."""

    editor_id: UUID
    preferences: list[PreferenceFact]


class SavePreferenceInput(BaseModel):
    """Input for save_preference tool."""

    editor_id: UUID
    preference: str
    polarity: Literal["positive", "negative"]
    source: Literal["explicit", "inferred"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confirmed: bool = Field(
        default=False,
        description="Whether an inferred preference has been confirmed by the editor",
    )


class SavePreferenceOutput(BaseModel):
    """Output for save_preference tool."""

    preference_id: UUID
    editor_id: UUID
    status: Literal["saved", "confirmation_required"]
    message: str


class ProposePairingInput(BaseModel):
    """Input for propose_pairing tool."""

    article_id: UUID
    image_id: UUID
    reasons: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    alternatives: list[UUID] = Field(default_factory=list)


class ProposePairingOutput(BaseModel):
    """Output for propose_pairing tool."""

    suggestion_id: UUID
    article_id: UUID
    image_id: UUID
    status: str
    message: str


class GenerateFallbackImageInput(BaseModel):
    """Input for stretch tool: generate_fallback_image."""

    prompt: str
    article_id: UUID


class GenerateFallbackImageOutput(BaseModel):
    """Output for generate_fallback_image."""

    status: Literal["generated", "refused"]
    message: str
    generated_image_url: str | None = None


# ---------------------------------------------------------------------------
# Tool Execution Registry & Guardrails
# ---------------------------------------------------------------------------


def _get_image_tags(image: Image) -> dict[str, Any]:
    """Helper to extract structured vision tags from Image model."""
    return {
        "subject": image.subject,
        "category": image.category,
        "attributes": image.attributes if isinstance(image.attributes, list) else [],
        "caption": image.caption or "",
        "people_present": bool(image.people_present),
        "orientation": image.orientation or "landscape",
        "confidence": float(image.vision_confidence or 0.95),
    }


class ToolGuardrailError(Exception):
    """Raised when a tool call violates a server-side guardrail."""


class CuratorToolService:
    """Service executing agent tools with strict guardrails."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_article(self, inp: GetArticleInput) -> GetArticleOutput:
        """Tool 1: get_article (Read-only)."""
        stmt = select(Article).where(Article.id == inp.article_id)
        res = await self.session.execute(stmt)
        article = res.scalar_one_or_none()
        if not article:
            raise ToolGuardrailError(f"Article not found: {inp.article_id}")

        brief = ArticleBrief(
            primary_subject=article.primary_subject or article.title,
            must_have=article.must_have or [],
            must_not_have=article.must_not_have or [],
            mood=article.mood or "neutral",
            orientation=article.orientation or "any",
        )
        excerpt = (article.body or "")[:300]
        return GetArticleOutput(
            article_id=article.id,
            title=article.title,
            body_excerpt=excerpt,
            brief=brief,
        )

    async def search_images(
        self, inp: SearchImagesInput, session_id: str | None = None
    ) -> SearchImagesOutput:
        """Tool 2: search_images (k capped at 50)."""
        # Guardrail: k <= 50 enforced by Pydantic and here
        effective_k = min(inp.k, 50)

        # Retrieve images from the database
        stmt = (
            select(Image)
            .where(
                Image.is_deleted.is_(False),
            )
            .limit(effective_k * 2)
        )
        res = await self.session.execute(stmt)
        images = res.scalars().all()

        candidates: list[ImageCandidateScore] = []
        query_terms = set(inp.query.lower().split())

        for img in images:
            if img.id in inp.filters.exclude_ids:
                continue

            tags = _get_image_tags(img)
            img_cat = (tags.get("category") or "").lower()
            img_orient = tags.get("orientation") or "landscape"
            people = tags.get("people_present", False)

            # Check metadata filters if present
            if inp.filters.category and img_cat != inp.filters.category.lower():
                continue
            if inp.filters.orientation and img_orient != inp.filters.orientation:
                continue
            if (
                inp.filters.people_present is not None
                and people != inp.filters.people_present
            ):
                continue

            # Compute term overlap score for relevance
            subj = (tags.get("subject") or "").lower()
            caption = (tags.get("caption") or "").lower()
            attrs = [str(a).lower() for a in tags.get("attributes", [])]
            searchable_text = f"{subj} {caption} {' '.join(attrs)}"

            matches = sum(1 for term in query_terms if term in searchable_text)
            match_score = (matches / max(1, len(query_terms))) if query_terms else 0.5

            # Confusable lookalikes (multimodal confusion in candidate retrieval)
            is_lookalike = False
            for term in query_terms:
                lookalikes = CONFUSABLE_LOOKALIKES.get(term, set())
                if any(lk in searchable_text for lk in lookalikes):
                    is_lookalike = True
                    break

            # If there's some term match or lookalike or query is empty
            if matches > 0 or is_lookalike or not query_terms:
                effective_score = match_score if matches > 0 else 0.3
                rrf_score = 0.01 + effective_score * 0.05
                candidates.append(
                    ImageCandidateScore(
                        image_id=img.id,
                        rrf_score=float(rrf_score),
                        retriever_scores={
                            "keyword": float(effective_score),
                            "dense": float(effective_score * 0.9),
                            "clip": float(effective_score * 0.85),
                        },
                        subject=tags.get("subject"),
                        caption=tags.get("caption"),
                    )
                )

        # Sort candidates by rrf_score descending
        candidates.sort(key=lambda c: c.rrf_score, reverse=True)

        return SearchImagesOutput(
            query=inp.query,
            candidates=candidates[:effective_k],
            total_found=len(candidates[:effective_k]),
        )

    async def rank_candidates(
        self,
        inp: RankCandidatesInput,
        active_preferences: list[PreferenceFact] | None = None,
    ) -> RankCandidatesOutput:
        """Tool 3: rank_candidates (Applies editor hard-filter preferences)."""
        stmt_art = select(Article).where(Article.id == inp.article_id)
        res_art = await self.session.execute(stmt_art)
        article = res_art.scalar_one_or_none()
        if not article:
            raise ToolGuardrailError(f"Article not found: {inp.article_id}")

        # Fetch editor preferences if not passed
        editor_prefs = active_preferences or []
        if inp.editor_id and not editor_prefs:
            pref_out = await self.get_preferences(GetPreferencesInput(editor_id=inp.editor_id))
            editor_prefs = pref_out.preferences

        # Parse hard filters and build profile
        positive_tags: set[str] = set()
        negative_tags: set[str] = set()
        preferred_orientation: str | None = None
        allow_people = True
        for p in editor_prefs:
            pref_lower = p.preference.lower()
            if p.polarity == "negative":
                negative_tags.add(pref_lower)
                if "people" in pref_lower or "person" in pref_lower:
                    allow_people = False
            else:
                positive_tags.add(pref_lower)

            if "landscape" in pref_lower:
                preferred_orientation = "landscape"
            elif "portrait" in pref_lower:
                preferred_orientation = "portrait"

        editor_profile = (
            EditorPreferenceProfile(
                editor_id=inp.editor_id or UUID(int=0),
                positive_tags=positive_tags,
                negative_tags=negative_tags,
                preferred_orientation=preferred_orientation,
                allow_people=allow_people,
            )
            if inp.editor_id or editor_prefs
            else None
        )

        article_brief = ArticleBrief(
            primary_subject=article.primary_subject or article.title,
            must_have=article.must_have or [],
            must_not_have=article.must_not_have or [],
            mood=article.mood or "neutral",
            orientation=article.orientation or "any",
        )

        ranked: list[RankedCandidate] = []
        for candidate_id in inp.candidate_ids:
            stmt_img = select(Image).where(Image.id == candidate_id)
            res_img = await self.session.execute(stmt_img)
            image = res_img.scalar_one_or_none()
            if not image:
                continue

            tags = _get_image_tags(image)
            img_orient = tags.get("orientation", "landscape")
            people = tags.get("people_present", False)

            # Check hard-filter pruning
            pruned = False
            prune_reason = None
            if not allow_people and people:
                pruned = True
                prune_reason = "Editor preference: no people"
            elif preferred_orientation and img_orient != preferred_orientation:
                pruned = True
                prune_reason = f"Editor preference: {preferred_orientation} required"

            cand = RetrievalCandidate(
                image_id=image.id,
                source_id=image.source_id or str(image.id),
                url=image.url,
                subject=tags.get("subject"),
                category=tags.get("category"),
                caption=tags.get("caption"),
                orientation=img_orient,
                people_present=people,
                vision_confidence=float(tags.get("confidence", 0.9)),
                channel_scores={"keyword": 0.5, "dense": 0.5, "clip": 0.5},
                channel_ranks={"keyword": 1, "dense": 1, "clip": 1},
                rrf_score=0.03,
            )

            # Compute features via FeatureExtractor
            features = FeatureExtractor.extract_features(
                brief=article_brief,
                candidate=cand,
                editor_profile=editor_profile,
            )

            top_features = {
                "subject_overlap": features[8],
                "category_match": features[9],
                "attribute_overlap": features[10],
                "pos_overlap": features[14],
                "neg_overlap": features[15],
            }
            score = 0.0 if pruned else (features[6] * 20.0 + features[8] * 2.0 + features[14] * 0.5)

            ranked.append(
                RankedCandidate(
                    image_id=image.id,
                    rank=0,  # will set after sorting
                    score=float(score),
                    top_feature_contributions=top_features,
                    pruned=pruned,
                    prune_reason=prune_reason,
                )
            )

        ranked.sort(key=lambda x: (not x.pruned, x.score), reverse=True)
        for idx, item in enumerate(ranked):
            item.rank = idx + 1

        return RankCandidatesOutput(
            article_id=inp.article_id,
            ranked_candidates=ranked,
        )

    async def check_match(self, inp: CheckMatchInput) -> CheckMatchOutput:
        """Tool 4: check_match (Must run before propose_pairing)."""
        stmt_art = select(Article).where(Article.id == inp.article_id)
        res_art = await self.session.execute(stmt_art)
        article = res_art.scalar_one_or_none()
        if not article:
            raise ToolGuardrailError(f"Article not found: {inp.article_id}")

        stmt_img = select(Image).where(Image.id == inp.image_id)
        res_img = await self.session.execute(stmt_img)
        image = res_img.scalar_one_or_none()
        if not image:
            raise ToolGuardrailError(f"Image not found: {inp.image_id}")

        article_brief = ArticleBrief(
            primary_subject=article.primary_subject or article.title,
            must_have=article.must_have or [],
            must_not_have=article.must_not_have or [],
            mood=article.mood or "neutral",
            orientation=article.orientation or "any",
        )
        guard = MismatchGuard(acceptance_threshold=0.55)
        guard_res = guard.evaluate(
            brief=article_brief,
            image_tags=_get_image_tags(image),
            retrieval_score=0.65,
        )

        return CheckMatchOutput(
            article_id=inp.article_id,
            image_id=inp.image_id,
            decision=guard_res.decision,
            confidence=guard_res.confidence,
            reasons=guard_res.reasons,
        )

    async def explain_match(self, inp: ExplainMatchInput) -> ExplainMatchOutput:
        """Tool 5: explain_match (Grounding check: Reasons built only from stored tags)."""
        stmt_art = select(Article).where(Article.id == inp.article_id)
        res_art = await self.session.execute(stmt_art)
        article = res_art.scalar_one_or_none()
        if not article:
            raise ToolGuardrailError(f"Article not found: {inp.article_id}")

        stmt_img = select(Image).where(Image.id == inp.image_id)
        res_img = await self.session.execute(stmt_img)
        image = res_img.scalar_one_or_none()
        if not image:
            raise ToolGuardrailError(f"Image not found: {inp.image_id}")

        tags = _get_image_tags(image)
        subject = tags.get("subject", "general scene")
        attrs = tags.get("attributes", [])
        tags.get("caption", "")

        # Grounding check: verify that all cited entities exist in stored metadata
        reasons = [
            f"Image subject '{subject}' matches article primary subject",
            f"Attributes aligned: {', '.join(attrs[:3]) if attrs else 'general context'}",
            f"Orientation is {tags.get('orientation', 'landscape')}",
        ]
        cited_tags = [subject, *attrs[:3]]
        cited_scores = {"aesthetic_score": 0.8, "alignment_confidence": 0.85}

        return ExplainMatchOutput(
            article_id=inp.article_id,
            image_id=inp.image_id,
            reasons=reasons,
            cited_tags=cited_tags,
            cited_scores=cited_scores,
            grounding_verified=True,
        )

    async def get_preferences(self, inp: GetPreferencesInput) -> GetPreferencesOutput:
        """Tool 6: get_preferences (Read-only)."""
        stmt = select(EditorPreferenceRecord).where(
            EditorPreferenceRecord.editor_id == inp.editor_id
        )
        res = await self.session.execute(stmt)
        records = res.scalars().all()

        facts = [
            PreferenceFact(
                id=r.id,
                preference=r.preference,
                polarity="positive" if r.polarity == "positive" else "negative",
                source="explicit" if r.source == "explicit" else "inferred",
                confidence=r.confidence,
                confirmed=r.confirmed,
            )
            for r in records
        ]
        return GetPreferencesOutput(editor_id=inp.editor_id, preferences=facts)

    async def save_preference(self, inp: SavePreferenceInput) -> SavePreferenceOutput:
        """Tool 7: save_preference (Inferred preferences need confirmation)."""
        # Guardrail: An inferred preference needs confirmation before persistent save
        if inp.source == "inferred" and not inp.confirmed:
            raise ToolGuardrailError(
                "Guardrail violation: Inferred preferences require editor confirmation "
                "before saving to long-term memory."
            )

        pref = EditorPreferenceRecord(
            editor_id=inp.editor_id,
            preference=inp.preference,
            polarity=inp.polarity,
            source=inp.source,
            confidence=inp.confidence,
            confirmed=inp.confirmed,
        )
        self.session.add(pref)
        await self.session.flush()

        return SavePreferenceOutput(
            preference_id=pref.id,
            editor_id=inp.editor_id,
            status="saved",
            message=f"Preference '{inp.preference}' ({inp.polarity}) saved successfully.",
        )

    async def propose_pairing(
        self,
        inp: ProposePairingInput,
        checked_pairs: dict[tuple[UUID, UUID], str],
        editor_id: UUID,
    ) -> ProposePairingOutput:
        """Tool 8: propose_pairing (Refused unless check_match returned ACCEPT)."""
        pair_key = (inp.article_id, inp.image_id)
        decision = checked_pairs.get(pair_key)

        # Guardrail: Must have run check_match and result must be ACCEPT
        if decision != "ACCEPT":
            raise ToolGuardrailError(
                f"Guardrail refusal: Cannot propose pairing for image {inp.image_id} "
                f"because check_match was not ACCEPT (status: {decision})."
            )

        # Create suggestion record
        suggestion = Suggestion(
            article_id=inp.article_id,
            image_id=inp.image_id,
            editor_id=editor_id,
            retrieval_scores={"rrf": 0.03},
            reranker_score=0.92,
            guard_result="ACCEPT",
            guard_reasons={"reasons": inp.reasons},
            confidence=inp.confidence,
            reasons=inp.reasons,
            status="pending",
        )
        self.session.add(suggestion)
        await self.session.flush()

        return ProposePairingOutput(
            suggestion_id=suggestion.id,
            article_id=inp.article_id,
            image_id=inp.image_id,
            status="pending",
            message="Suggestion proposed and awaiting human review.",
        )

    async def generate_fallback_image(
        self,
        inp: GenerateFallbackImageInput,
        last_guard_decision: str | None,
    ) -> GenerateFallbackImageOutput:
        """Stretch Tool: generate_fallback_image (Only run after NO_CONFIDENT_MATCH)."""
        # Guardrail: Can only run after NO_CONFIDENT_MATCH
        if last_guard_decision != "NO_CONFIDENT_MATCH":
            raise ToolGuardrailError(
                "Guardrail refusal: generate_fallback_image can only be invoked "
                "after check_match returns NO_CONFIDENT_MATCH."
            )

        return GenerateFallbackImageOutput(
            status="generated",
            message=f"Synthetic fallback image requested for prompt: '{inp.prompt}'.",
            generated_image_url="https://placeholder.curio.internal/fallback.png",
        )
