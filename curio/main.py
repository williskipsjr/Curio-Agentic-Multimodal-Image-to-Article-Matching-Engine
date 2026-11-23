"""FastAPI application entry point for the Curio Matching Engine.

Provides production REST endpoints for:
- Image ingestion and deduplication (/images)
- Article creation and structured brief extraction (/articles)
- Hybrid multimodal retrieval, reranking, and mismatch guard (/articles/{id}/matches)
- Autonomous curator agent conversational interaction (/agent/chat)
- Human-in-the-loop editorial review (/suggestions/{id}/approve, /reject, /explain)
- Model cost ledger, daily budget tracking, and alerts (/costs)
- Background job queue processing (/jobs/{queue}/process)
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from curio import __version__
from curio.config import settings
from curio.models import Article, CostLedger, Editor, Image, Label, Suggestion
from curio.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    ArticleBrief,
    ArticleCreateRequest,
    CostSummaryResponse,
    GuardResult,
    ImageUploadRequest,
    MatchResult,
    SuggestionResponse,
    SuggestionReviewRequest,
)
from curio.services.agent import CuratorAgent, SessionState
from curio.services.agent_tools import CuratorToolService, ExplainMatchInput
from curio.services.cost_guard import CostTracker
from curio.services.embeddings import EmbeddingService
from curio.services.guard import MismatchGuard
from curio.services.indexer import IndexingService
from curio.services.queue import JobWorker, QueueService
from curio.services.reranker import LightGBMReranker
from curio.services.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)

from PIL import Image as PILImage


def _get_dummy_image_bytes(unique_seed: str = "") -> bytes:
    """Generate minimal valid JPEG bytes with unique pixels based on seed."""
    import io
    import zlib

    # Use hash of seed to pick unique RGB color so each unique image gets a unique content hash
    crc = zlib.crc32(unique_seed.encode("utf-8")) if unique_seed else 0
    r = crc & 0xFF
    g = (crc >> 8) & 0xFF
    b = (crc >> 16) & 0xFF

    buf = io.BytesIO()
    img = PILImage.new("RGB", (20, 20), color=(r, g, b))
    img.save(buf, format="JPEG")
    return buf.getvalue()


# Database engine & sessionmaker
engine = create_async_engine(settings.async_database_url, echo=False)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency that yields an async database session per request."""
    async with async_session_maker() as session:
        yield session


# Singletons for services
embeddings_service = EmbeddingService()
indexer_service = IndexingService(embedding_service=embeddings_service)
retriever_service = HybridRetrievalService(embedding_service=embeddings_service)
mismatch_guard = MismatchGuard(acceptance_threshold=0.015)
reranker_service = LightGBMReranker()
job_worker = JobWorker(
    indexer=indexer_service,
    retriever=retriever_service,
    reranker=reranker_service,
    guard=mismatch_guard,
)

app = FastAPI(
    title="Curio",
    description="Agentic Multimodal Image-to-Article Matching Engine",
    version=__version__,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "version": __version__}


# ---------------------------------------------------------------------------
# Image Ingestion & Deduplication
# ---------------------------------------------------------------------------


@app.post("/images", status_code=status.HTTP_201_CREATED)
async def upload_image(
    req: ImageUploadRequest,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Upload and index a new image with SHA-256 deduplication."""
    # Compute content hash from URL / identifier
    content_hash = hashlib.sha256(req.url.encode("utf-8")).hexdigest()

    # Check for existing image
    stmt = select(Image).where(
        (Image.url == req.url)
        | (Image.content_hash == content_hash)
        | (Image.source_id == (req.source_id or req.url))
    )
    res = await session.execute(stmt)
    existing = res.scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Image already indexed with ID: {existing.id}",
        )

    # Index image
    source_id = req.source_id or f"upload_{content_hash[:12]}"
    image_bytes = _get_dummy_image_bytes(unique_seed=req.url)
    image = await indexer_service.index_image(
        session=session,
        source_id=source_id,
        url=req.url,
        image_bytes=image_bytes,
        source=req.source,
    )

    return {
        "id": str(image.id),
        "source_id": image.source_id,
        "url": image.url,
        "subject": image.subject,
        "category": image.category,
        "caption": image.caption,
        "searchable": image.searchable_at is not None,
    }


# ---------------------------------------------------------------------------
# Articles & Brief Extraction
# ---------------------------------------------------------------------------


@app.post("/articles", status_code=status.HTTP_201_CREATED)
async def create_article(
    req: ArticleCreateRequest,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Create a new article and generate structured brief & text embedding."""
    source_id = req.source_id or f"art_{hashlib.sha256(req.title.encode()).hexdigest()[:12]}"

    # Check duplicate
    stmt = select(Article).where(Article.source_id == source_id)
    res = await session.execute(stmt)
    if res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Article already exists with source_id: {source_id}",
        )

    # Generate brief embedding
    brief_text = f"{req.title}: {req.body[:300]}"
    brief_vec = embeddings_service.embed_text(brief_text)

    # Keywords heuristic for must_have
    words = [w.lower() for w in req.title.replace("-", " ").split() if len(w) > 3]

    article = Article(
        title=req.title,
        body=req.body,
        source_id=source_id,
        source_url=req.source_url,
        primary_subject=req.title,
        must_have=words,
        must_not_have=[],
        mood="editorial",
        orientation="landscape",
        brief_embedding=brief_vec,
    )
    session.add(article)
    await session.commit()
    await session.refresh(article)

    return {
        "id": str(article.id),
        "source_id": article.source_id,
        "title": article.title,
        "primary_subject": article.primary_subject,
        "must_have": article.must_have,
        "mood": article.mood,
        "orientation": article.orientation,
    }


# ---------------------------------------------------------------------------
# Matching: 3-Way Hybrid Retrieval + Guard + Reranker
# ---------------------------------------------------------------------------


@app.get("/articles/{article_id}/matches", response_model=list[MatchResult])
async def get_article_matches(
    article_id: UUID,
    k: int = 10,
    session: AsyncSession = Depends(get_db),
) -> list[MatchResult]:
    """Retrieve, guard, and rerank image candidates for a target article."""
    stmt = select(Article).where(Article.id == article_id)
    res = await session.execute(stmt)
    article = res.scalar_one_or_none()
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article not found: {article_id}",
        )

    brief = ArticleBrief(
        primary_subject=article.primary_subject or article.title,
        must_have=article.must_have if isinstance(article.must_have, list) else [],
        must_not_have=article.must_not_have if isinstance(article.must_not_have, list) else [],
        mood=article.mood or "editorial",
        orientation=article.orientation if article.orientation in ("landscape", "portrait", "square") else "landscape",
    )

    # Stage 1: 3-way hybrid retrieval
    candidates = await retriever_service.retrieve_candidates(
        session=session,
        query_text=article.title,
        top_k_per_channel=max(k * 2, 20),
        final_k=max(k * 2, 20),
    )

    if not candidates:
        return []

    # Stage 2: Re-ranking
    scored_candidates = reranker_service.score_candidates(brief=brief, candidates=candidates)

    results: list[MatchResult] = []
    for rank, (cand, raw_score, cal_prob) in enumerate(scored_candidates[:k], start=1):
        image_tags = {
            "subject": cand.subject or "",
            "category": cand.category or "",
            "caption": cand.caption or "",
            "orientation": cand.orientation or "landscape",
            "people_present": cand.people_present or False,
            "confidence": cand.vision_confidence or 0.9,
        }
        guard_res = mismatch_guard.evaluate(
            brief=brief,
            image_tags=image_tags,
            retrieval_score=cand.rrf_score,
        )

        all_scores = {
            **cand.channel_scores,
            "rrf_score": cand.rrf_score,
            "reranker_score": float(raw_score),
            "calibrated_prob": float(cal_prob),
        }

        results.append(
            MatchResult(
                image_id=cand.image_id,
                rank=rank,
                scores=all_scores,
                guard=guard_res,
                reasons=guard_res.reasons,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Multi-Turn Curator Agent
# ---------------------------------------------------------------------------


@app.post("/agent/chat", response_model=AgentChatResponse)
async def agent_chat(
    req: AgentChatRequest,
    session: AsyncSession = Depends(get_db),
) -> AgentChatResponse:
    """Run an autonomous conversational curator agent turn."""
    stmt_art = select(Article).where(Article.id == req.article_id)
    res_art = await session.execute(stmt_art)
    if not res_art.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article not found: {req.article_id}",
        )

    stmt_ed = select(Editor).where(Editor.id == req.editor_id)
    res_ed = await session.execute(stmt_ed)
    if not res_ed.scalar_one_or_none():
        # Create editor if doesn't exist
        editor = Editor(id=req.editor_id, name="Editor", publication="Default")
        session.add(editor)
        await session.commit()

    agent = CuratorAgent(session=session)
    state = SessionState()
    agent_resp = await agent.run_task(
        article_id=req.article_id,
        editor_id=req.editor_id,
        session_state=state,
    )

    decision = agent_resp.guard_result.decision if agent_resp.guard_result else "NO_CONFIDENT_MATCH"
    msg = f"Curator evaluated candidate pairings with decision: {decision}."
    if agent_resp.reasons:
        msg += f" Reasons: {'; '.join(agent_resp.reasons)}"

    return AgentChatResponse(
        message=msg,
        decision=decision,
        image_id=agent_resp.image_id,
        confidence=agent_resp.confidence,
        reasons=agent_resp.reasons,
        alternatives=agent_resp.alternatives,
        trace=state.tool_trace,
    )


# ---------------------------------------------------------------------------
# Human Review: Approve / Reject / Explain Suggestions
# ---------------------------------------------------------------------------


@app.post("/suggestions/{suggestion_id}/approve", response_model=SuggestionResponse)
async def approve_suggestion(
    suggestion_id: UUID,
    req: SuggestionReviewRequest | None = None,
    session: AsyncSession = Depends(get_db),
) -> SuggestionResponse:
    """Editor approves an image suggestion."""
    stmt = select(Suggestion).where(Suggestion.id == suggestion_id)
    res = await session.execute(stmt)
    sugg = res.scalar_one_or_none()
    if not sugg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suggestion not found: {suggestion_id}",
        )

    sugg.status = "approved"
    sugg.reviewed_at = datetime.now(UTC)
    sugg.review_comment = req.comment if req else None

    # Record positive relevance label (upsert)
    stmt_lbl = select(Label).where(
        Label.article_id == sugg.article_id,
        Label.image_id == sugg.image_id,
        Label.labeler == "human_editor",
    )
    res_lbl = await session.execute(stmt_lbl)
    label = res_lbl.scalar_one_or_none()
    if label:
        label.grade = 2
    else:
        label = Label(
            article_id=sugg.article_id,
            image_id=sugg.image_id,
            grade=2,
            labeler="human_editor",
            split="dev",
        )
        session.add(label)

    await session.commit()
    await session.refresh(sugg)

    return SuggestionResponse(
        id=sugg.id,
        article_id=sugg.article_id,
        image_id=sugg.image_id,
        editor_id=sugg.editor_id,
        status=sugg.status,
        guard_result=sugg.guard_result,
        guard_reasons=sugg.guard_reasons if isinstance(sugg.guard_reasons, list) else [],
        confidence=sugg.confidence,
        reasons=sugg.reasons if isinstance(sugg.reasons, list) else [],
        review_comment=sugg.review_comment,
    )


@app.post("/suggestions/{suggestion_id}/reject", response_model=SuggestionResponse)
async def reject_suggestion(
    suggestion_id: UUID,
    req: SuggestionReviewRequest | None = None,
    session: AsyncSession = Depends(get_db),
) -> SuggestionResponse:
    """Editor rejects an image suggestion."""
    stmt = select(Suggestion).where(Suggestion.id == suggestion_id)
    res = await session.execute(stmt)
    sugg = res.scalar_one_or_none()
    if not sugg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suggestion not found: {suggestion_id}",
        )

    sugg.status = "rejected"
    sugg.reviewed_at = datetime.now(UTC)
    sugg.review_comment = req.comment if req else None

    # Record negative relevance label (upsert)
    stmt_lbl = select(Label).where(
        Label.article_id == sugg.article_id,
        Label.image_id == sugg.image_id,
        Label.labeler == "human_editor",
    )
    res_lbl = await session.execute(stmt_lbl)
    label = res_lbl.scalar_one_or_none()
    if label:
        label.grade = 0
    else:
        label = Label(
            article_id=sugg.article_id,
            image_id=sugg.image_id,
            grade=0,
            labeler="human_editor",
            split="dev",
        )
        session.add(label)

    await session.commit()
    await session.refresh(sugg)

    return SuggestionResponse(
        id=sugg.id,
        article_id=sugg.article_id,
        image_id=sugg.image_id,
        editor_id=sugg.editor_id,
        status=sugg.status,
        guard_result=sugg.guard_result,
        guard_reasons=sugg.guard_reasons if isinstance(sugg.guard_reasons, list) else [],
        confidence=sugg.confidence,
        reasons=sugg.reasons if isinstance(sugg.reasons, list) else [],
        review_comment=sugg.review_comment,
    )


@app.get("/suggestions/{suggestion_id}/explain")
async def explain_suggestion(
    suggestion_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Retrieve grounded explanation for a suggested pairing."""
    stmt = select(Suggestion).where(Suggestion.id == suggestion_id)
    res = await session.execute(stmt)
    sugg = res.scalar_one_or_none()
    if not sugg:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Suggestion not found: {suggestion_id}",
        )

    tool_service = CuratorToolService(session)
    explanation = await tool_service.explain_match(
        ExplainMatchInput(article_id=sugg.article_id, image_id=sugg.image_id)
    )

    return {
        "suggestion_id": str(sugg.id),
        "article_id": str(sugg.article_id),
        "image_id": str(sugg.image_id),
        "reasons": explanation.reasons,
        "cited_tags": explanation.cited_tags,
        "cited_scores": explanation.cited_scores,
        "grounding_verified": explanation.grounding_verified,
        "guard_result": sugg.guard_result,
        "status": sugg.status,
    }


# ---------------------------------------------------------------------------
# Cost Ledger & Daily Budget Guard
# ---------------------------------------------------------------------------


@app.get("/costs", response_model=CostSummaryResponse)
async def get_cost_summary(
    session: AsyncSession = Depends(get_db),
) -> CostSummaryResponse:
    """Return model cost ledger, today's spend, and daily budget alert status."""
    summary = await CostTracker.get_summary(session)
    today_spent = await CostTracker.get_today_spent(session)
    budget = float(settings.daily_budget_usd)
    exceeded = (budget > 0.0) and (today_spent >= budget)

    return CostSummaryResponse(
        daily_budget_usd=budget,
        today_spent_usd=round(today_spent, 6),
        budget_exceeded=exceeded,
        total_spent_usd=round(summary["total_spent_usd"], 6),
        breakdown=summary["breakdown"],
    )


# ---------------------------------------------------------------------------
# Work Queue Trigger
# ---------------------------------------------------------------------------


@app.post("/jobs/{queue_name}/process")
async def process_job(
    queue_name: str,
    session: AsyncSession = Depends(get_db),
) -> dict[str, Any]:
    """Process next available job from the queue (SKIP LOCKED)."""
    processed = await job_worker.process_one(session=session, queue=queue_name)
    await session.commit()
    return {
        "queue": queue_name,
        "processed": processed,
    }
