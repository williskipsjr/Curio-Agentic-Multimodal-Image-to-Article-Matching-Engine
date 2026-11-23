"""PostgreSQL-backed resilient work queue with concurrency control and dead-letter queue.

Uses PostgreSQL's `SELECT ... FOR UPDATE SKIP LOCKED` to allow multiple concurrent workers
to dequeue without contention or race conditions. Supports idempotency keys, automatic
retries with max attempt thresholds, and dead-letter routing on repeated failure.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Article, Job, Suggestion
from curio.schemas import ArticleBrief
from curio.services.guard import MismatchGuard
from curio.services.indexer import IndexingService
from curio.services.reranker import LightGBMReranker
from curio.services.retrieval import HybridRetrievalService

logger = logging.getLogger(__name__)

# Type for job handler functions
JobHandler = Callable[[AsyncSession, dict[str, Any]], Coroutine[Any, Any, None]]


class QueueService:
    """Manages enqueueing, safe dequeuing (SKIP LOCKED), and state transitions for jobs."""

    @staticmethod
    async def enqueue(
        session: AsyncSession,
        queue: str,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        max_attempts: int = 3,
    ) -> Job:
        """Enqueue a new job with optional idempotency."""
        if idempotency_key:
            stmt = select(Job).where(Job.idempotency_key == idempotency_key)
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()
            if existing:
                logger.info("Job with idempotency key %s already exists (status=%s)", idempotency_key, existing.status)
                return existing

        job = Job(
            queue=queue,
            payload=payload,
            idempotency_key=idempotency_key,
            status="pending",
            attempts=0,
            max_attempts=max_attempts,
        )
        session.add(job)
        await session.flush()
        logger.info("Enqueued job %s on queue '%s'", job.id, queue)
        return job

    @staticmethod
    async def dequeue(
        session: AsyncSession,
        queue: str,
    ) -> Job | None:
        """Atomically dequeue the next pending job using SELECT ... FOR UPDATE SKIP LOCKED."""
        stmt = (
            select(Job)
            .where(Job.queue == queue, Job.status == "pending")
            .order_by(Job.created_at.asc())
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        res = await session.execute(stmt)
        job = res.scalar_one_or_none()
        if not job:
            return None

        job.status = "processing"
        job.locked_at = datetime.now(UTC)
        job.attempts += 1
        await session.flush()
        logger.info("Dequeued job %s on queue '%s' (attempt %d/%d)", job.id, queue, job.attempts, job.max_attempts)
        return job

    @staticmethod
    async def complete(session: AsyncSession, job_id: UUID) -> Job:
        """Mark a job as successfully completed."""
        stmt = select(Job).where(Job.id == job_id)
        res = await session.execute(stmt)
        job = res.scalar_one()
        job.status = "completed"
        job.completed_at = datetime.now(UTC)
        await session.flush()
        logger.info("Completed job %s", job_id)
        return job

    @staticmethod
    async def fail(session: AsyncSession, job_id: UUID, error: str) -> Job:
        """Handle job failure, with dead-letter routing when max attempts are exceeded."""
        stmt = select(Job).where(Job.id == job_id)
        res = await session.execute(stmt)
        job = res.scalar_one()
        job.last_error = error

        if job.attempts >= job.max_attempts:
            job.status = "failed"
            logger.error("Job %s permanently failed after %d attempts (dead-letter): %s", job_id, job.attempts, error)
        else:
            job.status = "pending"
            job.locked_at = None
            logger.warning("Job %s failed attempt %d/%d, re-queued: %s", job_id, job.attempts, job.max_attempts, error)

        await session.flush()
        return job


class JobWorker:
    """Background worker that pulls jobs from queues and executes registered handlers."""

    def __init__(
        self,
        indexer: IndexingService | None = None,
        retriever: HybridRetrievalService | None = None,
        reranker: LightGBMReranker | None = None,
        guard: MismatchGuard | None = None,
    ) -> None:
        self.indexer = indexer or IndexingService()
        self.retriever = retriever or HybridRetrievalService()
        self.reranker = reranker or LightGBMReranker()
        self.guard = guard or MismatchGuard()
        self._handlers: dict[str, JobHandler] = {}

        # Register default handlers
        self.register_handler("image_indexing", self._handle_image_indexing)
        self.register_handler("match_generation", self._handle_match_generation)

    def register_handler(self, queue: str, handler: JobHandler) -> None:
        """Register a handler function for a specific queue."""
        self._handlers[queue] = handler

    async def _handle_image_indexing(self, session: AsyncSession, payload: dict[str, Any]) -> None:
        """Handler for image indexing jobs."""
        source_id = payload["source_id"]
        url = payload["url"]
        local_path = payload.get("local_path")
        source = payload.get("source", "upload")

        raw_b64 = payload.get("image_bytes_b64")
        image_bytes: bytes | None = None
        if raw_b64:
            image_bytes = base64.b64decode(raw_b64)

        await self.indexer.index_image(
            session=session,
            source_id=source_id,
            url=url,
            image_bytes=image_bytes,
            source=source,
            local_path=local_path,
        )

    async def _handle_match_generation(self, session: AsyncSession, payload: dict[str, Any]) -> None:
        """Handler for generating matches for an article and recording suggestions."""
        article_id = UUID(str(payload["article_id"]))
        editor_id = UUID(str(payload["editor_id"])) if payload.get("editor_id") else None

        stmt = select(Article).where(Article.id == article_id)
        res = await session.execute(stmt)
        article = res.scalar_one()

        brief = ArticleBrief(
            primary_subject=article.primary_subject or article.title,
            must_have=article.must_have if isinstance(article.must_have, list) else [],
            must_not_have=article.must_not_have if isinstance(article.must_not_have, list) else [],
            mood=article.mood or "editorial",
            orientation=article.orientation if article.orientation in ("landscape", "portrait", "square") else "landscape",
        )

        # Stage 1: Hybrid retrieval
        candidates = await self.retriever.retrieve_candidates(
            session=session,
            query_text=article.title,
            final_k=10,
        )

        # Stage 2: Reranking
        scored_candidates = self.reranker.score_candidates(brief=brief, candidates=candidates)

        # Save top suggestions in DB if editor is provided
        if editor_id and scored_candidates:
            for cand, raw_score, cal_prob in scored_candidates[:3]:
                guard_res = self.guard.evaluate(
                    brief=brief,
                    image_tags={
                        "subject": cand.subject,
                        "category": cand.category,
                        "caption": cand.caption,
                        "orientation": cand.orientation or "landscape",
                        "people_present": cand.people_present or False,
                        "confidence": cand.vision_confidence or 0.9,
                    },
                    retrieval_score=cand.rrf_score,
                )
                suggestion = Suggestion(
                    article_id=article.id,
                    image_id=cand.image_id,
                    editor_id=editor_id,
                    retrieval_scores=cand.channel_scores,
                    reranker_score=float(raw_score),
                    guard_result=guard_res.decision,
                    guard_reasons=guard_res.reasons,
                    confidence=float(cal_prob),
                    reasons=guard_res.reasons,
                    status="pending",
                )
                session.add(suggestion)

    async def process_one(self, session: AsyncSession, queue: str) -> bool:
        """Process a single job from the specified queue. Returns True if a job was processed."""
        job = await QueueService.dequeue(session, queue)
        if not job:
            return False

        handler = self._handlers.get(queue)
        if not handler:
            await QueueService.fail(session, job.id, f"No handler registered for queue '{queue}'")
            return True

        try:
            await handler(session, job.payload)
            await QueueService.complete(session, job.id)
        except Exception as e:
            logger.exception("Error processing job %s on queue '%s': %s", job.id, queue, e)
            await QueueService.fail(session, job.id, str(e))

        return True
