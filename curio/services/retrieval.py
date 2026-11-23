"""Stage 1 Hybrid Retrieval and Reciprocal Rank Fusion (RRF).

Combines three complementary retrieval channels in parallel:
1. Keyword BM25 / PostgreSQL full-text search over tags and captions
2. Dense text similarity: MiniLM caption-vs-article embeddings (pgvector cosine)
3. Multimodal CLIP similarity: ViT-B/32 image pixels vs article text (pgvector cosine)

Merges channels using Reciprocal Rank Fusion (RRF) with constant k=60.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Image
from curio.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class RetrievalCandidate:
    """A retrieved image candidate with channel rank and score breakdown."""

    image_id: UUID
    source_id: str
    url: str
    subject: str | None
    category: str | None
    caption: str | None
    orientation: str | None
    people_present: bool | None
    vision_confidence: float | None
    channel_scores: dict[str, float]
    channel_ranks: dict[str, int]
    rrf_score: float = 0.0


def compute_rrf(
    channel_runs: dict[str, list[dict[str, Any]]],
    k: int = 60,
    top_n: int = 50,
) -> list[RetrievalCandidate]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion.

    Score(d) = sum_{channel} ( 1 / (k + rank_{channel}(d)) )
    """
    candidates_by_id: dict[UUID, dict[str, Any]] = {}
    rrf_scores: dict[UUID, float] = {}

    for channel, run in channel_runs.items():
        for rank, item in enumerate(run, start=1):
            img_id = item["id"]
            if img_id not in candidates_by_id:
                candidates_by_id[img_id] = {
                    "id": img_id,
                    "source_id": item["source_id"],
                    "url": item["url"],
                    "subject": item.get("subject"),
                    "category": item.get("category"),
                    "caption": item.get("caption"),
                    "orientation": item.get("orientation"),
                    "people_present": item.get("people_present"),
                    "vision_confidence": item.get("vision_confidence"),
                    "channel_scores": {},
                    "channel_ranks": {},
                }
                rrf_scores[img_id] = 0.0

            candidates_by_id[img_id]["channel_scores"][channel] = float(item.get("score", 0.0))
            candidates_by_id[img_id]["channel_ranks"][channel] = rank
            rrf_scores[img_id] += 1.0 / (k + rank)

    # Sort descending by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    results: list[RetrievalCandidate] = []
    for img_id in sorted_ids[:top_n]:
        info = candidates_by_id[img_id]
        results.append(
            RetrievalCandidate(
                image_id=info["id"],
                source_id=info["source_id"],
                url=info["url"],
                subject=info["subject"],
                category=info["category"],
                caption=info["caption"],
                orientation=info["orientation"],
                people_present=info["people_present"],
                vision_confidence=info["vision_confidence"],
                channel_scores=info["channel_scores"],
                channel_ranks=info["channel_ranks"],
                rrf_score=rrf_scores[img_id],
            )
        )

    return results


class HybridRetrievalService:
    """Orchestrates 3-way hybrid search: BM25/FTS + Dense Text + Multimodal CLIP."""

    def __init__(self, embedding_service: EmbeddingService | None = None) -> None:
        self.embeddings = embedding_service or EmbeddingService()

    async def search_keyword(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Channel 1: BM25 / Full-Text Search via Postgres tsvector."""
        # Clean query terms for websearch_to_tsquery or plainto_tsquery
        clean_query = query_text.strip()
        if not clean_query:
            return []

        # Use ts_rank_cd for Cover Density ranking
        tsquery = func.plainto_tsquery("english", clean_query)
        rank_expr = func.ts_rank_cd(Image.search_tsv, tsquery)

        stmt = (
            select(
                Image.id,
                Image.source_id,
                Image.url,
                Image.subject,
                Image.category,
                Image.caption,
                Image.orientation,
                Image.people_present,
                Image.vision_confidence,
                rank_expr.label("score"),
            )
            .where(
                Image.search_tsv.op("@@")(tsquery),
                Image.is_deleted.is_(False),
                Image.searchable_at.is_not(None),
            )
            .order_by(desc("score"))
            .limit(limit)
        )

        res = await session.execute(stmt)
        return [dict(r._mapping) for r in res.all()]

    async def search_dense(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Channel 2: Dense text embedding similarity (MiniLM 384-d, cosine)."""
        query_vec = self.embeddings.embed_text(query_text)

        # 1 - cosine distance = cosine similarity
        cosine_dist = Image.caption_embedding.cosine_distance(query_vec)
        score_expr = (1.0 - cosine_dist).label("score")

        stmt = (
            select(
                Image.id,
                Image.source_id,
                Image.url,
                Image.subject,
                Image.category,
                Image.caption,
                Image.orientation,
                Image.people_present,
                Image.vision_confidence,
                score_expr,
            )
            .where(
                Image.caption_embedding.is_not(None),
                Image.is_deleted.is_(False),
                Image.searchable_at.is_not(None),
            )
            .order_by(cosine_dist)
            .limit(limit)
        )

        res = await session.execute(stmt)
        return [dict(r._mapping) for r in res.all()]

    async def search_clip(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Channel 3: Multimodal CLIP cross-modal similarity (ViT-B/32 512-d, cosine)."""
        clip_vec = self.embeddings.embed_article_for_clip(query_text)

        cosine_dist = Image.clip_embedding.cosine_distance(clip_vec)
        score_expr = (1.0 - cosine_dist).label("score")

        stmt = (
            select(
                Image.id,
                Image.source_id,
                Image.url,
                Image.subject,
                Image.category,
                Image.caption,
                Image.orientation,
                Image.people_present,
                Image.vision_confidence,
                score_expr,
            )
            .where(
                Image.clip_embedding.is_not(None),
                Image.is_deleted.is_(False),
                Image.searchable_at.is_not(None),
            )
            .order_by(cosine_dist)
            .limit(limit)
        )

        res = await session.execute(stmt)
        return [dict(r._mapping) for r in res.all()]

    async def retrieve_candidates(
        self,
        session: AsyncSession,
        query_text: str,
        top_k_per_channel: int = 50,
        final_k: int = 50,
        rrf_constant: int = 60,
    ) -> list[RetrievalCandidate]:
        """Execute all 3 channels sequentially and merge with Reciprocal Rank Fusion.

        Note: AsyncSession is not safe for concurrent operations, so channels
        are run sequentially rather than with asyncio.gather().
        """
        keyword_run = await self.search_keyword(session, query_text, limit=top_k_per_channel)
        dense_run = await self.search_dense(session, query_text, limit=top_k_per_channel)
        clip_run = await self.search_clip(session, query_text, limit=top_k_per_channel)

        channel_runs = {
            "keyword": keyword_run,
            "dense": dense_run,
            "clip": clip_run,
        }

        fused = compute_rrf(channel_runs, k=rrf_constant, top_n=final_k)
        logger.info(
            "Hybrid retrieval returned %d candidates (keyword=%d, dense=%d, clip=%d)",
            len(fused),
            len(keyword_run),
            len(dense_run),
            len(clip_run),
        )
        return fused
