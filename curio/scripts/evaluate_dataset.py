"""Evaluation script for the Curio Multimodal Image-to-Article Matching Engine on real dataset.

Evaluates:
1. 3-Way Hybrid Retrieval (BM25 Full-text + Dense Text MiniLM + OpenCLIP ViT-B-32) with RRF
2. Mismatch Guard on confusable animal and vehicle pairs
3. Candidate scoring and ranking
"""

from __future__ import annotations

import asyncio
import logging
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from curio.config import settings
from curio.models import Article, Image
from curio.schemas import ArticleBrief, VisionTags
from curio.services.embeddings import EmbeddingService
from curio.services.guard import MismatchGuard
from curio.services.retrieval import HybridRetrievalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_dataset")


async def run_evaluation():
    engine = create_async_engine(settings.async_database_url, echo=False)
    session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    embeddings = EmbeddingService()
    retriever = HybridRetrievalService(embedding_service=embeddings)
    guard = MismatchGuard(acceptance_threshold=0.015)

    async with session_maker() as session:
        # Load sample articles across match and confusable categories
        stmt = (
            select(Article)
            .where(Article.brief_embedding.is_not(None))
            .limit(20)
        )
        res = await session.execute(stmt)
        articles = res.scalars().all()

        logger.info("Evaluating Curio Pipeline across %d real articles...", len(articles))

        total_articles = 0
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        total_retrieved = 0
        guard_accepts = 0
        guard_vetoes = 0

        for art in articles:
            total_articles += 1
            query = art.title
            logger.info("--------------------------------------------------")
            logger.info("Article [%d/%d]: '%s' (Subject: %s)", total_articles, len(articles), query, art.primary_subject)

            # Build an ArticleBrief
            brief = ArticleBrief(
                primary_subject=art.primary_subject or art.title,
                must_have=art.must_have if isinstance(art.must_have, list) else [],
                must_not_have=art.must_not_have if isinstance(art.must_not_have, list) else [],
                mood=art.mood or "editorial",
                orientation=art.orientation if art.orientation in ("landscape", "portrait", "square") else "landscape",
            )

            # 1. Stage 1: 3-way Hybrid Retrieval with RRF
            candidates = await retriever.retrieve_candidates(
                session=session,
                query_text=query,
                top_k_per_channel=30,
                final_k=15,
            )
            total_retrieved += len(candidates)

            if not candidates:
                logger.warning("No candidates retrieved for '%s'", query)
                continue

            # Check top candidates with Mismatch Guard
            for rank, c in enumerate(candidates[:5], start=1):
                image_tags = {
                    "subject": c.subject or "",
                    "category": c.category or "",
                    "caption": c.caption or "",
                    "attributes": [],
                    "orientation": c.orientation or "landscape",
                    "people_present": c.people_present or False,
                    "confidence": c.vision_confidence or 0.9,
                }

                guard_res = guard.evaluate(
                    brief=brief,
                    image_tags=image_tags,
                    retrieval_score=c.rrf_score,
                )

                if guard_res.decision == "ACCEPT":
                    guard_accepts += 1
                    status = "ACCEPT"
                else:
                    guard_vetoes += 1
                    status = f"{guard_res.decision} ({','.join(guard_res.reasons)})"

                logger.info(
                    "  #%d [%s] (RRF=%.4f) Subj='%s' Cat='%s' ChScores=%s",
                    rank,
                    status,
                    c.rrf_score,
                    c.subject,
                    c.category,
                    {k: round(v, 3) for k, v in c.channel_scores.items()},
                )

            # Measure Recall / Hit@K
            keywords = [w.lower() for w in query.replace("-", " ").split() if len(w) > 3]
            def matches(cand):
                txt = f"{cand.subject or ''} {cand.category or ''} {cand.caption or ''}".lower()
                return any(k in txt for k in keywords)

            if matches(candidates[0]):
                hits_at_1 += 1
                hits_at_5 += 1
                hits_at_10 += 1
            elif any(matches(c) for c in candidates[:5]):
                hits_at_5 += 1
                hits_at_10 += 1
            elif any(matches(c) for c in candidates[:10]):
                hits_at_10 += 1

        logger.info("==================================================")
        logger.info("DATASET MATCHING EVALUATION RESULTS:")
        logger.info("Total Evaluated Articles: %d", total_articles)
        logger.info("Recall / Hit@1:  %d/%d (%.1f%%)", hits_at_1, total_articles, (hits_at_1 / total_articles) * 100)
        logger.info("Recall / Hit@5:  %d/%d (%.1f%%)", hits_at_5, total_articles, (hits_at_5 / total_articles) * 100)
        logger.info("Recall / Hit@10: %d/%d (%.1f%%)", hits_at_10, total_articles, (hits_at_10 / total_articles) * 100)
        logger.info("Guard Accepts: %d, Guard Vetoes / Cautions: %d", guard_accepts, guard_vetoes)
        logger.info("Average Candidates per Article: %.1f", total_retrieved / total_articles)
        logger.info("==================================================")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(run_evaluation())
