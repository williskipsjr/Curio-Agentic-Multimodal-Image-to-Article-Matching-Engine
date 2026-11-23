"""Index loaded images and articles with dense and multimodal embeddings and tsvectors."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from curio.config import settings
from curio.models import Article, Image
from curio.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


async def index_articles(session: AsyncSession, embeddings: EmbeddingService) -> int:
    """Generate brief embeddings for all articles."""
    stmt = select(Article).where(Article.brief_embedding.is_(None))
    res = await session.execute(stmt)
    articles = res.scalars().all()

    if not articles:
        logger.info("All articles already have embeddings.")
        return 0

    logger.info("Generating embeddings for %d articles...", len(articles))
    count = 0
    for art in articles:
        text_to_embed = f"{art.title}. {art.body[:300]}"
        vec = embeddings.embed_text(text_to_embed)
        art.brief_embedding = vec
        count += 1
        if count % 50 == 0:
            await session.commit()
            logger.info("Embedded %d/%d articles", count, len(articles))

    await session.commit()
    logger.info("Successfully embedded %d articles", count)
    return count


async def index_images(
    session: AsyncSession, embeddings: EmbeddingService, batch_size: int = 50
) -> int:
    """Generate caption and CLIP embeddings and tsvectors for images."""
    stmt = select(Image).where(
        Image.clip_embedding.is_(None) | Image.caption_embedding.is_(None)
    )
    res = await session.execute(stmt)
    images = res.scalars().all()

    if not images:
        logger.info("All images already have embeddings.")
        return 0

    logger.info("Generating embeddings and search indexes for %d images...", len(images))
    count = 0

    for img in images:
        # 1. Caption dense embedding (MiniLM)
        caption_text = img.caption or img.subject or "general photograph"
        img.caption_embedding = embeddings.embed_text(caption_text)

        # 2. Multimodal CLIP embedding (OpenCLIP)
        if img.local_path:
            img_path = PROJECT_ROOT / img.local_path
            if img_path.exists():
                try:
                    img_bytes = img_path.read_bytes()
                    img.clip_embedding = embeddings.embed_image_bytes(img_bytes)
                except Exception as e:
                    logger.warning("Failed CLIP embedding for %s: %s", img.source_id, e)
                    # Fallback to text CLIP embedding
                    img.clip_embedding = embeddings.embed_article_for_clip(caption_text)
            else:
                img.clip_embedding = embeddings.embed_article_for_clip(caption_text)
        else:
            img.clip_embedding = embeddings.embed_article_for_clip(caption_text)

        # 3. PostgreSQL tsvector update
        subj_weight = func.setweight(
            func.to_tsvector("english", func.coalesce(img.subject, "")),
            literal_column("'A'::\"char\""),
        )
        caption_weight = func.setweight(
            func.to_tsvector("english", func.coalesce(img.caption, "")),
            literal_column("'B'::\"char\""),
        )
        cat_weight = func.setweight(
            func.to_tsvector("english", func.coalesce(img.category, "")),
            literal_column("'C'::\"char\""),
        )
        img.search_tsv = subj_weight.op("||")(caption_weight).op("||")(cat_weight)
        img.searchable_at = func.now()

        count += 1
        if count % batch_size == 0:
            await session.commit()
            logger.info("Indexed and embedded %d/%d images", count, len(images))

    await session.commit()
    logger.info("Successfully indexed and embedded %d images", count)
    return count


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    embeddings = EmbeddingService()
    engine = create_async_engine(settings.async_database_url, echo=False)
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with session_maker() as session:
            await index_articles(session, embeddings)
            await index_images(session, embeddings)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
