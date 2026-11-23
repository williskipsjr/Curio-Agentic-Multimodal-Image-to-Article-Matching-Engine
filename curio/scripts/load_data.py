"""Load downloaded images and articles into PostgreSQL.

Reads the JSON manifests produced by download_images and download_articles,
then upserts records into the images and articles tables.

Usage:
    uv run python -m curio.scripts.load_data
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from curio.config import settings
from curio.models import Article, Base, Image

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_MANIFEST = DATA_DIR / "image_manifest.json"
ARTICLE_MANIFEST = DATA_DIR / "article_manifest.json"


async def ensure_extensions(engine) -> None:  # type: ignore[no-untyped-def]
    """Enable required PostgreSQL extensions."""
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        logger.info("PostgreSQL extensions enabled: vector, pg_trgm")


async def create_tables(engine) -> None:  # type: ignore[no-untyped-def]
    """Create all tables from ORM models."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("All tables created")


async def load_images(session: AsyncSession) -> int:
    """Load images from the manifest into the database."""
    if not IMAGE_MANIFEST.exists():
        logger.warning("Image manifest not found at %s — skipping", IMAGE_MANIFEST)
        return 0

    manifest = json.loads(IMAGE_MANIFEST.read_text())

    all_images = manifest.get("corpus", []) + manifest.get("holdback", [])
    loaded = 0

    for entry in all_images:
        category_q = entry.get("category_query", "")
        caption = (
            entry.get("unsplash_alt_description")
            or entry.get("unsplash_description")
            or f"A photo of {category_q}"
        )
        is_people = any(
            p in category_q.lower()
            for p in ["people", "portrait", "person", "group of people"]
        )

        stmt = (
            pg_insert(Image.__table__)
            .values(
                source_id=entry["source_id"],
                source=entry.get("source", "unsplash"),
                url=entry["url"],
                local_path=entry.get("local_path"),
                content_hash=entry.get("content_hash"),
                orientation=entry.get("orientation", "landscape"),
                subject=category_q,
                category=entry.get("confusable_group") or category_q,
                caption=caption,
                attributes=[category_q, entry.get("photographer", "")],
                people_present=is_people,
                vision_confidence=0.95,
                vision_tagged=True,
                searchable_at=func.now(),
            )
            .on_conflict_do_update(
                index_elements=["source_id"],
                set_={
                    "subject": category_q,
                    "category": entry.get("confusable_group") or category_q,
                    "caption": caption,
                    "attributes": [category_q, entry.get("photographer", "")],
                    "people_present": is_people,
                    "orientation": entry.get("orientation", "landscape"),
                    "vision_confidence": 0.95,
                    "vision_tagged": True,
                    "searchable_at": func.now(),
                },
            )
        )

        await session.execute(stmt)
        loaded += 1

    await session.commit()
    logger.info("Loaded %d image records", loaded)
    return loaded


async def load_articles(session: AsyncSession) -> int:
    """Load articles from the manifest into the database."""
    if not ARTICLE_MANIFEST.exists():
        logger.warning("Article manifest not found at %s — skipping", ARTICLE_MANIFEST)
        return 0

    manifest = json.loads(ARTICLE_MANIFEST.read_text())

    all_articles = manifest.get("dev", []) + manifest.get("test", [])
    loaded = 0

    for entry in all_articles:
        title = entry["title"]
        cat = entry.get("category", "")

        stmt = (
            pg_insert(Article.__table__)
            .values(
                source_id=entry["source_id"],
                title=title,
                body=entry["body"],
                source_url=entry.get("source_url"),
                is_no_match=entry.get("is_no_match", False),
                split=entry.get("split"),
                primary_subject=title,
                must_have=[cat, title] if cat else [title],
                must_not_have=[],
                mood="editorial",
                orientation="any",
            )
            .on_conflict_do_update(
                index_elements=["source_id"],
                set_={
                    "title": title,
                    "body": entry["body"],
                    "primary_subject": title,
                    "must_have": [cat, title] if cat else [title],
                    "is_no_match": entry.get("is_no_match", False),
                    "split": entry.get("split"),
                },
            )
        )

        await session.execute(stmt)
        loaded += 1

    await session.commit()
    logger.info("Loaded %d article records", loaded)
    return loaded
    logger.info("Loaded %d article records", loaded)
    return loaded


async def main() -> None:
    """Main data loading pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    engine = create_async_engine(settings.async_database_url, echo=False)

    try:
        # 1. Enable extensions
        await ensure_extensions(engine)

        # 2. Create tables
        await create_tables(engine)

        # 3. Load data
        async with AsyncSession(engine) as session:
            img_count = await load_images(session)
            art_count = await load_articles(session)

        logger.info(
            "Data loading complete: %d images, %d articles", img_count, art_count
        )

    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
