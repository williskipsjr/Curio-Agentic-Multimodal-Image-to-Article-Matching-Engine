"""Live image indexing pipeline.

Coordinates:
- Perceptual hashing (imagehash) & SHA-256 deduplication
- Vision tagging (Gemini Flash with validation & retry)
- Dense caption embeddings (MiniLM, 384-d)
- Multimodal CLIP embeddings (open_clip, 512-d)
- PostgreSQL tsvector full-text search column updates
- Live index searchable timestamp logging
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import UTC, datetime
from io import BytesIO

import imagehash
from PIL import Image as PILImage
from sqlalchemy import func, literal_column, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Image
from curio.services.cost_guard import CostTracker
from curio.services.embeddings import EmbeddingService
from curio.services.vision import VisionTaggingService

logger = logging.getLogger(__name__)


class IndexingService:
    """Indexes images into PostgreSQL with vectors and keyword full-text search."""

    def __init__(
        self,
        vision_service: VisionTaggingService | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.vision = vision_service or VisionTaggingService()
        self.embeddings = embedding_service or EmbeddingService()

    @staticmethod
    def compute_hashes(image_bytes: bytes) -> tuple[str, str]:
        """Compute SHA-256 content hash and perceptual dhash."""
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        pil_img = PILImage.open(BytesIO(image_bytes))
        p_hash = str(imagehash.dhash(pil_img))
        return content_hash, p_hash

    async def index_image(
        self,
        session: AsyncSession,
        source_id: str,
        url: str,
        image_bytes: bytes,
        source: str = "upload",
        local_path: str | None = None,
    ) -> Image:
        """Full pipeline to index or update an image into PostgreSQL."""
        start_time = time.perf_counter()

        # 1. Hashing & Deduplication
        content_hash, p_hash = self.compute_hashes(image_bytes)

        # Check existing content hash
        existing = await session.execute(
            select(Image).where(Image.content_hash == content_hash)
        )
        found = existing.scalar_one_or_none()
        if found and found.searchable_at is not None:
            logger.info("Image %s already indexed (hash match)", source_id)
            return found

        # 2. Vision Tagging with retry
        vision_flagged = False
        try:
            tags, tokens = self.vision.tag_image_bytes(image_bytes)
            await CostTracker.record_call(
                session=session,
                model="gemini-3.5-flash-lite",
                operation="vision_tag",
                input_tokens=tokens["input"],
                output_tokens=tokens["output"],
            )
        except Exception as e:
            logger.error("Vision tagging permanently failed for %s: %s", source_id, e)
            vision_flagged = True
            tags = VisionTaggingService._heuristic_tag(image_bytes)

        # 3. Generate Embeddings
        # A. Caption text embedding (384-d)
        caption_text = f"{tags.subject}. {tags.caption}. {' '.join(tags.attributes)}"
        caption_vec = self.embeddings.embed_text(caption_text)

        # B. CLIP image embedding (512-d)
        clip_vec = self.embeddings.embed_image_bytes(image_bytes)

        # 4. Upsert into database
        now = datetime.now(UTC)
        tsv_expr = (
            func.setweight(func.to_tsvector("english", func.coalesce(tags.subject, "")), literal_column("'A'::\"char\"")).op("||")(
                func.setweight(func.to_tsvector("english", func.coalesce(tags.caption, "")), literal_column("'B'::\"char\""))
            ).op("||")(
                func.setweight(func.to_tsvector("english", func.coalesce(tags.category, "")), literal_column("'C'::\"char\""))
            )
        )

        stmt = (
            pg_insert(Image)
            .values(
                source_id=source_id,
                source=source,
                url=url,
                local_path=local_path,
                subject=tags.subject,
                category=tags.category,
                attributes=tags.attributes,
                caption=tags.caption,
                people_present=tags.people_present,
                orientation=tags.orientation,
                vision_confidence=tags.confidence,
                vision_tagged=not vision_flagged,
                vision_flagged=vision_flagged,
                caption_embedding=caption_vec,
                clip_embedding=clip_vec,
                content_hash=content_hash,
                perceptual_hash=p_hash,
                search_tsv=tsv_expr,
                searchable_at=now,
            )
            .on_conflict_do_update(
                index_elements=["source_id"],
                set_={
                    "subject": tags.subject,
                    "category": tags.category,
                    "attributes": tags.attributes,
                    "caption": tags.caption,
                    "people_present": tags.people_present,
                    "orientation": tags.orientation,
                    "vision_confidence": tags.confidence,
                    "vision_tagged": not vision_flagged,
                    "vision_flagged": vision_flagged,
                    "caption_embedding": caption_vec,
                    "clip_embedding": clip_vec,
                    "content_hash": content_hash,
                    "perceptual_hash": p_hash,
                    "search_tsv": tsv_expr,
                    "searchable_at": now,
                    "updated_at": now,
                },
            )
            .returning(Image)
        )

        res = await session.execute(stmt)
        record = res.scalar_one()
        await session.commit()
        await session.refresh(record)

        duration = time.perf_counter() - start_time
        logger.info(
            "Indexed image '%s' in %.2fs (searchable: %s, tagged: %s)",
            source_id,
            duration,
            record.searchable_at is not None,
            record.vision_tagged,
        )
        return record
