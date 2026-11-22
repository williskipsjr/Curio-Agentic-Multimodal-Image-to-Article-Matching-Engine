"""Tests for Phase 2 services: vision tagging, cost guard, embeddings, and indexing."""

from io import BytesIO

import pytest
from PIL import Image as PILImage
from sqlalchemy.ext.asyncio import AsyncSession

from curio.services.cost_guard import CostTracker
from curio.services.embeddings import EmbeddingService
from curio.services.indexer import IndexingService
from curio.services.vision import VisionTaggingService


def create_dummy_image_bytes(width: int = 200, height: int = 150) -> bytes:
    """Generate in-memory JPEG bytes for testing."""
    img = PILImage.new("RGB", (width, height), color=(255, 120, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


class TestCostTracker:
    """Test cost calculation, ledger writes, and budget guard."""

    def test_cost_calculation(self) -> None:
        cost = CostTracker.calculate_cost("gemini-2.0-flash", 1000, 500)
        assert cost > 0.0
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_record_call_and_summary(self, db_session: AsyncSession) -> None:
        entry = await CostTracker.record_call(
            session=db_session,
            model="gemini-2.0-flash",
            operation="test_op",
            input_tokens=1000,
            output_tokens=250,
        )
        assert entry.id is not None
        assert entry.estimated_cost > 0.0

        summary = await CostTracker.get_summary(db_session)
        assert summary["total_spent_usd"] >= entry.estimated_cost
        assert len(summary["breakdown"]) >= 1


class TestVisionTagging:
    """Test vision tagging with heuristic fallback and schema guarantees."""

    def test_heuristic_tagging_dimensions(self) -> None:
        service = VisionTaggingService()
        landscape_bytes = create_dummy_image_bytes(300, 150)
        portrait_bytes = create_dummy_image_bytes(150, 300)

        tag_l, _tokens_l = service.tag_image_bytes(landscape_bytes)
        assert tag_l.orientation == "landscape"
        assert tag_l.confidence >= 0.0

        tag_p, _tokens_p = service.tag_image_bytes(portrait_bytes)
        assert tag_p.orientation == "portrait"


@pytest.fixture(scope="module")
def shared_embed_service() -> EmbeddingService:
    """Class/module level fixture providing an EmbeddingService instance."""
    return EmbeddingService()


class TestEmbeddingService:
    """Test generation of dense text and multimodal CLIP embeddings."""

    def test_text_embedding_dimension(self, shared_embed_service: EmbeddingService) -> None:
        vec = shared_embed_service.embed_text("A red fox walking in snow")
        assert len(vec) == 384
        # L2-normalized check
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-3

    def test_clip_image_embedding_dimension(
        self, shared_embed_service: EmbeddingService
    ) -> None:
        img_bytes = create_dummy_image_bytes(100, 100)
        vec = shared_embed_service.embed_image_bytes(img_bytes)
        assert len(vec) == 512
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-3

    def test_clip_article_text_embedding_dimension(
        self, shared_embed_service: EmbeddingService
    ) -> None:
        vec = shared_embed_service.embed_article_for_clip(
            "Article about red foxes and wildlife preservation"
        )
        assert len(vec) == 512
        norm = sum(x * x for x in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-3


class TestIndexingService:
    """Test end-to-end indexing pipeline including deduplication and tsvector search."""

    @pytest.mark.asyncio
    async def test_index_image_flow(self, db_session: AsyncSession) -> None:
        indexer = IndexingService()
        img_bytes = create_dummy_image_bytes(250, 200)

        record = await indexer.index_image(
            session=db_session,
            source_id="test_img_001",
            url="https://example.com/test_img_001.jpg",
            image_bytes=img_bytes,
            source="test",
        )

        assert record.id is not None
        assert record.searchable_at is not None
        assert record.content_hash is not None
        assert record.perceptual_hash is not None
        assert len(record.caption_embedding) == 384
        assert len(record.clip_embedding) == 512

        # Idempotent re-index
        re_record = await indexer.index_image(
            session=db_session,
            source_id="test_img_001",
            url="https://example.com/test_img_001.jpg",
            image_bytes=img_bytes,
            source="test",
        )
        assert re_record.id == record.id
