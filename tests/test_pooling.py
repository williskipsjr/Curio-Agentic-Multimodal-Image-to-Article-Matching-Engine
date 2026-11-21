"""Tests for the pooling script."""

from curio.scripts.create_pool import (
    ArticlePool,
    PooledCandidate,
    merge_retriever_runs,
)


class TestPooling:
    """Test TREC-style pooling logic."""

    def test_merge_retriever_runs_deduplication(self) -> None:
        """Images retrieved by multiple systems should be merged cleanly."""
        runs = {
            "keyword": [
                {"image_id": "img1", "score": 0.9},
                {"image_id": "img2", "score": 0.8},
            ],
            "dense": [
                {"image_id": "img2", "score": 0.85},
                {"image_id": "img3", "score": 0.75},
            ],
            "clip": [
                {"image_id": "img1", "score": 0.95},
                {"image_id": "img4", "score": 0.70},
            ],
        }

        pool = merge_retriever_runs(runs, depth=20)
        # Unique images: img1, img2, img3, img4
        assert len(pool) == 4
        by_id = {c.image_id: c for c in pool}

        # img1 appeared in keyword and clip
        assert sorted(by_id["img1"].retrievers) == ["clip", "keyword"]
        assert by_id["img1"].ranks["keyword"] == 1
        assert by_id["img1"].ranks["clip"] == 1

        # img2 appeared in keyword and dense
        assert sorted(by_id["img2"].retrievers) == ["dense", "keyword"]
        assert by_id["img2"].ranks["keyword"] == 2
        assert by_id["img2"].ranks["dense"] == 1

    def test_merge_retriever_runs_respects_depth(self) -> None:
        """Pooling should only take top-k candidates from each retriever."""
        runs = {
            "keyword": [{"image_id": f"img_{i}", "score": 1.0 - i * 0.01} for i in range(50)]
        }
        pool = merge_retriever_runs(runs, depth=20)
        assert len(pool) == 20
        assert pool[0].ranks["keyword"] == 1
        assert pool[-1].ranks["keyword"] == 20

    def test_article_pool_metadata(self) -> None:
        candidates = [
            PooledCandidate(
                image_id="img1",
                retrievers=["keyword"],
                ranks={"keyword": 1},
                scores={"keyword": 0.9},
            )
        ]
        pool = ArticlePool(
            article_id="art-1",
            article_title="Red Fox",
            split="dev",
            candidates=candidates,
        )
        assert pool.total_candidates == 1
        assert pool.split == "dev"
