"""TREC-style pooling script for search evaluation.

For each evaluation article:
1. Runs the three Stage 1 retrievers (keyword BM25, dense caption, CLIP image-vs-text)
2. Takes the top-k candidates (default k=20) from each retriever
3. Merges them into an unranked, deduplicated pool for human / LLM evaluation
4. Stores candidate pairs ready to be graded (0=wrong, 1=acceptable, 2=ideal)

Usage:
    uv run python -m curio.scripts.create_pool --split dev --depth 20
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
POOL_DIR = DATA_DIR / "pools"


@dataclass
class PooledCandidate:
    """A candidate image pooled for an article."""

    image_id: str
    retrievers: list[str]  # e.g., ["keyword", "clip"]
    ranks: dict[str, int]  # e.g., {"keyword": 3, "clip": 1}
    scores: dict[str, float]  # raw retriever scores


@dataclass
class ArticlePool:
    """The complete candidate pool for an article."""

    article_id: str
    article_title: str
    split: str  # 'dev' | 'test'
    candidates: list[PooledCandidate]

    @property
    def total_candidates(self) -> int:
        return len(self.candidates)


def merge_retriever_runs(
    runs: dict[str, list[dict[str, Any]]],
    depth: int = 20,
) -> list[PooledCandidate]:
    """Merge ranked outputs from multiple retrievers using depth-k pooling.

    Args:
        runs: mapping from retriever_name (e.g. 'keyword', 'dense', 'clip')
              to a list of dicts with 'image_id' and 'score'.
        depth: top-k cutoff per retriever to include in the pool.

    Returns:
        Deduplicated list of PooledCandidate objects.
    """
    merged: dict[str, dict[str, Any]] = {}

    for retriever_name, candidates in runs.items():
        for rank, cand in enumerate(candidates[:depth], start=1):
            img_id = str(cand["image_id"])
            score = float(cand.get("score", 0.0))

            if img_id not in merged:
                merged[img_id] = {
                    "image_id": img_id,
                    "retrievers": [],
                    "ranks": {},
                    "scores": {},
                }

            merged[img_id]["retrievers"].append(retriever_name)
            merged[img_id]["ranks"][retriever_name] = rank
            merged[img_id]["scores"][retriever_name] = score

    return [
        PooledCandidate(
            image_id=c["image_id"],
            retrievers=sorted(c["retrievers"]),
            ranks=c["ranks"],
            scores=c["scores"],
        )
        for c in merged.values()
    ]


def build_evaluation_pool(
    articles: list[dict[str, Any]],
    retriever_fn: Any,
    split: str = "dev",
    depth: int = 20,
) -> list[ArticlePool]:
    """Build evaluation pools across a set of articles.

    Args:
        articles: list of article dicts with 'id' and 'title'.
        retriever_fn: callable(article) -> dict[str, list[dict[str, Any]]]
                      returning top candidates for each retriever.
        split: 'dev' or 'test'.
        depth: depth cutoff (k=20).
    """
    pools: list[ArticlePool] = []

    for art in articles:
        art_id = str(art.get("id") or art.get("source_id"))
        title = art.get("title", "")
        runs = retriever_fn(art)
        candidates = merge_retriever_runs(runs, depth=depth)

        pools.append(
            ArticlePool(
                article_id=art_id,
                article_title=title,
                split=split,
                candidates=candidates,
            )
        )

    return pools


def save_pool(pool: ArticlePool, output_dir: Path) -> Path:
    """Save an ArticlePool to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = pool.article_title.lower().replace(" ", "_").replace("/", "_")[:60]
    out_file = output_dir / f"pool_{pool.split}_{slug}.json"

    data = {
        "article_id": pool.article_id,
        "article_title": pool.article_title,
        "split": pool.split,
        "total_candidates": pool.total_candidates,
        "candidates": [asdict(c) for c in pool.candidates],
    }

    out_file.write_text(json.dumps(data, indent=2))
    return out_file
