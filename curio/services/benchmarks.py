"""Information Retrieval metrics and baseline ablation benchmark runner.

Calculates:
- Recall@50
- Top-1 Precision
- nDCG@5 with 95% bootstrap confidence intervals

Generates the B0-B3 retrieval ablation table:
- B0: Keyword only (BM25)
- B1: Dense text only (MiniLM)
- B2: Multimodal CLIP only (ViT-B/32)
- B3: Hybrid with Reciprocal Rank Fusion (RRF)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalMetrics:
    """Metrics calculated over an evaluation split."""

    recall_at_50: float
    ndcg_at_5: float
    ndcg_ci_low: float
    ndcg_ci_high: float
    top1_precision: float


def dcg_at_k(relevances: list[int], k: int = 5) -> float:
    """Compute Discounted Cumulative Gain at rank k."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    return float(np.sum((2**rel - 1) / np.log2(np.arange(2, rel.size + 2))))


def ndcg_at_k(relevances: list[int], k: int = 5) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank k."""
    actual_dcg = dcg_at_k(relevances, k)
    ideal_dcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if ideal_dcg == 0.0:
        return 0.0
    return float(actual_dcg / ideal_dcg)


def bootstrap_ci(
    values: list[float], n_bootstraps: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    """Calculate 95% bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    bootstrapped_means = []
    rng = np.random.default_rng(42)

    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrapped_means.append(np.mean(sample))

    low = float(np.percentile(bootstrapped_means, 100 * (alpha / 2)))
    high = float(np.percentile(bootstrapped_means, 100 * (1 - alpha / 2)))
    return round(low, 4), round(high, 4)


def evaluate_run(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    k: int = 5,
) -> EvalMetrics:
    """Evaluate retrieval rankings against relevance judgments (qrels).

    Args:
        qrels: mapping from query_id -> {image_id: grade (0, 1, 2)}
        run: mapping from query_id -> ranked list of image_ids
        k: cutoff for nDCG
    """
    ndcgs = []
    recalls = []
    top1s = []

    for q_id, ranked_docs in run.items():
        doc_grades = qrels.get(q_id, {})
        relevance_vector = [doc_grades.get(doc, 0) for doc in ranked_docs]

        # nDCG@k
        score_ndcg = ndcg_at_k(relevance_vector, k=k)
        ndcgs.append(score_ndcg)

        # Top-1 precision (ideal=2 or acceptable=1)
        top1 = 1.0 if relevance_vector and relevance_vector[0] >= 1 else 0.0
        top1s.append(top1)

        # Recall@50 (has at least one relevant document in top 50)
        has_rel = any(g >= 1 for g in relevance_vector[:50])
        recalls.append(1.0 if has_rel else 0.0)

    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    ci_low, ci_high = bootstrap_ci(ndcgs)
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    mean_top1 = float(np.mean(top1s)) if top1s else 0.0

    return EvalMetrics(
        recall_at_50=round(mean_recall, 4),
        ndcg_at_5=round(mean_ndcg, 4),
        ndcg_ci_low=ci_low,
        ndcg_ci_high=ci_high,
        top1_precision=round(mean_top1, 4),
    )
