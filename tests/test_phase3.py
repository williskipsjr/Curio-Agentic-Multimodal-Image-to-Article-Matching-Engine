"""Tests for Phase 3: Hybrid retrieval, Reciprocal Rank Fusion, Mismatch Guard, and Judge."""

from uuid import uuid4

import pytest

from curio.schemas import ArticleBrief
from curio.services.benchmarks import bootstrap_ci, evaluate_run, ndcg_at_k
from curio.services.guard import MismatchGuard
from curio.services.judge import LLMJudge, compute_cohens_kappa
from curio.services.retrieval import (
    compute_rrf,
)


class TestReciprocalRankFusion:
    """Test RRF fusion logic."""

    def test_rrf_scoring(self) -> None:
        id1, id2, id3 = uuid4(), uuid4(), uuid4()

        channel_runs = {
            "keyword": [
                {"id": id1, "source_id": "s1", "url": "u1", "score": 10.0},
                {"id": id2, "source_id": "s2", "url": "u2", "score": 5.0},
            ],
            "dense": [
                {"id": id2, "source_id": "s2", "url": "u2", "score": 0.9},
                {"id": id1, "source_id": "s1", "url": "u1", "score": 0.8},
            ],
            "clip": [
                {"id": id1, "source_id": "s1", "url": "u1", "score": 0.95},
                {"id": id3, "source_id": "s3", "url": "u3", "score": 0.7},
            ],
        }

        fused = compute_rrf(channel_runs, k=60, top_n=10)
        assert len(fused) == 3

        # id1 ranked 1st in keyword, 2nd in dense, 1st in clip -> should lead
        assert fused[0].image_id == id1
        assert fused[0].rrf_score > fused[1].rrf_score


class TestMismatchGuard:
    """Test the calibrated mismatch guard on positive, confusable, and negative cases."""

    @pytest.fixture
    def guard(self) -> MismatchGuard:
        return MismatchGuard(acceptance_threshold=0.50)

    def test_accepts_valid_match(self, guard: MismatchGuard) -> None:
        brief = ArticleBrief(
            primary_subject="red fox",
            must_have=["fox", "snow"],
            must_not_have=["wolf", "dog"],
            mood="wild",
            orientation="landscape",
        )
        image_tags = {
            "subject": "red fox",
            "category": "wildlife",
            "attributes": ["snow", "winter", "fur"],
            "caption": "A red fox in a snowy field.",
            "people_present": False,
            "confidence": 0.92,
        }

        result = guard.evaluate(brief, image_tags, retrieval_score=0.85)
        assert result.decision == "ACCEPT"
        assert result.confidence >= 0.50

    def test_rejects_confusable_lookalike(self, guard: MismatchGuard) -> None:
        """The Fox and the Wolf test: wolf must be rejected with an explicit reason."""
        brief = ArticleBrief(
            primary_subject="red fox",
            must_have=["fox"],
            must_not_have=[],
            mood="wild",
            orientation="landscape",
        )
        image_tags = {
            "subject": "gray wolf",
            "category": "wildlife",
            "attributes": ["snow", "predator"],
            "caption": "A gray wolf hunting in winter.",
            "people_present": False,
            "confidence": 0.95,
        }

        result = guard.evaluate(brief, image_tags, retrieval_score=0.80)
        assert result.decision == "REJECT"
        assert any("confusable" in r.lower() for r in result.reasons)

    def test_rejects_must_not_have_violation(self, guard: MismatchGuard) -> None:
        brief = ArticleBrief(
            primary_subject="city park",
            must_have=["trees"],
            must_not_have=["people"],
            mood="quiet",
            orientation="any",
        )
        image_tags = {
            "subject": "city park",
            "category": "landscape",
            "attributes": ["trees", "benches", "crowd"],
            "caption": "A park with people walking.",
            "people_present": True,
            "confidence": 0.88,
        }

        result = guard.evaluate(brief, image_tags, retrieval_score=0.75)
        assert result.decision == "REJECT"
        assert any("people" in r.lower() or "must_not_have" in r.lower() for r in result.reasons)

    def test_abstention_on_weak_confidence(self, guard: MismatchGuard) -> None:
        brief = ArticleBrief(
            primary_subject="quantum entanglement",
            must_have=[],
            must_not_have=[],
            mood="abstract",
            orientation="any",
        )
        image_tags = {
            "subject": "abstract blur",
            "category": "abstract",
            "attributes": ["colors"],
            "caption": "Light rays blurred in motion.",
            "people_present": False,
            "confidence": 0.30,
        }

        result = guard.evaluate(brief, image_tags, retrieval_score=0.20)
        assert result.decision == "NO_CONFIDENT_MATCH"


class TestLLMJudgeAndKappa:
    """Test the LLM judge and Cohen's kappa agreement calculation."""

    def test_cohens_kappa_perfect_agreement(self) -> None:
        human = [2, 1, 0, 2, 0, 1]
        judge = [2, 1, 0, 2, 0, 1]
        kappa = compute_cohens_kappa(human, judge)
        assert kappa == 1.0

    def test_cohens_kappa_high_agreement(self) -> None:
        human = [2, 2, 1, 0, 0, 1, 2, 0, 1, 2]
        judge = [2, 2, 1, 0, 0, 1, 2, 1, 1, 2]  # 1 disagreement
        kappa = compute_cohens_kappa(human, judge)
        assert kappa > 0.70

    def test_judge_heuristic_grading(self) -> None:
        judge = LLMJudge()
        grade, explanation = judge.grade_pair(
            article_title="Red Fox",
            primary_subject="red fox",
            image_subject="Red Fox in Snow",
            image_category="wildlife",
            image_caption="A red fox",
        )
        assert grade == 2
        assert "match" in explanation.lower()


class TestBenchmarksAndMetrics:
    """Test IR metrics: DCG, nDCG@5, bootstrap confidence intervals, and evaluation runner."""

    def test_ndcg_at_k(self) -> None:
        perfect = [2, 2, 1, 0, 0]
        assert ndcg_at_k(perfect, k=5) == 1.0

        worse = [0, 0, 1, 2, 2]
        assert ndcg_at_k(worse, k=5) < 1.0

    def test_bootstrap_ci(self) -> None:
        scores = [0.8, 0.85, 0.78, 0.92, 0.88, 0.84, 0.90]
        low, high = bootstrap_ci(scores, n_bootstraps=200)
        assert low <= high
        assert 0.70 < low < 0.95

    def test_evaluate_run_metrics(self) -> None:
        qrels = {
            "q1": {"img1": 2, "img2": 1, "img3": 0},
            "q2": {"img4": 2, "img5": 0},
        }
        run = {
            "q1": ["img1", "img2", "img3"],
            "q2": ["img4", "img5"],
        }
        metrics = evaluate_run(qrels, run, k=5)
        assert metrics.ndcg_at_5 == 1.0
        assert metrics.recall_at_50 == 1.0
        assert metrics.top1_precision == 1.0
