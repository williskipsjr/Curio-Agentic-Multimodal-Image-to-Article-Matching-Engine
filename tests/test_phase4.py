"""Tests for Phase 4: Feature extraction, LightGBM LambdaMART, Gemini listwise re-ranking, and personalisation."""

from uuid import uuid4

import numpy as np

from curio.schemas import ArticleBrief
from curio.services.features import (
    EditorPreferenceProfile,
    FeatureExtractor,
)
from curio.services.llm_reranker import (
    GeminiListwiseReranker,
    ReRankerComparisonHarness,
)
from curio.services.personalisation_eval import (
    create_simulated_editors,
    evaluate_personalisation_lift,
)
from curio.services.reranker import LightGBMReranker
from curio.services.retrieval import RetrievalCandidate


def make_candidate(
    image_id: str | None = None,
    subject: str = "red fox",
    orientation: str = "landscape",
    people_present: bool = False,
    rrf_score: float = 0.05,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        image_id=uuid4(),
        source_id=image_id or "img_1",
        url="https://example.com/img.jpg",
        subject=subject,
        category="wildlife",
        caption=f"A photo of {subject}",
        orientation=orientation,
        people_present=people_present,
        vision_confidence=0.92,
        channel_scores={"keyword": 5.0, "dense": 0.85, "clip": 0.90},
        channel_ranks={"keyword": 1, "dense": 2, "clip": 1},
        rrf_score=rrf_score,
    )


class TestFeatureExtraction:
    """Test tabular feature extraction for LambdaMART."""

    def test_feature_vector_dimension_and_values(self) -> None:
        brief = ArticleBrief(
            primary_subject="red fox",
            must_have=["fox", "snow"],
            must_not_have=["people"],
            mood="wild",
            orientation="landscape",
        )
        cand = make_candidate(subject="red fox", orientation="landscape", people_present=False)

        profile = EditorPreferenceProfile(
            editor_id=uuid4(),
            positive_tags={"landscape", "wildlife"},
            negative_tags={"people"},
            preferred_orientation="landscape",
            allow_people=False,
        )

        features = FeatureExtractor.extract_features(brief, cand, profile)
        assert len(features) == len(FeatureExtractor.FEATURE_NAMES)
        # Check alignment: subject exact match is 1.0, orientation match is 1.0
        assert features[FeatureExtractor.FEATURE_NAMES.index("subject_exact_match")] == 1.0
        assert features[FeatureExtractor.FEATURE_NAMES.index("orientation_match")] == 1.0

    def test_editor_hard_filter(self) -> None:
        c1 = make_candidate(subject="fox", people_present=False)
        c2 = make_candidate(subject="person with fox", people_present=True)

        profile = EditorPreferenceProfile(
            editor_id=uuid4(),
            positive_tags=set(),
            negative_tags=set(),
            allow_people=False,
        )

        filtered = FeatureExtractor.apply_editor_hard_filters([c1, c2], profile)
        assert len(filtered) == 1
        assert filtered[0].people_present is False


class TestLightGBMReranker:
    """Test LambdaMART model training, scoring, and isotonic calibration."""

    def test_train_and_calibrate(self) -> None:
        reranker = LightGBMReranker()

        # Generate synthetic ranking dataset (3 queries, 5 docs each)
        n_queries = 5
        docs_per_query = 6
        n_samples = n_queries * docs_per_query
        n_features = len(FeatureExtractor.FEATURE_NAMES)

        X = np.random.rand(n_samples, n_features)
        # Give higher relevance to higher feature 0 values
        y = np.clip((X[:, 0] * 3).astype(int), 0, 2)
        groups = [docs_per_query] * n_queries

        reranker.train(X, y, groups)
        assert reranker.model is not None
        assert reranker.calibrator is not None

        # Test candidate scoring
        brief = ArticleBrief(
            primary_subject="red fox",
            must_have=[],
            must_not_have=[],
            mood="wild",
            orientation="any",
        )
        cands = [make_candidate(rrf_score=0.01 * i) for i in range(5)]
        scored = reranker.score_candidates(brief, cands)

        assert len(scored) == 5
        # Verify calibrated probability is bounded between 0 and 1
        for _, _raw_score, prob in scored:
            assert 0.0 <= prob <= 1.0


class TestGeminiAndComparison:
    """Test Gemini listwise re-ranking and comparison harness."""

    def test_comparison_harness(self) -> None:
        brief = ArticleBrief(
            primary_subject="mountain",
            must_have=[],
            must_not_have=[],
            mood="serene",
            orientation="any",
        )
        cands = [make_candidate(subject="mountain peak") for _ in range(4)]
        lgb_reranker = LightGBMReranker()
        gemini_reranker = GeminiListwiseReranker()

        lgb_res, gemini_res = ReRankerComparisonHarness.compare(
            brief, cands, lgb_reranker, gemini_reranker
        )

        assert lgb_res.model_name == "LightGBM LambdaMART"
        assert lgb_res.cost_per_1000_queries_usd == 0.0
        assert gemini_res.model_name == "Gemini 2.0 Flash Listwise"


class TestPersonalisationEval:
    """Test simulated editors and personalisation lift calculations."""

    def test_simulated_editors(self) -> None:
        editors = create_simulated_editors()
        assert len(editors) >= 3
        assert any(e.profile.allow_people is False for e in editors)

    def test_personalisation_lift_calculation(self) -> None:
        base = [0, 1, 0, 2, 0]
        personalised = [2, 1, 0, 0, 0]

        before, after, lift = evaluate_personalisation_lift(base, personalised, k=5)
        assert after > before
        assert lift > 0.0
