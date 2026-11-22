"""LightGBM LambdaMART learning-to-rank model with isotonic score calibration.

Stage 2 Re-ranker:
- Trains on query groups using lambdarank objective
- Evaluates pairwise/listwise relevance to optimize nDCG@5
- Applies IsotonicRegression calibration to transform raw ranking scores into calibrated probabilities
- Saves and loads trained models for inference
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

from curio.schemas import ArticleBrief
from curio.services.features import (
    EditorPreferenceProfile,
    FeatureExtractor,
)
from curio.services.retrieval import RetrievalCandidate

logger = logging.getLogger(__name__)


class LightGBMReranker:
    """LambdaMART re-ranking model with calibrated score output."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model: lgb.Booster | None = None
        self.calibrator: IsotonicRegression | None = None
        if model_path and model_path.exists():
            self.load(model_path)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: list[int],
        eval_X: np.ndarray | None = None,
        eval_y: np.ndarray | None = None,
        eval_groups: list[int] | None = None,
    ) -> None:
        """Train LightGBM with lambdarank objective."""
        train_data = lgb.Dataset(
            X,
            label=y,
            group=groups,
            feature_name=FeatureExtractor.FEATURE_NAMES,
            free_raw_data=False,
        )

        params: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": [5],
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "verbosity": -1,
            "seed": 42,
        }

        valid_sets = [train_data]
        if eval_X is not None and eval_y is not None and eval_groups is not None:
            val_data = lgb.Dataset(
                eval_X,
                label=eval_y,
                group=eval_groups,
                reference=train_data,
                feature_name=FeatureExtractor.FEATURE_NAMES,
                free_raw_data=False,
            )
            valid_sets.append(val_data)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=valid_sets,
        )

        # Fit isotonic regression calibrator on train predictions vs binary relevance (y >= 1)
        raw_scores = self.model.predict(X)
        binary_labels = (y >= 1).astype(int)
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.calibrator.fit(raw_scores, binary_labels)
        logger.info("LightGBM LambdaMART model trained and calibrated successfully.")

    def score_candidates(
        self,
        brief: ArticleBrief,
        candidates: list[RetrievalCandidate],
        editor_profile: EditorPreferenceProfile | None = None,
    ) -> list[tuple[RetrievalCandidate, float, float]]:
        """Score and calibrate a list of candidates.

        Returns:
            List of (candidate, raw_score, calibrated_prob) sorted descending by score.
        """
        if not candidates:
            return []

        # Extract features for all candidates
        feature_matrix = np.array(
            [
                FeatureExtractor.extract_features(brief, c, editor_profile)
                for c in candidates
            ]
        )

        if self.model is None:
            # Fallback heuristic: weighted combination of RRF + alignment
            raw_scores = np.array([c.rrf_score for c in candidates])
            calibrated_probs = np.clip(raw_scores * 10.0, 0.0, 1.0)
        else:
            raw_scores = self.model.predict(feature_matrix)
            if self.calibrator is not None:
                calibrated_probs = self.calibrator.predict(raw_scores)
            else:
                calibrated_probs = 1.0 / (1.0 + np.exp(-raw_scores))

        scored = list(zip(candidates, raw_scores, calibrated_probs, strict=True))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def save(self, directory: Path) -> None:
        """Save model booster and parameters."""
        directory.mkdir(parents=True, exist_ok=True)
        if self.model:
            self.model.save_model(str(directory / "lambdarank_model.txt"))
        logger.info("Saved LambdaMART model to %s", directory)

    def load(self, directory: Path) -> None:
        """Load model booster."""
        model_file = directory / "lambdarank_model.txt"
        if model_file.exists():
            self.model = lgb.Booster(model_file=str(model_file))
            logger.info("Loaded LambdaMART model from %s", model_file)
