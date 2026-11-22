"""Simulated editor personalisation evaluation harness.

Simulates content editors with distinct preferences (e.g. landscape only, no people, muted mood).
Tracks nDCG@5 before and after editor feedback events to measure personalisation lift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from curio.services.benchmarks import ndcg_at_k
from curio.services.features import EditorPreferenceProfile

logger = logging.getLogger(__name__)


@dataclass
class SimulatedEditor:
    """An editor with a distinct preference profile."""

    editor_id: UUID
    name: str
    profile: EditorPreferenceProfile
    feedback_history: list[dict[str, Any]]


def create_simulated_editors() -> list[SimulatedEditor]:
    """Generate canonical simulated editors for evaluation."""
    return [
        SimulatedEditor(
            editor_id=uuid4(),
            name="Editor Landscape-Only",
            profile=EditorPreferenceProfile(
                editor_id=uuid4(),
                positive_tags={"landscape"},
                negative_tags={"portrait"},
                preferred_orientation="landscape",
                allow_people=True,
            ),
            feedback_history=[],
        ),
        SimulatedEditor(
            editor_id=uuid4(),
            name="Editor No-People",
            profile=EditorPreferenceProfile(
                editor_id=uuid4(),
                positive_tags={"nature", "wildlife", "architecture"},
                negative_tags={"people", "person", "crowd"},
                preferred_orientation=None,
                allow_people=False,
            ),
            feedback_history=[],
        ),
        SimulatedEditor(
            editor_id=uuid4(),
            name="Editor Wildlife-Focus",
            profile=EditorPreferenceProfile(
                editor_id=uuid4(),
                positive_tags={"wildlife", "animal"},
                negative_tags={"technology", "food"},
                preferred_orientation=None,
                allow_people=True,
            ),
            feedback_history=[],
        ),
    ]


def evaluate_personalisation_lift(
    base_rankings: list[int],
    personalised_rankings: list[int],
    k: int = 5,
) -> tuple[float, float, float]:
    """Calculate nDCG@k before, after, and the percentage lift.

    Returns:
        (ndcg_before, ndcg_after, lift_points)
    """
    ndcg_before = ndcg_at_k(base_rankings, k=k)
    ndcg_after = ndcg_at_k(personalised_rankings, k=k)
    lift = (ndcg_after - ndcg_before) * 100.0
    return round(ndcg_before, 4), round(ndcg_after, 4), round(lift, 2)
