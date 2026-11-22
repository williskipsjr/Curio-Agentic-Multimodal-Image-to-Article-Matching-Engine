"""Feature extraction pipeline for Stage 2 re-ranking and editor personalisation.

Extracts tabular feature vectors from retrieved candidates for LightGBM LambdaMART:
1. Retriever scores & ranks (keyword, dense, CLIP, RRF)
2. Lexical & semantic overlap between structured article brief and image tags
3. Category & orientation compatibility
4. Vision model confidence score
5. Editor personalisation features (preference alignment & hard-filter flags)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from uuid import UUID

from curio.schemas import ArticleBrief
from curio.services.retrieval import RetrievalCandidate


@dataclass
class EditorPreferenceProfile:
    """Active preferences for a specific editor."""

    editor_id: UUID
    positive_tags: set[str]  # e.g., {'landscape', 'muted colors'}
    negative_tags: set[str]  # e.g., {'people', 'bright'}
    preferred_orientation: str | None = None
    allow_people: bool = True


class FeatureExtractor:
    """Transforms (ArticleBrief, RetrievalCandidate, EditorPreferenceProfile) into numeric features."""

    FEATURE_NAMES: ClassVar[list[str]] = [
        # Channel Scores & Ranks
        "keyword_score",
        "keyword_rank",
        "dense_score",
        "dense_rank",
        "clip_score",
        "clip_rank",
        "rrf_score",
        # Alignment Features
        "subject_exact_match",
        "subject_token_overlap",
        "category_match",
        "attribute_overlap_count",
        "orientation_match",
        "vision_confidence",
        "people_penalty",
        # Personalisation Features
        "editor_pref_positive_overlap",
        "editor_pref_negative_overlap",
        "editor_orientation_match",
    ]

    @classmethod
    def extract_features(
        cls,
        brief: ArticleBrief,
        candidate: RetrievalCandidate,
        editor_profile: EditorPreferenceProfile | None = None,
    ) -> list[float]:
        """Extract a 1D vector of float features."""
        # 1. Channel Scores & Ranks (impute defaults for missing channels)
        kw_score = float(candidate.channel_scores.get("keyword", 0.0))
        kw_rank = float(candidate.channel_ranks.get("keyword", 100.0))
        dense_score = float(candidate.channel_scores.get("dense", 0.0))
        dense_rank = float(candidate.channel_ranks.get("dense", 100.0))
        clip_score = float(candidate.channel_scores.get("clip", 0.0))
        clip_rank = float(candidate.channel_ranks.get("clip", 100.0))
        rrf_score = float(candidate.rrf_score)

        # 2. Alignment features
        target_subject = brief.primary_subject.lower().strip()
        cand_subject = (candidate.subject or "").lower().strip()
        cand_category = (candidate.category or "").lower().strip()
        cand_caption = (candidate.caption or "").lower().strip()

        subject_exact = 1.0 if target_subject == cand_subject else 0.0

        target_tokens = set(target_subject.split())
        cand_tokens = set(cand_subject.split()).union(set(cand_caption.split()))
        overlap_tokens = len(target_tokens.intersection(cand_tokens))
        subject_token_overlap = float(overlap_tokens) / max(len(target_tokens), 1)

        category_match = 1.0 if any(t in cand_category for t in target_tokens) else 0.0

        # Must have overlap count
        must_haves = [m.lower() for m in brief.must_have]
        attr_overlap_count = float(
            sum(1 for m in must_haves if m in cand_caption or m in cand_subject)
        )

        # Orientation
        orientation_match = 1.0
        if brief.orientation != "any" and candidate.orientation:
            orientation_match = 1.0 if brief.orientation == candidate.orientation else 0.0

        vision_confidence = float(candidate.vision_confidence or 1.0)

        # Must not have violation penalty
        must_not_haves = [m.lower() for m in brief.must_not_have]
        people_penalty = 0.0
        if (
            any("people" in m or "person" in m for m in must_not_haves)
            and candidate.people_present
        ):
            people_penalty = 1.0

        # 3. Personalisation features
        pos_overlap = 0.0
        neg_overlap = 0.0
        editor_orient_match = 1.0

        if editor_profile:
            cand_features_set = cand_tokens.union(
                {cand_category, candidate.orientation or ""}
            )

            # Positive preference overlap
            pos_overlap = float(
                len(editor_profile.positive_tags.intersection(cand_features_set))
            )

            # Negative preference overlap
            neg_overlap = float(
                len(editor_profile.negative_tags.intersection(cand_features_set))
            )
            if not editor_profile.allow_people and candidate.people_present:
                neg_overlap += 1.0

            if editor_profile.preferred_orientation and candidate.orientation:
                editor_orient_match = (
                    1.0
                    if editor_profile.preferred_orientation == candidate.orientation
                    else 0.0
                )

        return [
            kw_score,
            kw_rank,
            dense_score,
            dense_rank,
            clip_score,
            clip_rank,
            rrf_score,
            subject_exact,
            subject_token_overlap,
            category_match,
            attr_overlap_count,
            orientation_match,
            vision_confidence,
            people_penalty,
            pos_overlap,
            neg_overlap,
            editor_orient_match,
        ]

    @classmethod
    def apply_editor_hard_filters(
        cls,
        candidates: list[RetrievalCandidate],
        profile: EditorPreferenceProfile | None,
    ) -> list[RetrievalCandidate]:
        """Filter candidates that violate strict editor preferences (e.g., 'no people')."""
        if not profile:
            return candidates

        filtered: list[RetrievalCandidate] = []
        for c in candidates:
            if not profile.allow_people and c.people_present:
                continue
            filtered.append(c)
        return filtered
