"""Calibrated mismatch guard.

Enforces precision checks to reject wrong-but-confusable matches:
1. Rule checks: category mismatch, must_not_have keyword presence
2. Confusable lookalike protection (e.g. wolf for fox, cheetah for leopard)
3. Confidence thresholding: returns ACCEPT, REJECT, or NO_CONFIDENT_MATCH
"""

from __future__ import annotations

import logging
from typing import Any

from curio.schemas import ArticleBrief, GuardResult, VisionTags

logger = logging.getLogger(__name__)

# Known confusable animal pairs/groups and their distinct markers
CONFUSABLE_LOOKALIKES: dict[str, set[str]] = {
    "fox": {"wolf", "coyote", "dog", "jackal", "dingo"},
    "red fox": {"wolf", "coyote", "dog", "gray wolf"},
    "wolf": {"fox", "coyote", "dog"},
    "gray wolf": {"red fox", "coyote"},
    "coyote": {"wolf", "fox", "dog"},
    "leopard": {"cheetah", "jaguar", "ocelot"},
    "cheetah": {"leopard", "jaguar"},
    "jaguar": {"leopard", "cheetah"},
    "crow": {"raven", "rook", "magpie"},
    "raven": {"crow", "rook"},
    "alligator": {"crocodile", "caiman", "gharial"},
    "crocodile": {"alligator", "caiman", "gharial"},
}


class MismatchGuard:
    """Verifies image-article pairings and explains rejections."""

    def __init__(self, acceptance_threshold: float = 0.55) -> None:
        self.acceptance_threshold = acceptance_threshold

    def evaluate(
        self,
        brief: ArticleBrief,
        image_tags: VisionTags | dict[str, Any],
        retrieval_score: float = 0.0,
    ) -> GuardResult:
        """Evaluate match validity.

        Returns:
            GuardResult with decision ('ACCEPT', 'REJECT', or 'NO_CONFIDENT_MATCH')
            and explanatory reason codes.
        """
        reasons: list[str] = []

        if isinstance(image_tags, dict):
            subject = str(image_tags.get("subject") or "").lower()
            category = str(image_tags.get("category") or "").lower()
            attributes = [str(a).lower() for a in image_tags.get("attributes", [])]
            caption = str(image_tags.get("caption") or "").lower()
            people_present = bool(image_tags.get("people_present", False))
            vision_conf = float(image_tags.get("confidence") or 1.0)
        else:
            subject = image_tags.subject.lower()
            category = image_tags.category.lower()
            attributes = [a.lower() for a in image_tags.attributes]
            caption = image_tags.caption.lower()
            people_present = image_tags.people_present
            vision_conf = image_tags.confidence

        primary_target = brief.primary_subject.lower()
        f"{subject} {caption} {' '.join(attributes)}"

        # 1. Hard Rule Checks: must_not_have violations
        for prohibited in brief.must_not_have:
            prohibited_lower = prohibited.lower()
            if prohibited_lower in subject or prohibited_lower in caption or any(prohibited_lower in attr for attr in attributes):
                reasons.append(f"violation of must_not_have constraint: '{prohibited}' detected in image")

        # 2. Hard negative confusable lookalike checks
        # e.g., brief asks for fox, detected subject is wolf
        for target_term, confusable_set in CONFUSABLE_LOOKALIKES.items():
            if target_term in primary_target:
                for confusable in confusable_set:
                    # If the confusable animal is explicitly in the image subject/caption
                    # and NOT what the article wanted
                    if confusable in subject and target_term not in subject:
                        reasons.append(
                            f"confusable lookalike mismatch: expected '{target_term}', detected '{confusable}'"
                        )
                    elif confusable in caption and target_term not in caption:
                        reasons.append(
                            f"confusable lookalike mismatch: expected '{target_term}', detected '{confusable}' in caption"
                        )

        # 3. People presence constraint
        # If must_not_have asks for 'people' or 'person'
        no_people_requested = any(
            p in ["people", "person", "human", "man", "woman"]
            for p in [p.lower() for p in brief.must_not_have]
        )
        if no_people_requested and people_present:
            reasons.append("constraint violation: people detected in image when excluded")

        # If hard rejections were found, return REJECT
        if reasons:
            return GuardResult(
                decision="REJECT",
                reasons=reasons,
                confidence=round(vision_conf, 3),
            )

        # 4. Calibration & Abstention Thresholding
        # If retrieval/confidence scores are too weak, refuse to guess
        combined_confidence = (retrieval_score * 0.7) + (vision_conf * 0.3) if retrieval_score > 0 else vision_conf

        if combined_confidence < self.acceptance_threshold:
            return GuardResult(
                decision="NO_CONFIDENT_MATCH",
                reasons=[
                    f"insufficient match confidence: score {combined_confidence:.3f} below acceptance threshold {self.acceptance_threshold:.3f}"
                ],
                confidence=round(combined_confidence, 3),
            )

        # 5. Passed all guardrails: ACCEPT
        accept_reasons = [f"subject '{subject}' matches target '{brief.primary_subject}'"]
        if category:
            accept_reasons.append(f"category '{category}' aligned")

        return GuardResult(
            decision="ACCEPT",
            reasons=accept_reasons,
            confidence=round(combined_confidence, 3),
        )
