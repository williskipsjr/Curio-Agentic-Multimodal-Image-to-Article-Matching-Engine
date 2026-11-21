"""Tests for core Pydantic schemas."""

import pytest
from pydantic import ValidationError

from curio.schemas import (
    ArticleBrief,
    EditorPreference,
    GuardResult,
    VisionTags,
)


class TestVisionTags:
    """Test VisionTags schema validation."""

    def test_valid_vision_tags(self) -> None:
        tags = VisionTags(
            subject="red fox",
            category="wildlife",
            attributes=["furry", "orange", "standing"],
            caption="A red fox standing in a snowy field",
            people_present=False,
            orientation="landscape",
            confidence=0.95,
        )
        assert tags.subject == "red fox"
        assert tags.confidence == 0.95

    def test_confidence_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            VisionTags(
                subject="fox",
                category="wildlife",
                attributes=[],
                caption="A fox",
                people_present=False,
                orientation="landscape",
                confidence=1.5,
            )

    def test_invalid_orientation(self) -> None:
        with pytest.raises(ValidationError):
            VisionTags(
                subject="fox",
                category="wildlife",
                attributes=[],
                caption="A fox",
                people_present=False,
                orientation="panoramic",  # type: ignore[arg-type]
                confidence=0.9,
            )


class TestArticleBrief:
    """Test ArticleBrief schema validation."""

    def test_valid_brief(self) -> None:
        brief = ArticleBrief(
            primary_subject="red fox",
            must_have=["fox", "wildlife"],
            must_not_have=["wolf", "dog"],
            mood="natural",
            orientation="landscape",
        )
        assert brief.primary_subject == "red fox"
        assert "wolf" in brief.must_not_have

    def test_any_orientation(self) -> None:
        brief = ArticleBrief(
            primary_subject="architecture",
            must_have=["building"],
            must_not_have=[],
            mood="modern",
            orientation="any",
        )
        assert brief.orientation == "any"


class TestGuardResult:
    """Test GuardResult schema validation."""

    def test_accept(self) -> None:
        result = GuardResult(
            decision="ACCEPT",
            reasons=["subject match", "category match"],
            confidence=0.92,
        )
        assert result.decision == "ACCEPT"

    def test_reject_with_reasons(self) -> None:
        result = GuardResult(
            decision="REJECT",
            reasons=["category mismatch: expected fox, detected wolf"],
            confidence=0.88,
        )
        assert result.decision == "REJECT"
        assert len(result.reasons) == 1

    def test_no_confident_match(self) -> None:
        result = GuardResult(
            decision="NO_CONFIDENT_MATCH",
            reasons=["no image above confidence threshold"],
            confidence=0.3,
        )
        assert result.decision == "NO_CONFIDENT_MATCH"


class TestEditorPreference:
    """Test EditorPreference schema validation."""

    def test_explicit_preference(self) -> None:
        pref = EditorPreference(
            preference="landscape orientation",
            polarity="positive",
            source="explicit",
        )
        assert pref.confidence == 1.0

    def test_inferred_preference(self) -> None:
        pref = EditorPreference(
            preference="no people",
            polarity="negative",
            source="inferred",
            confidence=0.75,
        )
        assert pref.source == "inferred"
        assert pref.confidence == 0.75
