"""Tests for SQLAlchemy ORM models."""

import uuid

from curio.models import (
    Article,
    CostLedger,
    Editor,
    EditorPreferenceRecord,
    Image,
    Job,
    Label,
    Suggestion,
)


class TestImageModel:
    """Test Image ORM model instantiation."""

    def test_create_image(self) -> None:
        img = Image(
            source_id="abc123",
            source="unsplash",
            url="https://images.unsplash.com/photo-abc123",
            vision_tagged=False,
            vision_flagged=False,
            is_deleted=False,
        )
        assert img.source_id == "abc123"
        assert img.source == "unsplash"
        assert img.vision_tagged is False
        assert img.vision_flagged is False
        assert img.is_deleted is False

    def test_image_defaults(self) -> None:
        img = Image(
            source_id="xyz",
            source="upload",
            url="https://example.com/img.jpg",
            vision_tagged=False,
            is_deleted=False,
        )
        assert img.vision_tagged is False
        assert img.is_deleted is False


class TestArticleModel:
    """Test Article ORM model instantiation."""

    def test_create_article(self) -> None:
        article = Article(
            source_id="wikipedia:12345",
            title="Red Fox",
            body="The red fox is the largest of the true foxes...",
            is_no_match=False,
        )
        assert article.title == "Red Fox"
        assert article.is_no_match is False

    def test_no_match_article(self) -> None:
        article = Article(
            source_id="wikipedia:99999",
            title="Quantum Entanglement",
            body="Quantum entanglement is a phenomenon...",
            is_no_match=True,
        )
        assert article.is_no_match is True


class TestEditorModel:
    """Test Editor ORM model."""

    def test_create_editor(self) -> None:
        editor = Editor(name="Alice", publication="GDG Blog")
        assert editor.name == "Alice"
        assert editor.publication == "GDG Blog"


class TestEditorPreferenceModel:
    """Test EditorPreferenceRecord ORM model."""

    def test_create_preference(self) -> None:
        pref = EditorPreferenceRecord(
            editor_id=uuid.uuid4(),
            preference="landscape orientation",
            polarity="positive",
            source="explicit",
            confidence=1.0,
            confirmed=False,
        )
        assert pref.polarity == "positive"
        assert pref.confidence == 1.0
        assert pref.confirmed is False


class TestSuggestionModel:
    """Test Suggestion ORM model."""

    def test_create_suggestion(self) -> None:
        suggestion = Suggestion(
            article_id=uuid.uuid4(),
            image_id=uuid.uuid4(),
            editor_id=uuid.uuid4(),
            retrieval_scores={"keyword": 0.8, "dense": 0.7, "clip": 0.9},
            guard_result="ACCEPT",
            status="pending",
        )
        assert suggestion.status == "pending"
        assert suggestion.guard_result == "ACCEPT"


class TestCostLedgerModel:
    """Test CostLedger ORM model."""

    def test_create_cost_entry(self) -> None:
        entry = CostLedger(
            model="gemini-2.0-flash",
            operation="vision_tag",
            input_tokens=500,
            output_tokens=200,
            estimated_cost=0.0,
        )
        assert entry.model == "gemini-2.0-flash"
        assert entry.estimated_cost == 0.0


class TestJobModel:
    """Test Job ORM model."""

    def test_create_job(self) -> None:
        job = Job(
            queue="vision_tag",
            payload={"image_id": str(uuid.uuid4())},
            idempotency_key="tag:abc123",
            status="pending",
            attempts=0,
            max_attempts=3,
        )
        assert job.status == "pending"
        assert job.attempts == 0
        assert job.max_attempts == 3


class TestLabelModel:
    """Test Label ORM model."""

    def test_create_label(self) -> None:
        label = Label(
            article_id=uuid.uuid4(),
            image_id=uuid.uuid4(),
            grade=2,
            labeler="human",
            split="dev",
        )
        assert label.grade == 2
        assert label.labeler == "human"
        assert label.split == "dev"
