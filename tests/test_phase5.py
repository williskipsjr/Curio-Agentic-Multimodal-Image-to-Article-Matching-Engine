"""Tests for Phase 5: Curator Agent and Memory."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Article, Editor, Image
from curio.services.agent import CuratorAgent, SessionState, verify_grounding
from curio.services.agent_eval import generate_60_tasks, run_smoke_eval
from curio.services.agent_tools import (
    CheckMatchInput,
    CuratorToolService,
    GenerateFallbackImageInput,
    GetArticleInput,
    ProposePairingInput,
    SavePreferenceInput,
    SearchImagesInput,
    ToolGuardrailError,
)


@pytest_asyncio.fixture
async def sample_editor(db_session: AsyncSession) -> Editor:
    """Create a sample editor in the database."""
    editor = Editor(name="Alice Editor", publication="GDG Weekly")
    db_session.add(editor)
    await db_session.flush()
    return editor


@pytest_asyncio.fixture
async def sample_article(db_session: AsyncSession) -> Article:
    """Create a sample article in the database."""
    article = Article(
        title="Red Fox Ecology",
        body="The red fox is the largest of the true foxes and one of the most widely distributed.",
        primary_subject="Red fox",
        must_have=["fox", "wildlife", "fur"],
        must_not_have=["wolf", "dog"],
        mood="natural",
        orientation="landscape",
        source_id=f"fox_{uuid.uuid4().hex[:6]}",
    )
    db_session.add(article)
    await db_session.flush()
    return article


@pytest_asyncio.fixture
async def sample_image(db_session: AsyncSession) -> Image:
    """Create a sample matching image in the database."""
    image = Image(
        url="https://images.example.com/fox1.jpg",
        content_hash=uuid.uuid4().hex,
        source="unsplash",
        source_id=f"fox_img_{uuid.uuid4().hex[:6]}",
        subject="Red fox",
        category="wildlife",
        attributes=["fox", "orange fur", "snow"],
        caption="A red fox standing in white snow",
        people_present=False,
        orientation="landscape",
        vision_confidence=0.95,
        vision_tagged=True,
    )
    db_session.add(image)
    await db_session.flush()
    return image


class TestAgentToolsAndGuardrails:
    """Test suite for agent tools and strict server-side guardrails."""

    @pytest.mark.asyncio
    async def test_search_images_k_cap(self, db_session: AsyncSession):
        """Guardrail: Search candidates k is strictly capped at 50."""
        tool = CuratorToolService(db_session)
        # Verify schema validator caps at 50
        inp = SearchImagesInput(query="fox", k=100)
        assert inp.k == 50

        # Execute
        out = await tool.search_images(inp)
        assert isinstance(out.candidates, list)

    @pytest.mark.asyncio
    async def test_get_article_tool(self, db_session: AsyncSession, sample_article: Article):
        """Tool 1: get_article returns article and structured brief."""
        tool = CuratorToolService(db_session)
        out = await tool.get_article(GetArticleInput(article_id=sample_article.id))
        assert out.article_id == sample_article.id
        assert out.title == "Red Fox Ecology"
        assert out.brief.primary_subject == "Red fox"

    @pytest.mark.asyncio
    async def test_check_match_tool(
        self, db_session: AsyncSession, sample_article: Article, sample_image: Image
    ):
        """Tool 4: check_match evaluates mismatch guard."""
        tool = CuratorToolService(db_session)
        out = await tool.check_match(
            CheckMatchInput(article_id=sample_article.id, image_id=sample_image.id)
        )
        assert out.decision in ("ACCEPT", "REJECT", "NO_CONFIDENT_MATCH")
        assert out.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_propose_pairing_guardrail_refusal(
        self, db_session: AsyncSession, sample_article: Article, sample_image: Image, sample_editor: Editor
    ):
        """Guardrail: propose_pairing is refused unless check_match returned ACCEPT."""
        tool = CuratorToolService(db_session)

        # Attempt to propose without prior check_match in checked_pairs
        with pytest.raises(ToolGuardrailError, match="Guardrail refusal: Cannot propose pairing"):
            await tool.propose_pairing(
                ProposePairingInput(
                    article_id=sample_article.id,
                    image_id=sample_image.id,
                    reasons=["Looks nice"],
                    confidence=0.9,
                ),
                checked_pairs={},  # Empty -> not checked!
                editor_id=sample_editor.id,
            )

        # Attempt to propose when check_match was REJECT
        with pytest.raises(ToolGuardrailError, match="Guardrail refusal"):
            await tool.propose_pairing(
                ProposePairingInput(
                    article_id=sample_article.id,
                    image_id=sample_image.id,
                    reasons=["Looks nice"],
                    confidence=0.9,
                ),
                checked_pairs={(sample_article.id, sample_image.id): "REJECT"},
                editor_id=sample_editor.id,
            )

        # Propose succeeds when check_match was ACCEPT
        out = await tool.propose_pairing(
            ProposePairingInput(
                article_id=sample_article.id,
                image_id=sample_image.id,
                reasons=["Exact species match", "Grounded tags"],
                confidence=0.92,
            ),
            checked_pairs={(sample_article.id, sample_image.id): "ACCEPT"},
            editor_id=sample_editor.id,
        )
        assert out.status == "pending"
        assert out.suggestion_id is not None

    @pytest.mark.asyncio
    async def test_save_preference_inferred_confirmation_guardrail(
        self, db_session: AsyncSession, sample_editor: Editor
    ):
        """Guardrail: Inferred preferences require editor confirmation."""
        tool = CuratorToolService(db_session)

        # Inferred + unconfirmed -> must raise guardrail error
        with pytest.raises(ToolGuardrailError, match="Inferred preferences require editor confirmation"):
            await tool.save_preference(
                SavePreferenceInput(
                    editor_id=sample_editor.id,
                    preference="Landscape only",
                    polarity="positive",
                    source="inferred",
                    confirmed=False,
                )
            )

        # Inferred + confirmed -> succeeds
        out = await tool.save_preference(
            SavePreferenceInput(
                editor_id=sample_editor.id,
                preference="Landscape only",
                polarity="positive",
                source="inferred",
                confirmed=True,
            )
        )
        assert out.status == "saved"

    @pytest.mark.asyncio
    async def test_fallback_image_guardrail(self, db_session: AsyncSession, sample_article: Article):
        """Guardrail: generate_fallback_image can only run after NO_CONFIDENT_MATCH."""
        tool = CuratorToolService(db_session)

        # Refuses if last decision was ACCEPT
        with pytest.raises(ToolGuardrailError, match="can only be invoked after check_match returns NO_CONFIDENT_MATCH"):
            await tool.generate_fallback_image(
                GenerateFallbackImageInput(prompt="A cute fox", article_id=sample_article.id),
                last_guard_decision="ACCEPT",
            )

        # Succeeds if last decision was NO_CONFIDENT_MATCH
        out = await tool.generate_fallback_image(
            GenerateFallbackImageInput(prompt="A cute fox", article_id=sample_article.id),
            last_guard_decision="NO_CONFIDENT_MATCH",
        )
        assert out.status == "generated"


class TestGroundingChecker:
    """Test suite for grounding check on match explanations."""

    def test_verify_grounding_clean(self):
        """Grounded reasons cite only stored attributes and standard terms."""
        image_tags = {
            "subject": "Red fox",
            "attributes": ["orange fur", "snow"],
            "category": "wildlife",
        }
        article_brief = {
            "primary_subject": "Red fox",
            "must_have": ["wildlife"],
        }
        reasons = [
            "Image subject 'Red fox' matches article primary subject",
            "Attributes aligned: orange fur, snow",
        ]
        grounded, all_ok = verify_grounding(reasons, image_tags, article_brief)
        assert all_ok is True
        assert len(grounded) == 2

    def test_verify_grounding_flags_hallucinations(self):
        """Ungrounded reasons with external hallucinated entities are flagged."""
        image_tags = {
            "subject": "Red fox",
            "attributes": ["snow"],
        }
        article_brief = {
            "primary_subject": "Red fox",
        }
        # Cites arbitrary unmentioned objects
        reasons = [
            "A majestic golden eagle soaring near a volcanic eruption with helicopters",
        ]
        grounded, all_ok = verify_grounding(reasons, image_tags, article_brief)
        assert all_ok is False
        assert len(grounded) == 0


class TestCuratorAgentLoop:
    """Test suite for agent loop, step cap, and session memory."""

    @pytest.mark.asyncio
    async def test_agent_step_cap(
        self, db_session: AsyncSession, sample_article: Article, sample_editor: Editor
    ):
        """Agent execution is strictly capped at 8 steps."""
        agent = CuratorAgent(db_session)
        state = SessionState()
        state.step_count = 8  # Already reached cap

        resp = await agent.run_task(
            article_id=sample_article.id,
            editor_id=sample_editor.id,
            session_state=state,
        )
        assert resp.image_id is None
        assert resp.guard_result.decision == "NO_CONFIDENT_MATCH"
        assert "Step limit reached" in resp.reasons[0]


class TestAgentEvaluationHarness:
    """Test suite for the 60-task eval suite and 15-task CI smoke eval."""

    def test_generate_60_tasks_structure(self):
        """Ensure full benchmark suite generates exactly 60 tasks with canonical splits."""
        tasks = generate_60_tasks()
        assert len(tasks) == 60

        task_types = [t.task_type for t in tasks]
        assert task_types.count("direct_match") == 20
        assert task_types.count("confusable_negative") == 15
        assert task_types.count("no_match") == 10
        assert task_types.count("editor_constraint") == 10
        assert task_types.count("inferred_pref") == 5

    @pytest.mark.asyncio
    async def test_smoke_eval_runs_in_ci(self, db_session: AsyncSession):
        """Run 15-task smoke eval to verify task success and tool-call validity."""
        metrics = await run_smoke_eval(db_session)
        assert metrics.total_tasks == 15
        assert metrics.task_success_rate >= 0.70
        assert metrics.tool_call_validity_rate >= 0.90
        assert metrics.total_tool_calls > 0
