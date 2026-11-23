"""Tests for Phase 6: Production FastAPI Endpoints and Background Job Queue."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from curio.config import settings
from curio.main import app, get_db
from curio.models import Article, Editor, Image, Job, Suggestion
from curio.services.queue import JobWorker, QueueService


class TestJobQueueService:
    """Test PostgreSQL-backed work queue with SKIP LOCKED and retry mechanics."""

    @pytest.mark.asyncio
    async def test_enqueue_and_idempotency(self, db_session: AsyncSession) -> None:
        key = f"test_idem_{uuid.uuid4().hex}"
        payload = {"data": "test_payload"}

        # 1. Enqueue job
        job1 = await QueueService.enqueue(
            session=db_session,
            queue="test_queue",
            payload=payload,
            idempotency_key=key,
            max_attempts=3,
        )
        assert job1.status == "pending"
        assert job1.idempotency_key == key

        # 2. Re-enqueue with same key returns existing job
        job2 = await QueueService.enqueue(
            session=db_session,
            queue="test_queue",
            payload=payload,
            idempotency_key=key,
        )
        assert job2.id == job1.id

    @pytest.mark.asyncio
    async def test_dequeue_and_complete(self, db_session: AsyncSession) -> None:
        q_name = f"queue_{uuid.uuid4().hex[:6]}"
        job = await QueueService.enqueue(
            session=db_session,
            queue=q_name,
            payload={"task": 1},
        )

        # Dequeue
        dequeued = await QueueService.dequeue(session=db_session, queue=q_name)
        assert dequeued is not None
        assert dequeued.id == job.id
        assert dequeued.status == "processing"
        assert dequeued.attempts == 1
        assert dequeued.locked_at is not None

        # Complete
        completed = await QueueService.complete(session=db_session, job_id=job.id)
        assert completed.status == "completed"
        assert completed.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_retry_and_dead_letter(self, db_session: AsyncSession) -> None:
        q_name = f"queue_{uuid.uuid4().hex[:6]}"
        job = await QueueService.enqueue(
            session=db_session,
            queue=q_name,
            payload={"fail": True},
            max_attempts=2,
        )

        # First failure: should re-queue as pending
        dequeued1 = await QueueService.dequeue(session=db_session, queue=q_name)
        assert dequeued1 is not None
        job_retry = await QueueService.fail(
            session=db_session, job_id=job.id, error="Transient error 1"
        )
        assert job_retry.status == "pending"
        assert job_retry.attempts == 1

        # Second failure: exceeds max_attempts (2) -> dead letter queue ("failed")
        dequeued2 = await QueueService.dequeue(session=db_session, queue=q_name)
        assert dequeued2 is not None
        job_failed = await QueueService.fail(
            session=db_session, job_id=job.id, error="Permanent error 2"
        )
        assert job_failed.status == "failed"
        assert job_failed.attempts == 2
        assert "Permanent error 2" in (job_failed.last_error or "")

    @pytest.mark.asyncio
    async def test_job_worker_execution(self, db_session: AsyncSession) -> None:
        q_name = f"queue_worker_{uuid.uuid4().hex[:6]}"
        executed_payloads: list[dict[str, Any]] = []

        async def dummy_handler(session: AsyncSession, payload: dict[str, Any]) -> None:
            executed_payloads.append(payload)

        worker = JobWorker()
        worker.register_handler(q_name, dummy_handler)

        await QueueService.enqueue(session=db_session, queue=q_name, payload={"action": "test"})

        # Process job
        processed = await worker.process_one(session=db_session, queue=q_name)
        assert processed is True
        assert len(executed_payloads) == 1
        assert executed_payloads[0]["action"] == "test"

        # No more jobs
        processed_empty = await worker.process_one(session=db_session, queue=q_name)
        assert processed_empty is False


class TestFastAPIProductionEndpoints:
    """Test all production REST API routes."""

    @pytest_asyncio.fixture
    async def client(self, db_session: AsyncSession) -> AsyncClient:
        async def override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = override_get_db
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
        app.dependency_overrides.clear()

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_upload_image_and_deduplication(self, client: AsyncClient) -> None:
        unique_url = f"https://example.com/{uuid.uuid4().hex}.jpg"

        # 1. First upload -> 201 Created
        resp1 = await client.post(
            "/images",
            json={"url": unique_url, "source": "upload"},
        )
        assert resp1.status_code == 201
        data1 = resp1.json()
        assert "id" in data1
        assert data1["url"] == unique_url

        # 2. Duplicate upload -> 409 Conflict
        resp2 = await client.post(
            "/images",
            json={"url": unique_url, "source": "upload"},
        )
        assert resp2.status_code == 409

    @pytest.mark.asyncio
    async def test_create_article_and_matches(self, client: AsyncClient) -> None:
        title = f"Wild Foxes of Alaska {uuid.uuid4().hex[:6]}"
        resp_art = await client.post(
            "/articles",
            json={"title": title, "body": "A wildlife article documenting wild red foxes and canine behavior."},
        )
        assert resp_art.status_code == 201
        art_data = resp_art.json()
        art_id = art_data["id"]

        # Duplicate article -> 409 Conflict
        resp_dup = await client.post(
            "/articles",
            json={"title": title, "body": "Duplicate", "source_id": art_data["source_id"]},
        )
        assert resp_dup.status_code == 409

        # Get matches
        resp_matches = await client.get(f"/articles/{art_id}/matches?k=5")
        assert resp_matches.status_code == 200
        matches = resp_matches.json()
        assert isinstance(matches, list)
        if matches:
            top = matches[0]
            assert "image_id" in top
            assert "scores" in top
            assert "guard" in top

    @pytest.mark.asyncio
    async def test_get_matches_404(self, client: AsyncClient) -> None:
        fake_id = str(uuid.uuid4())
        resp = await client.get(f"/articles/{fake_id}/matches")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_agent_chat_endpoint(self, client: AsyncClient, db_session: AsyncSession) -> None:
        # Create test article and editor
        art = Article(
            title="Bengal Tigers in Ranthambore",
            body="Conservation of wild tigers in national parks.",
            source_id=f"art_{uuid.uuid4().hex[:8]}",
            primary_subject="tiger",
            must_have=["tiger", "wildlife"],
        )
        editor = Editor(name="Senior Photo Editor", publication="Wildlife Monthly")
        db_session.add_all([art, editor])
        await db_session.flush()

        resp = await client.post(
            "/agent/chat",
            json={
                "article_id": str(art.id),
                "editor_id": str(editor.id),
                "message": "Find an authentic tiger image for the cover.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert "decision" in data
        assert isinstance(data["reasons"], list)

    @pytest.mark.asyncio
    async def test_suggestions_lifecycle(self, client: AsyncClient, db_session: AsyncSession) -> None:
        art = Article(
            title="Ocean Coral Reefs",
            body="Marine biodiversity in tropical reefs.",
            source_id=f"art_{uuid.uuid4().hex[:8]}",
            primary_subject="coral reef",
        )
        img = Image(
            url=f"https://example.com/reef_{uuid.uuid4().hex}.jpg",
            content_hash=uuid.uuid4().hex,
            source="eval",
            source_id=f"reef_{uuid.uuid4().hex[:6]}",
            subject="coral reef",
            category="marine",
        )
        editor = Editor(name="Marine Editor", publication="Ocean Life")
        db_session.add_all([art, img, editor])
        await db_session.flush()

        sugg = Suggestion(
            article_id=art.id,
            image_id=img.id,
            editor_id=editor.id,
            retrieval_scores={"dense": 0.8},
            reranker_score=0.85,
            guard_result="ACCEPT",
            guard_reasons=["Direct subject match"],
            confidence=0.9,
            reasons=["Coral reef subject matches article topic"],
            status="pending",
        )
        db_session.add(sugg)
        await db_session.flush()

        # 1. Explain suggestion
        resp_exp = await client.get(f"/suggestions/{sugg.id}/explain")
        assert resp_exp.status_code == 200
        exp_data = resp_exp.json()
        assert exp_data["grounding_verified"] is True
        assert len(exp_data["reasons"]) > 0

        # 2. Approve suggestion
        resp_app = await client.post(
            f"/suggestions/{sugg.id}/approve",
            json={"comment": "Looks great for the hero image!"},
        )
        assert resp_app.status_code == 200
        app_data = resp_app.json()
        assert app_data["status"] == "approved"
        assert app_data["review_comment"] == "Looks great for the hero image!"

        # 3. Reject suggestion
        resp_rej = await client.post(
            f"/suggestions/{sugg.id}/reject",
            json={"comment": "Orientation too tight."},
        )
        assert resp_rej.status_code == 200
        rej_data = resp_rej.json()
        assert rej_data["status"] == "rejected"

    @pytest.mark.asyncio
    async def test_costs_endpoint(self, client: AsyncClient) -> None:
        resp = await client.get("/costs")
        assert resp.status_code == 200
        data = resp.json()
        assert "daily_budget_usd" in data
        assert "today_spent_usd" in data
        assert "budget_exceeded" in data
        assert "total_spent_usd" in data
        assert isinstance(data["breakdown"], list)

    @pytest.mark.asyncio
    async def test_process_job_endpoint(self, client: AsyncClient) -> None:
        resp = await client.post("/jobs/test_queue/process")
        assert resp.status_code == 200
        data = resp.json()
        assert data["queue"] == "test_queue"
        assert "processed" in data
