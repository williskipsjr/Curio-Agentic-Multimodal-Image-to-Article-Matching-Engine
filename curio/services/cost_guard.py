"""Cost ledger and budget guard.

Tracks token counts and estimated USD costs across model invocations.
Enforces a configurable daily budget limit to safeguard against unexpected spend.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from curio.config import settings
from curio.models import CostLedger

logger = logging.getLogger(__name__)

# Approximate pricing per 1M tokens (USD)
# Gemini 2.0 Flash is in free tier, but we track nominal rates for metrics
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
    "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    "text-embedding-004": {"input": 0.025 / 1_000_000, "output": 0.0},
    "default": {"input": 0.10 / 1_000_000, "output": 0.40 / 1_000_000},
}


class BudgetExceededError(Exception):
    """Raised when the daily budget limit is reached."""

    pass


class CostTracker:
    """Manages recording API costs and enforcing budget limits."""

    @staticmethod
    def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate estimated cost in USD based on token counts."""
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
        return round(cost, 8)

    @classmethod
    async def record_call(
        cls,
        session: AsyncSession,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostLedger:
        """Record an API invocation in the cost ledger."""
        cost = cls.calculate_cost(model, input_tokens, output_tokens)

        # Check daily budget if budget > 0
        if settings.daily_budget_usd > 0.0:
            today_spent = await cls.get_today_spent(session)
            if today_spent + cost > settings.daily_budget_usd:
                logger.warning(
                    "Budget exceeded: spent $%.4f + call $%.4f > limit $%.4f",
                    today_spent,
                    cost,
                    settings.daily_budget_usd,
                )
                raise BudgetExceededError(
                    f"Daily budget of ${settings.daily_budget_usd:.2f} exceeded."
                )

        entry = CostLedger(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost=cost,
        )
        session.add(entry)
        await session.commit()
        await session.refresh(entry)
        return entry

    @staticmethod
    async def get_today_spent(session: AsyncSession) -> float:
        """Calculate total spend for the current UTC day."""
        today_start = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = select(func.coalesce(func.sum(CostLedger.estimated_cost), 0.0)).where(
            CostLedger.created_at >= today_start
        )
        result = await session.execute(query)
        total: float = result.scalar() or 0.0
        return float(total)

    @staticmethod
    async def get_summary(session: AsyncSession) -> dict[str, Any]:
        """Return aggregate usage breakdown by model and operation."""
        query = select(
            CostLedger.model,
            CostLedger.operation,
            func.count(CostLedger.id).label("calls"),
            func.sum(CostLedger.input_tokens).label("total_input_tokens"),
            func.sum(CostLedger.output_tokens).label("total_output_tokens"),
            func.sum(CostLedger.estimated_cost).label("total_cost"),
        ).group_by(CostLedger.model, CostLedger.operation)

        result = await session.execute(query)
        rows = result.all()

        breakdown = []
        total_usd = 0.0
        for r in rows:
            cost = float(r.total_cost or 0.0)
            total_usd += cost
            breakdown.append(
                {
                    "model": r.model,
                    "operation": r.operation,
                    "calls": r.calls,
                    "input_tokens": r.total_input_tokens or 0,
                    "output_tokens": r.total_output_tokens or 0,
                    "cost_usd": round(cost, 6),
                }
            )

        return {
            "total_spent_usd": round(total_usd, 6),
            "breakdown": breakdown,
        }
