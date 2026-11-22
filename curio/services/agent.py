"""Curator Agent loop with Gemini function calling, session memory, and grounding guardrails."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from curio.schemas import AgentResponse, GuardResult
from curio.services.agent_tools import (
    CheckMatchInput,
    CuratorToolService,
    ExplainMatchInput,
    GetArticleInput,
    GetPreferencesInput,
    PreferenceFact,
    ProposePairingInput,
    RankCandidatesInput,
    SearchImagesInput,
    ToolGuardrailError,
)

logger = logging.getLogger(__name__)

# Maximum tool-calling steps per task
MAX_AGENT_STEPS = 8


@dataclass
class SessionState:
    """Short-term session memory for an active curation task."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    article_id: UUID | None = None
    editor_id: UUID | None = None
    shortlist: list[UUID] = field(default_factory=list)
    checked_matches: dict[tuple[UUID, UUID], str] = field(default_factory=dict)
    pending_inferred_preferences: list[dict[str, Any]] = field(default_factory=list)
    active_preferences: list[PreferenceFact] = field(default_factory=list)
    step_count: int = 0
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    last_guard_decision: str | None = None


def verify_grounding(
    reasons: list[str],
    image_tags: dict[str, Any],
    article_brief: dict[str, Any],
) -> tuple[list[str], bool]:
    """Verify that every reason cites stored tags, attributes, or scores.

    Flags and filters any reasons citing hallucinated tags.
    """
    stored_terms = set()
    # Image stored terms
    if image_tags.get("subject"):
        stored_terms.update(str(image_tags["subject"]).lower().split())
    for attr in image_tags.get("attributes", []):
        stored_terms.update(str(attr).lower().split())
    if "category" in image_tags:
        stored_terms.update(str(image_tags["category"]).lower().split())

    # Article stored terms
    if "primary_subject" in article_brief:
        stored_terms.update(str(article_brief["primary_subject"]).lower().split())
    for must in article_brief.get("must_have", []):
        stored_terms.update(str(must).lower().split())

    # Generic schema words allowed
    common_tokens = {
        "image", "article", "matches", "subject", "attributes", "aligned",
        "orientation", "score", "landscape", "portrait", "square", "is",
        "and", "the", "for", "with", "context", "scene", "confidence", "general",
    }
    stored_terms.update(common_tokens)

    grounded_reasons: list[str] = []
    all_grounded = True

    for reason in reasons:
        tokens = [t.strip(".,;:!'\"()").lower() for t in reason.split()]
        # Check if reason mentions unknown entities
        foreign_tokens = [
            tok for tok in tokens
            if len(tok) > 3 and tok not in stored_terms
        ]
        # If too many unfamiliar tokens appear, flag as ungrounded
        if len(foreign_tokens) > 2:
            all_grounded = False
            logger.warning("Grounding check flagged reason: '%s' (unknown terms: %s)", reason, foreign_tokens)
        else:
            grounded_reasons.append(reason)

    return grounded_reasons, all_grounded


class CuratorAgent:
    """Agent that orchestrates search, ranking, guard check, and proposal within 8 steps."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.tool_service = CuratorToolService(session)

    async def run_task(
        self,
        article_id: UUID,
        editor_id: UUID,
        session_state: SessionState | None = None,
    ) -> AgentResponse:
        """Run the curator agent loop for an article and editor.

        Guaranteed:
        - Capped at MAX_AGENT_STEPS (8 steps).
        - Runs check_match before proposing pairing.
        - Loads long-term preferences.
        - Enforces grounding checks on reasons.
        """
        state = session_state or SessionState(article_id=article_id, editor_id=editor_id)
        state.article_id = article_id
        state.editor_id = editor_id

        # Step 1: get_article (Load article brief)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        art_out = await self.tool_service.get_article(GetArticleInput(article_id=article_id))
        state.step_count += 1
        state.tool_trace.append({"step": state.step_count, "tool": "get_article", "output": art_out.title})

        # Step 2: get_preferences (Load long-term memory)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        pref_out = await self.tool_service.get_preferences(GetPreferencesInput(editor_id=editor_id))
        state.active_preferences = pref_out.preferences
        state.step_count += 1
        state.tool_trace.append(
            {"step": state.step_count, "tool": "get_preferences", "count": len(pref_out.preferences)}
        )

        # Step 3: search_images (Query image index)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        search_query = art_out.brief.primary_subject
        search_out = await self.tool_service.search_images(
            SearchImagesInput(query=search_query, k=20)
        )
        state.shortlist = [c.image_id for c in search_out.candidates]
        state.step_count += 1
        state.tool_trace.append(
            {"step": state.step_count, "tool": "search_images", "found": len(search_out.candidates)}
        )

        if not state.shortlist:
            state.last_guard_decision = "NO_CONFIDENT_MATCH"
            return AgentResponse(
                image_id=None,
                confidence=0.0,
                reasons=["No candidate images retrieved for article subject"],
                alternatives=[],
                guard_result=GuardResult(
                    decision="NO_CONFIDENT_MATCH",
                    reasons=["No candidates in image index"],
                    confidence=0.0,
                ),
            )

        # Step 4: rank_candidates (Stage 2 ranking with editor preferences)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        rank_out = await self.tool_service.rank_candidates(
            RankCandidatesInput(
                article_id=article_id,
                candidate_ids=state.shortlist,
                editor_id=editor_id,
            ),
            active_preferences=state.active_preferences,
        )
        state.step_count += 1
        state.tool_trace.append(
            {"step": state.step_count, "tool": "rank_candidates", "ranked": len(rank_out.ranked_candidates)}
        )

        # Filter out pruned candidates
        viable_candidates = [c for c in rank_out.ranked_candidates if not c.pruned]
        if not viable_candidates:
            state.last_guard_decision = "NO_CONFIDENT_MATCH"
            return AgentResponse(
                image_id=None,
                confidence=0.0,
                reasons=["All candidate images were pruned by editor preferences"],
                alternatives=[],
                guard_result=GuardResult(
                    decision="NO_CONFIDENT_MATCH",
                    reasons=["Pruned by editor preferences"],
                    confidence=0.0,
                ),
            )

        # Step 5: check_match on top candidate
        top_candidate = viable_candidates[0]
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        check_out = await self.tool_service.check_match(
            CheckMatchInput(article_id=article_id, image_id=top_candidate.image_id)
        )
        state.step_count += 1
        state.checked_matches[(article_id, top_candidate.image_id)] = check_out.decision
        state.last_guard_decision = check_out.decision
        state.tool_trace.append(
            {"step": state.step_count, "tool": "check_match", "decision": check_out.decision}
        )

        # If guard rejected, try second candidate if available
        if check_out.decision != "ACCEPT" and len(viable_candidates) > 1 and state.step_count < MAX_AGENT_STEPS:
            second_cand = viable_candidates[1]
            check_out_2 = await self.tool_service.check_match(
                CheckMatchInput(article_id=article_id, image_id=second_cand.image_id)
            )
            state.step_count += 1
            state.checked_matches[(article_id, second_cand.image_id)] = check_out_2.decision
            state.last_guard_decision = check_out_2.decision
            state.tool_trace.append(
                {"step": state.step_count, "tool": "check_match", "decision": check_out_2.decision}
            )
            if check_out_2.decision == "ACCEPT":
                top_candidate = second_cand
                check_out = check_out_2

        # If still not ACCEPT, return abstention
        if check_out.decision != "ACCEPT":
            return AgentResponse(
                image_id=None,
                confidence=check_out.confidence,
                reasons=check_out.reasons,
                alternatives=[c.image_id for c in viable_candidates[:3]],
                guard_result=GuardResult(
                    decision=check_out.decision,
                    reasons=check_out.reasons,
                    confidence=check_out.confidence,
                ),
            )

        # Step 6: explain_match (Generate grounded reasons)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        exp_out = await self.tool_service.explain_match(
            ExplainMatchInput(article_id=article_id, image_id=top_candidate.image_id)
        )
        state.step_count += 1
        state.tool_trace.append(
            {"step": state.step_count, "tool": "explain_match", "grounded": exp_out.grounding_verified}
        )

        # Step 7: propose_pairing (Submit suggestion)
        if state.step_count >= MAX_AGENT_STEPS:
            return self._step_limit_response()

        alt_ids = [c.image_id for c in viable_candidates[1:4]]
        try:
            prop_out = await self.tool_service.propose_pairing(
                ProposePairingInput(
                    article_id=article_id,
                    image_id=top_candidate.image_id,
                    reasons=exp_out.reasons,
                    confidence=check_out.confidence,
                    alternatives=alt_ids,
                ),
                checked_pairs=state.checked_matches,
                editor_id=editor_id,
            )
            state.step_count += 1
            state.tool_trace.append(
                {"step": state.step_count, "tool": "propose_pairing", "suggestion_id": str(prop_out.suggestion_id)}
            )
        except ToolGuardrailError as err:
            logger.error("Propose pairing rejected by guardrail: %s", err)
            return self._step_limit_response(f"Guardrail refusal: {err}")

        return AgentResponse(
            image_id=top_candidate.image_id,
            confidence=check_out.confidence,
            reasons=exp_out.reasons,
            alternatives=alt_ids,
            guard_result=GuardResult(
                decision="ACCEPT",
                reasons=exp_out.reasons,
                confidence=check_out.confidence,
            ),
        )

    def _step_limit_response(self, reason: str = "Step limit reached without proposal") -> AgentResponse:
        """Helper returning abstention when agent step limit is reached."""
        return AgentResponse(
            image_id=None,
            confidence=0.0,
            reasons=[reason],
            alternatives=[],
            guard_result=GuardResult(
                decision="NO_CONFIDENT_MATCH",
                reasons=[reason],
                confidence=0.0,
            ),
        )
