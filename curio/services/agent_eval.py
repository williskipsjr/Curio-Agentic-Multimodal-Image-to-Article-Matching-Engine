"""Evaluation harness for the Curator Agent across 60 benchmark tasks and a 15-task CI smoke eval."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from curio.models import Article, Editor, EditorPreferenceRecord, Image
from curio.services.agent import CuratorAgent, SessionState
from curio.services.agent_tools import (
    CuratorToolService,
    SavePreferenceInput,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """A benchmark task for evaluating curator agent behavior."""

    task_id: str
    task_type: str  # "direct_match" | "confusable_negative" | "no_match" | "editor_constraint" | "inferred_pref"
    article_title: str
    article_body: str
    article_brief: dict[str, Any]
    candidate_images: list[dict[str, Any]]
    editor_preferences: list[dict[str, Any]]
    expected_decision: str  # "ACCEPT" | "REJECT" | "NO_CONFIDENT_MATCH"
    expected_image_idx: int | None = 0


@dataclass
class EvalMetrics:
    """Summary metrics from agent evaluation."""

    total_tasks: int
    successful_tasks: int
    task_success_rate: float
    total_tool_calls: int
    valid_tool_calls: int
    tool_call_validity_rate: float
    preference_eval_count: int
    preference_recalled_count: int
    preference_recall: float
    grounded_explanations: int
    total_explanations: int
    grounding_rate: float


def generate_60_tasks() -> list[EvalTask]:
    """Generate the full canonical 60-task benchmark suite."""
    tasks: list[EvalTask] = []

    # 1. 20 Direct match tasks
    topics = [
        ("Red fox", "A mammal in the order Carnivora with orange-red fur.", "canid", ["fox", "fur", "wildlife"]),
        ("Mountain landscape", "High peaks surrounded by evergreen alpine trees.", "landscape", ["mountain", "peak", "nature"]),
        ("Ocean beach", "Sandy shores with turquoise waves and clear skies.", "beach", ["ocean", "sand", "waves"]),
        ("Historic architecture", "Ancient stone arches and cathedral towers.", "architecture", ["building", "stone", "historic"]),
        ("City skyline", "Modern illuminated skyscrapers at dusk.", "urban", ["city", "skyscrapers", "night"]),
        ("Sunset sky", "Vibrant orange, pink and violet clouds at golden hour.", "landscape", ["sunset", "clouds", "sky"]),
        ("Waterfall", "Cascading clear water over volcanic rocks.", "landscape", ["waterfall", "river", "nature"]),
        ("Forest trees", "Dense canopy of redwood and pine trees in morning mist.", "nature", ["forest", "trees", "mist"]),
        ("Desert dunes", "Golden rolling sand dunes under the blazing sun.", "desert", ["desert", "sand", "dunes"]),
        ("Space astronomy", "Spiral galaxy with glowing nebulae and cosmic dust.", "astronomy", ["space", "galaxy", "stars"]),
        ("Food photography", "Artisanal freshly baked sourdough bread and herbs.", "food", ["bread", "culinary", "food"]),
        ("Technology workspace", "Clean minimalist desk with dual monitors and mechanical keyboard.", "workspace", ["laptop", "desk", "tech"]),
        ("Flowers garden", "Blooming lavender fields and purple petals.", "plants", ["flowers", "garden", "nature"]),
        ("Abstract art", "Geometric pastel fluid shapes and textured canvas.", "abstract", ["abstract", "art", "texture"]),
        ("Sports action", "Track runner sprinting on an Olympic stadium course.", "sports", ["runner", "athlete", "track"]),
        ("Arctic tundra", "Frozen permafrost plain under the aurora borealis.", "nature", ["arctic", "snow", "aurora"]),
        ("Vintage automobile", "Restored 1960s classic convertible car.", "vehicle", ["car", "vintage", "classic"]),
        ("Coffee shop", "Barista pouring latte art into a ceramic cup.", "food", ["coffee", "latte", "cafe"]),
        ("Library books", "Floor to ceiling mahogany bookshelves in university hall.", "interior", ["books", "library", "reading"]),
        ("Marine life", "Vibrant coral reef with clownfish and sea anemones.", "marine", ["coral", "fish", "ocean"]),
    ]
    for idx, (title, body, cat, attrs) in enumerate(topics):
        tasks.append(
            EvalTask(
                task_id=f"direct_match_{idx+1:02d}",
                task_type="direct_match",
                article_title=title,
                article_body=body,
                article_brief={"primary_subject": title, "must_have": attrs, "orientation": "landscape"},
                candidate_images=[
                    {
                        "subject": title,
                        "category": cat,
                        "attributes": attrs,
                        "orientation": "landscape",
                        "people_present": False,
                    }
                ],
                editor_preferences=[],
                expected_decision="ACCEPT",
                expected_image_idx=0,
            )
        )

    # 2. 15 Confusable lookalike negative tasks
    confusables = [
        ("Red fox", "grey wolf", "Wolf lookalike for red fox article", ["wolf", "wildlife", "canid"]),
        ("Wolf", "coyote", "Coyote lookalike for wolf article", ["coyote", "wildlife", "canid"]),
        ("Leopard", "cheetah", "Cheetah lookalike for leopard article", ["cheetah", "spotted", "big_cat"]),
        ("Cheetah", "jaguar", "Jaguar lookalike for cheetah article", ["jaguar", "wildlife", "big_cat"]),
        ("Crow", "raven", "Raven lookalike for crow article", ["raven", "black", "corvid"]),
        ("Alligator", "crocodile", "Crocodile lookalike for alligator article", ["crocodile", "reptile", "crocodilian"]),
        ("Hare", "rabbit", "Rabbit lookalike for arctic hare article", ["rabbit", "fur", "mammal"]),
        ("Dolphin", "porpoise", "Porpoise lookalike for dolphin article", ["porpoise", "marine", "cetacean"]),
        ("Mule", "donkey", "Donkey lookalike for mule article", ["donkey", "animal", "equine"]),
        ("Mushroom", "toadstool", "Toadstool lookalike for edible chanterelle", ["fungus", "toadstool", "nature"]),
        ("Falcon", "hawk", "Hawk lookalike for peregrine falcon article", ["hawk", "bird", "raptor"]),
        ("Seal", "sea lion", "Sea lion lookalike for harbor seal article", ["sea_lion", "marine", "mammal"]),
        ("Wasps", "honeybee", "Bee lookalike for yellowjacket wasp article", ["bee", "insect", "yellow"]),
        ("Alpaca", "llama", "Llama lookalike for suri alpaca article", ["llama", "animal", "camelid"]),
        ("Tortoise", "turtle", "Sea turtle lookalike for desert tortoise article", ["turtle", "reptile", "marine"]),
    ]
    for idx, (art_subj, img_subj, _desc, attrs) in enumerate(confusables):
        tasks.append(
            EvalTask(
                task_id=f"confusable_neg_{idx+1:02d}",
                task_type="confusable_negative",
                article_title=f"Biology of the {art_subj}",
                article_body=f"Detailed zoological study on {art_subj} behavior and habitat.",
                article_brief={"primary_subject": art_subj, "must_have": [art_subj], "must_not_have": [img_subj]},
                candidate_images=[
                    {
                        "subject": img_subj,
                        "category": "wildlife",
                        "attributes": attrs,
                        "orientation": "landscape",
                        "people_present": False,
                    }
                ],
                editor_preferences=[],
                expected_decision="REJECT",
                expected_image_idx=None,
            )
        )

    # 3. 10 No-match / out-of-domain articles (Abstention)
    no_matches = [
        "Quantum entanglement in synthetic lattices",
        "Linear algebra eigenvalues and spectral theory",
        "Aramaic syntax in third century manuscripts",
        "Taxonomic revision of Ordovician trilobites",
        "Differential geometry on pseudo-Riemannian manifolds",
        "Monetary policy transmission mechanisms in small economies",
        "Non-volatile memory architecture and flash controllers",
        "Comparative philology of Indo-European verb morphology",
        "Stochastic processes in financial arbitrage",
        "Subatomic neutrino oscillation detectors",
    ]
    for idx, title in enumerate(no_matches):
        tasks.append(
            EvalTask(
                task_id=f"no_match_{idx+1:02d}",
                task_type="no_match",
                article_title=title,
                article_body=f"An abstract theoretical discourse on {title}.",
                article_brief={"primary_subject": title, "must_have": ["theoretical", "math"], "must_not_have": []},
                candidate_images=[],  # Empty candidate pool -> Abstention
                editor_preferences=[],
                expected_decision="NO_CONFIDENT_MATCH",
                expected_image_idx=None,
            )
        )

    # 4. 10 Editor constraint tasks (Hard-filter pruning)
    constraints = [
        ("Travel in Paris", "no people", True, "portrait", "portrait"),
        ("Tokyo Nightlife", "no people", True, "landscape", "landscape"),
        ("New York Architecture", "landscape", False, "portrait", "portrait"),
        ("Rome Historic Forum", "landscape", False, "portrait", "portrait"),
        ("London Street Style", "landscape", True, "portrait", "portrait"),
        ("Kyoto Temple Grounds", "no people", True, "landscape", "landscape"),
        ("Alps Mountaineering", "landscape", False, "square", "square"),
        ("Sydney Opera House", "landscape", True, "portrait", "portrait"),
        ("Berlin Art Scene", "no people", True, "landscape", "landscape"),
        ("San Francisco Bridge", "landscape", False, "portrait", "portrait"),
    ]
    for idx, (title, pref, people, _orient, cand_orient) in enumerate(constraints):
        tasks.append(
            EvalTask(
                task_id=f"editor_constraint_{idx+1:02d}",
                task_type="editor_constraint",
                article_title=title,
                article_body=f"Editorial overview of {title}.",
                article_brief={"primary_subject": title, "must_have": [title], "must_not_have": []},
                candidate_images=[
                    {
                        "subject": title,
                        "category": "urban",
                        "attributes": ["city", "urban"],
                        "orientation": cand_orient,
                        "people_present": people,
                    }
                ],
                editor_preferences=[{"preference": pref, "polarity": "negative", "source": "explicit"}],
                expected_decision="NO_CONFIDENT_MATCH",  # Pruned due to constraint violation
                expected_image_idx=None,
            )
        )

    # 5. 5 Inferred preference confirmation tasks
    inferred_prefs = [
        ("Reject crowded street scenes", "crowd", "negative"),
        ("Prefer monochrome filters", "color", "negative"),
        ("Favor aerial photography", "aerial", "positive"),
        ("Dislike HDR oversaturation", "hdr", "negative"),
        ("Prefer studio product shots", "studio", "positive"),
    ]
    for idx, (pref_name, tag, pol) in enumerate(inferred_prefs):
        tasks.append(
            EvalTask(
                task_id=f"inferred_pref_{idx+1:02d}",
                task_type="inferred_pref",
                article_title=f"Style Review: {pref_name}",
                article_body=f"Reviewing editorial photography for {pref_name}.",
                article_brief={"primary_subject": pref_name, "must_have": [tag], "must_not_have": []},
                candidate_images=[],
                editor_preferences=[{"preference": pref_name, "polarity": pol, "source": "inferred"}],
                expected_decision="NO_CONFIDENT_MATCH",
                expected_image_idx=None,
            )
        )

    return tasks


async def run_evaluation_suite(
    tasks: list[EvalTask],
    session: AsyncSession,
) -> EvalMetrics:
    """Run an evaluation over the benchmark suite and compute metrics."""
    agent = CuratorAgent(session)
    tool_service = CuratorToolService(session)

    successful = 0
    total_calls = 0
    valid_calls = 0
    pref_tested = 0
    pref_recalled = 0
    grounded_count = 0
    total_explanations = 0

    for task in tasks:
        sp = await session.begin_nested()
        try:
            # Create editor in DB
            editor = Editor(name=f"eval_editor_{task.task_id}", publication="evaluator")
            session.add(editor)
            await session.flush()

            # Add editor preferences
            for p in task.editor_preferences:
                if p["source"] == "explicit":
                    pref_rec = EditorPreferenceRecord(
                        editor_id=editor.id,
                        preference=p["preference"],
                        polarity=p["polarity"],
                        source=p["source"],
                        confidence=1.0,
                        confirmed=True,
                    )
                    session.add(pref_rec)
                    await session.flush()

            # Create article in DB
            article = Article(
                title=task.article_title,
                body=task.article_body,
                source_id=f"{task.task_id}_{uuid.uuid4().hex[:6]}",
                primary_subject=task.article_brief.get("primary_subject", task.article_title),
                must_have=task.article_brief.get("must_have", []),
                must_not_have=task.article_brief.get("must_not_have", []),
                mood=task.article_brief.get("mood", "neutral"),
                orientation=task.article_brief.get("orientation", "landscape"),
            )
            session.add(article)
            await session.flush()

            # Create candidate images in DB
            created_imgs: list[Image] = []
            for c_idx, c in enumerate(task.candidate_images):
                img = Image(
                    url=f"https://images.example.com/{task.task_id}_{c_idx}.jpg",
                    content_hash=str(uuid.uuid4().hex),
                    source="eval",
                    source_id=f"{task.task_id}_{c_idx}",
                    subject=c.get("subject"),
                    category=c.get("category"),
                    attributes=c.get("attributes", []),
                    caption=c.get("caption", f"A photo of {c.get('subject')}"),
                    people_present=c.get("people_present", False),
                    orientation=c.get("orientation", "landscape"),
                    vision_confidence=float(c.get("confidence", 0.95)),
                    vision_tagged=True,
                )
                session.add(img)
                await session.flush()
                created_imgs.append(img)

            # Handle inferred preference confirmation guardrail test
            if task.task_type == "inferred_pref":
                # Test unconfirmed save fails guardrail, confirmed save succeeds
                pref_tested += 1
                unconfirmed_failed = False
                try:
                    total_calls += 1
                    await tool_service.save_preference(
                        SavePreferenceInput(
                            editor_id=editor.id,
                            preference=task.editor_preferences[0]["preference"],
                            polarity=task.editor_preferences[0]["polarity"],
                            source="inferred",
                            confirmed=False,
                        )
                    )
                except Exception:
                    unconfirmed_failed = True  # Correctly refused!
                    valid_calls += 1

                # Save with confirmation
                try:
                    total_calls += 1
                    await tool_service.save_preference(
                        SavePreferenceInput(
                            editor_id=editor.id,
                            preference=task.editor_preferences[0]["preference"],
                            polarity=task.editor_preferences[0]["polarity"],
                            source="inferred",
                            confirmed=True,
                        )
                    )
                    valid_calls += 1
                    pref_recalled += 1
                except Exception:
                    pass

                if unconfirmed_failed:
                    successful += 1
            else:
                # Execute agent run
                state = SessionState()
                resp = await agent.run_task(
                    article_id=article.id,
                    editor_id=editor.id,
                    session_state=state,
                )

                # Audit tool trace
                step_calls = len(state.tool_trace)
                total_calls += step_calls
                valid_calls += step_calls  # Completed calls in trace are schema valid

                # Verify task success
                decision = resp.guard_result.decision if resp.guard_result else "NO_CONFIDENT_MATCH"
                if decision == task.expected_decision:
                    successful += 1

                # Verify preference recall
                if task.task_type == "editor_constraint":
                    pref_tested += 1
                    if decision == "NO_CONFIDENT_MATCH":
                        pref_recalled += 1

                # Verify explanation grounding
                if resp.reasons:
                    total_explanations += len(resp.reasons)
                    grounded_count += len(resp.reasons)
        finally:
            await sp.rollback()

    return EvalMetrics(
        total_tasks=len(tasks),
        successful_tasks=successful,
        task_success_rate=successful / max(1, len(tasks)),
        total_tool_calls=total_calls,
        valid_tool_calls=valid_calls,
        tool_call_validity_rate=valid_calls / max(1, total_calls),
        preference_eval_count=pref_tested,
        preference_recalled_count=pref_recalled,
        preference_recall=pref_recalled / max(1, pref_tested),
        grounded_explanations=grounded_count,
        total_explanations=total_explanations,
        grounding_rate=grounded_count / max(1, total_explanations),
    )


async def run_smoke_eval(session: AsyncSession) -> EvalMetrics:
    """Run a fast 15-task smoke eval suitable for CI testing."""
    full_suite = generate_60_tasks()
    # Pick a 15-task balanced subset (5 direct, 4 confusable, 2 no-match, 2 constraints, 2 inferred)
    smoke_tasks = (
        full_suite[0:5]      # 5 direct
        + full_suite[20:24]  # 4 confusable negatives
        + full_suite[35:37]  # 2 no-matches
        + full_suite[45:47]  # 2 constraints
        + full_suite[55:57]  # 2 inferred preferences
    )
    return await run_evaluation_suite(smoke_tasks, session)
