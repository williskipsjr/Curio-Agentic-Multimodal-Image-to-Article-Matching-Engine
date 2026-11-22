"""LLM-as-a-judge relevance grader with Cohen's kappa agreement calculation.

Grades image-article pairings on the standard 0/1/2 TREC scale:
- 2: Ideal (directly depicts the primary article subject and fulfills requirements)
- 1: Acceptable (thematically relevant and visually plausible, but not the exact primary subject)
- 0: Wrong / Irrelevant (incorrect entity, conflicting mood, or lookalike error)

Includes statistical Cohen's kappa agreement testing against human ground truth.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from curio.config import settings

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are an expert editorial judge evaluating whether an image is appropriate for an article.
Given the article title and brief, and the image's tags, grade the relevance on a 0-2 scale:
- 2 (Ideal): The image directly features the primary subject of the article and fulfills all visual requirements.
- 1 (Acceptable): The image is relevant to the topic or mood, acceptable for publication, but not the ideal primary subject.
- 0 (Wrong): The image is irrelevant, depicts the wrong entity/lookalike, or violates constraints.

Explain your decision concisely and output JSON.
"""


class JudgeGradeResponse(BaseModel):
    """Structured response from the LLM judge."""

    grade: int = Field(ge=0, le=2, description="Relevance grade 0, 1, or 2")
    explanation: str = Field(description="1-2 sentences explaining the grade")


class LLMJudge:
    """Grades image-article relevance using Gemini Flash or deterministic heuristic."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None and self.api_key:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def grade_pair(
        self,
        article_title: str,
        primary_subject: str,
        image_subject: str,
        image_category: str,
        image_caption: str,
    ) -> tuple[int, str]:
        """Grade an article-image pair."""
        client = self._get_client()

        if client is None:
            # Deterministic heuristic grader for local dev / offline mode
            return self._heuristic_grade(
                article_title, primary_subject, image_subject, image_category
            )

        prompt = f"""Article Title: {article_title}
Target Subject: {primary_subject}

Candidate Image:
- Subject: {image_subject}
- Category: {image_category}
- Caption: {image_caption}
"""
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[JUDGE_PROMPT, prompt],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": JudgeGradeResponse,
                },
            )
            parsed: JudgeGradeResponse = response.parsed
            return parsed.grade, parsed.explanation
        except Exception as e:
            logger.warning("LLM judge failed, using heuristic fallback: %s", e)
            return self._heuristic_grade(
                article_title, primary_subject, image_subject, image_category
            )

    @staticmethod
    def _heuristic_grade(
        article_title: str,
        primary_subject: str,
        image_subject: str,
        image_category: str,
    ) -> tuple[int, str]:
        """Heuristic grading based on string alignment and category matching."""
        target = primary_subject.lower()
        candidate = image_subject.lower()

        # Exact subject overlap -> 2
        if target in candidate or candidate in target:
            return 2, f"Direct subject match between '{primary_subject}' and '{image_subject}'"

        # Broad category overlap -> 1
        if any(cat in target for cat in ["wildlife", "animal", "landscape", "building"]):
            return 1, f"Acceptable topic match between '{article_title}' and '{image_subject}'"

        return 0, f"Subject '{image_subject}' is irrelevant to '{article_title}'"


def compute_cohens_kappa(human_labels: list[int], judge_labels: list[int]) -> float:
    """Compute Cohen's kappa coefficient to measure inter-annotator agreement.

    kappa = (P_o - P_e) / (1 - P_e)
    Where:
      P_o = Observed agreement proportion
      P_e = Expected agreement by chance
    """
    if len(human_labels) != len(judge_labels) or not human_labels:
        return 0.0

    n = len(human_labels)
    categories = sorted(set(human_labels).union(set(judge_labels)))
    k = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Confusion matrix
    matrix = np.zeros((k, k), dtype=np.int64)
    for h, j in zip(human_labels, judge_labels, strict=False):
        matrix[cat_to_idx[h], cat_to_idx[j]] += 1

    # Observed agreement
    p_o = np.trace(matrix) / n

    # Expected agreement by chance
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    p_e = np.sum(row_sums * col_sums)

    if p_e == 1.0:
        return 1.0

    kappa = (p_o - p_e) / (1.0 - p_e)
    return float(round(kappa, 4))
