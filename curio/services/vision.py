"""Vision tagging service using Gemini Flash with structured Pydantic output.

Extracts structured metadata (subject, category, attributes, caption,
people_present, orientation, confidence) from image bytes with retries
and fallback handling.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from PIL import Image as PILImage
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from curio.config import settings
from curio.schemas import VisionTags

logger = logging.getLogger(__name__)

VISION_TAG_PROMPT = """You are an expert visual curator analyzing an image for editorial article matching.
Extract structured metadata about this image strictly adhering to the specified schema:
- subject: Specific primary focal subject (e.g., 'red fox', 'eiffel tower', 'cheetah').
- category: High-level category (e.g., 'wildlife', 'architecture', 'landscape', 'food', 'technology', 'people', 'sports', 'flowers', 'abstract').
- attributes: List of 3-6 distinct descriptive keywords (e.g., ['snow', 'vibrant', 'morning light', 'close-up']).
- caption: A clear, objective 1-2 sentence description of what is seen.
- people_present: true if human faces/bodies are visibly present, false otherwise.
- orientation: 'landscape' (wider than tall), 'portrait' (taller than wide), or 'square'.
- confidence: Your confidence rating between 0.0 and 1.0 in this classification.
"""


class VisionTaggingService:
    """Performs structured vision tagging with validation and retries."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.gemini_api_key
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None and self.api_key:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def tag_image_bytes(self, image_bytes: bytes) -> tuple[VisionTags, dict[str, int]]:
        """Tag an image using Gemini Flash.

        Returns:
            Tuple of (VisionTags, token_counts_dict)
        """
        client = self._get_client()

        if client is None:
            # Deterministic fallback when Gemini API key is unset (dev/testing mode)
            return self._heuristic_tag(image_bytes), {"input": 100, "output": 50}

        try:
            image = PILImage.open(BytesIO(image_bytes))

            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[image, VISION_TAG_PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "response_schema": VisionTags,
                },
            )

            token_usage = {
                "input": getattr(response.usage_metadata, "prompt_token_count", 258)
                if hasattr(response, "usage_metadata")
                else 258,
                "output": getattr(response.usage_metadata, "candidates_token_count", 64)
                if hasattr(response, "usage_metadata")
                else 64,
            }

            parsed: VisionTags = response.parsed
            return parsed, token_usage

        except (ValidationError, Exception) as err:
            logger.warning("Gemini vision tagging attempt failed: %s", err)
            raise

    @staticmethod
    def _heuristic_tag(image_bytes: bytes) -> VisionTags:
        """Derive orientation and fallback tags when API is in mock/stub mode."""
        try:
            img = PILImage.open(BytesIO(image_bytes))
            w, h = img.size
            if w > h * 1.1:
                orientation = "landscape"
            elif h > w * 1.1:
                orientation = "portrait"
            else:
                orientation = "square"
        except Exception:
            orientation = "landscape"

        return VisionTags(
            subject="curated subject",
            category="general",
            attributes=["photograph", "editorial"],
            caption="High resolution editorial photograph.",
            people_present=False,
            orientation=orientation,
            confidence=0.85,
        )
