"""Embedding service providing text and multimodal CLIP embeddings.

Generates:
1. Dense text embeddings (384-d) for captions and article text via sentence-transformers (all-MiniLM-L6-v2)
2. CLIP embeddings (512-d) for image pixels and cross-modal article text via open_clip ViT-B/32
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Manages both dense text embeddings and open_clip ViT-B/32 multimodal embeddings."""

    def __init__(self) -> None:
        self._text_model: Any = None
        self._clip_model: Any = None
        self._clip_preprocess: Any = None
        self._clip_tokenizer: Any = None

    def _load_text_model(self) -> Any:
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers 'all-MiniLM-L6-v2'...")
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._text_model

    def _load_clip(self) -> tuple[Any, Any, Any]:
        if self._clip_model is None:
            import open_clip

            logger.info("Loading open_clip ViT-B-32 on CPU...")
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            )
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = tokenizer

        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    def embed_text(self, text: str) -> list[float]:
        """Embed text using MiniLM (384 dimensions), L2-normalized."""
        model = self._load_text_model()
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()  # type: ignore[no-any-return]

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Embed an image using open_clip ViT-B/32 (512 dimensions), L2-normalized."""
        import torch

        model, preprocess, _ = self._load_clip()
        pil_img = PILImage.open(BytesIO(image_bytes)).convert("RGB")
        tensor = preprocess(pil_img).unsqueeze(0)

        with torch.no_grad():
            features = model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().tolist()  # type: ignore[no-any-return]

    def embed_article_for_clip(self, text: str) -> list[float]:
        """Embed text query/article into the CLIP multimodal space (512 dimensions)."""
        import torch

        model, _, tokenizer = self._load_clip()
        tokens = tokenizer([text[:300]])  # CLIP max token length is 77 tokens

        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0).cpu().numpy().tolist()  # type: ignore[no-any-return]
