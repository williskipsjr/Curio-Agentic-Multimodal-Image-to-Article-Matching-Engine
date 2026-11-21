"""Streaming poller simulating live incoming images.

Continuously pulls held-back images into the live index at intervals,
measuring end-to-end freshness latency (time until searchable).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from curio.services.indexer import IndexingService

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "data" / "image_manifest.json"


class StreamingPoller:
    """Pulls held-back images over time into the live index to verify freshness."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        indexer: IndexingService | None = None,
        poll_interval_seconds: float = 15.0,
    ) -> None:
        self.session_factory = session_factory
        self.indexer = indexer or IndexingService()
        self.interval = poll_interval_seconds
        self._running = False
        self._latencies: list[float] = []

    @property
    def p95_latency(self) -> float:
        """Return 95th percentile freshness latency in seconds."""
        if not self._latencies:
            return 0.0
        import numpy as np

        return float(np.percentile(self._latencies, 95))

    async def ingest_one(self, entry: dict[str, Any], client: httpx.AsyncClient) -> bool:
        """Ingest a single held-back image and log freshness."""
        source_id = entry["source_id"]
        url = entry["url"]
        local_path = entry.get("local_path")

        image_bytes: bytes | None = None

        # Check local file if available
        if local_path:
            full_path = PROJECT_ROOT / local_path
            if full_path.exists():
                image_bytes = full_path.read_bytes()

        if image_bytes is None:
            try:
                resp = await client.get(url, timeout=15.0)
                resp.raise_for_status()
                image_bytes = resp.content
            except Exception as e:
                logger.warning("Poller failed to fetch image %s: %s", url, e)
                return False

        async with self.session_factory() as session:
            start = asyncio.get_event_loop().time()
            img = await self.indexer.index_image(
                session=session,
                source_id=source_id,
                url=url,
                image_bytes=image_bytes,
                source="streaming_poller",
                local_path=local_path,
            )
            elapsed = asyncio.get_event_loop().time() - start
            self._latencies.append(elapsed)

            logger.info(
                "[Poller] Ingested image %s in %.2fs (p95: %.2fs)",
                source_id,
                elapsed,
                self.p95_latency,
            )
            return img.searchable_at is not None

    async def run(self, max_images: int = 200) -> None:
        """Run the streaming poller until finished or stopped."""
        if not MANIFEST_PATH.exists():
            logger.warning("Manifest not found at %s. Streaming poller stopped.", MANIFEST_PATH)
            return

        manifest = json.loads(MANIFEST_PATH.read_text())
        holdback = manifest.get("holdback", [])
        if not holdback:
            logger.info("No holdback images in manifest.")
            return

        self._running = True
        logger.info("Streaming poller starting for %d images (interval: %.1fs)", len(holdback), self.interval)

        async with httpx.AsyncClient() as client:
            for _i, entry in enumerate(holdback[:max_images]):
                if not self._running:
                    break
                await self.ingest_one(entry, client)
                await asyncio.sleep(self.interval)

        self._running = False
        logger.info("Streaming poller completed. p95 latency: %.2fs", self.p95_latency)

    def stop(self) -> None:
        """Signal poller to stop."""
        self._running = False
