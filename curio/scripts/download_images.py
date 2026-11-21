"""Download and prepare the Unsplash Lite dataset for Curio.

Downloads the Unsplash Lite dataset TSV, selects a curated subset of ~3,000
images across ~20 categories (including confusable groups), and saves the
metadata as a JSON manifest. Images are downloaded to data/images/corpus/.

Usage:
    uv run python -m curio.scripts.download_images
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
CORPUS_DIR = IMAGES_DIR / "corpus"
HOLDBACK_DIR = IMAGES_DIR / "holdback"
MANIFEST_PATH = DATA_DIR / "image_manifest.json"

# Unsplash Lite dataset URLs
# The Lite dataset is a TSV with photo metadata including download URLs
UNSPLASH_LITE_PHOTOS_URL = (
    "https://unsplash.com/data/lite/latest"  # Requires accepting terms
)

# We use the Unsplash API for targeted category downloads instead
UNSPLASH_API_BASE = "https://api.unsplash.com"

# Categories with target counts — total ~3,200 (3,000 corpus + 200 holdback)
CATEGORIES: dict[str, dict[str, Any]] = {
    # Wildlife — including confusable groups
    "red fox": {"count": 90, "confusable_group": "canid"},
    "wolf": {"count": 90, "confusable_group": "canid"},
    "coyote": {"count": 70, "confusable_group": "canid"},
    "wild dog": {"count": 50, "confusable_group": "canid"},
    "leopard": {"count": 80, "confusable_group": "big_cat"},
    "cheetah": {"count": 80, "confusable_group": "big_cat"},
    "jaguar animal": {"count": 70, "confusable_group": "big_cat"},
    "crow bird": {"count": 70, "confusable_group": "corvid"},
    "raven bird": {"count": 70, "confusable_group": "corvid"},
    "alligator": {"count": 70, "confusable_group": "crocodilian"},
    "crocodile": {"count": 70, "confusable_group": "crocodilian"},
    # Nature & Landscapes
    "mountain landscape": {"count": 160, "confusable_group": None},
    "ocean beach": {"count": 160, "confusable_group": None},
    "forest trees": {"count": 130, "confusable_group": None},
    "desert landscape": {"count": 110, "confusable_group": None},
    "sunset sky": {"count": 130, "confusable_group": None},
    "waterfall": {"count": 90, "confusable_group": None},
    # Urban & Architecture
    "city skyline": {"count": 160, "confusable_group": None},
    "historic architecture": {"count": 130, "confusable_group": None},
    "street photography": {"count": 130, "confusable_group": None},
    # Food
    "food photography": {"count": 160, "confusable_group": None},
    # Technology
    "technology workspace": {"count": 130, "confusable_group": None},
    # People & Portraits
    "portrait photography": {"count": 160, "confusable_group": None},
    "group of people": {"count": 110, "confusable_group": None},
    # Abstract & Textures
    "abstract art": {"count": 110, "confusable_group": None},
    "texture pattern": {"count": 90, "confusable_group": None},
    # Sports & Action
    "sports action": {"count": 110, "confusable_group": None},
    # Science & Space
    "space astronomy": {"count": 90, "confusable_group": None},
    # Flowers & Plants
    "flowers garden": {"count": 130, "confusable_group": None},
}

HOLDBACK_COUNT = 200  # Images held back for freshness testing
TOTAL_TARGET = sum(c["count"] for c in CATEGORIES.values())

# Rate limiting
REQUEST_DELAY = 0.1  # seconds between requests


def _sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def _download_image(url: str, dest: Path, client: httpx.Client) -> str | None:
    """Download a single image. Returns content_hash or None on failure."""
    try:
        resp = client.get(url, follow_redirects=True, timeout=30.0)
        resp.raise_for_status()
        content = resp.content
        content_hash = _sha256(content)
        dest.write_bytes(content)
        return content_hash
    except (httpx.HTTPError, OSError) as e:
        logger.warning("Failed to download %s: %s", url, e)
        return None


def _search_unsplash(
    query: str,
    per_page: int,
    page: int,
    client: httpx.Client,
    api_key: str,
) -> list[dict[str, Any]]:
    """Search Unsplash API for photos matching a query."""
    resp = client.get(
        f"{UNSPLASH_API_BASE}/search/photos",
        params={"query": query, "per_page": per_page, "page": page},
        headers={"Authorization": f"Client-ID {api_key}"},
        timeout=15.0,
    )
    resp.raise_for_status()
    return resp.json().get("results", [])  # type: ignore[no-any-return]


def download_category(
    query: str,
    count: int,
    category_info: dict[str, Any],
    client: httpx.Client,
    api_key: str,
    dest_dir: Path,
    seen_ids: set[str],
) -> list[dict[str, Any]]:
    """Download images for a single category. Returns manifest entries."""
    entries: list[dict[str, Any]] = []
    page = 1
    per_page = min(count, 30)  # Unsplash API max is 30

    while len(entries) < count:
        try:
            results = _search_unsplash(query, per_page, page, client, api_key)
        except httpx.HTTPError as e:
            logger.error("API error for '%s' page %d: %s", query, page, e)
            break

        if not results:
            logger.warning(
                "No more results for '%s' at page %d (got %d/%d)",
                query, page, len(entries), count,
            )
            break

        for photo in results:
            if len(entries) >= count:
                break

            photo_id = photo["id"]
            if photo_id in seen_ids:
                continue
            seen_ids.add(photo_id)

            # Use the "small" size (400px wide) to keep download size manageable
            img_url = photo["urls"].get("small", photo["urls"]["raw"] + "&w=400")
            width = photo.get("width", 0)
            height = photo.get("height", 0)

            # Determine orientation
            if width > height:
                orientation = "landscape"
            elif height > width:
                orientation = "portrait"
            else:
                orientation = "square"

            filename = f"{photo_id}.jpg"
            dest_path = dest_dir / filename

            content_hash = None
            if not dest_path.exists():
                content_hash = _download_image(img_url, dest_path, client)
                if content_hash is None:
                    continue
                time.sleep(REQUEST_DELAY)
            else:
                content_hash = _sha256(dest_path.read_bytes())

            entry = {
                "source_id": photo_id,
                "source": "unsplash",
                "url": img_url,
                "local_path": str(dest_path.relative_to(PROJECT_ROOT)),
                "content_hash": content_hash,
                "category_query": query,
                "confusable_group": category_info.get("confusable_group"),
                "orientation": orientation,
                "width": width,
                "height": height,
                "unsplash_description": photo.get("description", ""),
                "unsplash_alt_description": photo.get("alt_description", ""),
                "photographer": photo.get("user", {}).get("name", ""),
            }
            entries.append(entry)

        page += 1
        time.sleep(REQUEST_DELAY)

    logger.info("Downloaded %d/%d images for '%s'", len(entries), count, query)
    return entries


def main() -> None:
    """Main download pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Check for Unsplash API key
    api_key = os.environ.get("UNSPLASH_ACCESS_KEY", "")
    if not api_key:
        logger.error(
            "UNSPLASH_ACCESS_KEY not set. "
            "Get one at https://unsplash.com/developers and add it to .env"
        )
        sys.exit(1)

    # Create directories
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    HOLDBACK_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Target: %d images across %d categories", TOTAL_TARGET, len(CATEGORIES))

    all_entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with httpx.Client() as client:
        for query, info in CATEGORIES.items():
            count = info["count"]
            logger.info("Downloading '%s' (%d images)...", query, count)

            entries = download_category(
                query=query,
                count=count,
                category_info=info,
                client=client,
                api_key=api_key,
                dest_dir=CORPUS_DIR,
                seen_ids=seen_ids,
            )
            all_entries.extend(entries)

    # Split: last HOLDBACK_COUNT go to holdback
    if len(all_entries) > HOLDBACK_COUNT:
        corpus_entries = all_entries[:-HOLDBACK_COUNT]
        holdback_entries = all_entries[-HOLDBACK_COUNT:]

        # Move holdback images to holdback directory
        for entry in holdback_entries:
            src = PROJECT_ROOT / entry["local_path"]
            dst = HOLDBACK_DIR / src.name
            if src.exists():
                src.rename(dst)
                entry["local_path"] = str(dst.relative_to(PROJECT_ROOT))
                entry["holdback"] = True
    else:
        corpus_entries = all_entries
        holdback_entries = []

    manifest = {
        "total": len(all_entries),
        "corpus_count": len(corpus_entries),
        "holdback_count": len(holdback_entries),
        "categories": list(CATEGORIES.keys()),
        "corpus": corpus_entries,
        "holdback": holdback_entries,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Done. %d corpus + %d holdback = %d total. Manifest at %s",
        len(corpus_entries),
        len(holdback_entries),
        len(all_entries),
        MANIFEST_PATH,
    )


if __name__ == "__main__":
    main()
