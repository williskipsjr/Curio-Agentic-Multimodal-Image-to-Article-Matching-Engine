"""Download Wikipedia article intros for Curio evaluation.

Fetches article introductions from Wikipedia using the MediaWiki API.
Selects 200 articles that have matching images in our corpus, plus 30
"no-match" articles on topics the image corpus deliberately excludes.

Usage:
    uv run python -m curio.scripts.download_articles
"""

from __future__ import annotations

import json
import logging
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
ARTICLES_DIR = DATA_DIR / "articles"
DEV_DIR = ARTICLES_DIR / "dev"
TEST_DIR = ARTICLES_DIR / "test"
MANIFEST_PATH = DATA_DIR / "article_manifest.json"

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"

# Articles with subjects that match our image categories
MATCH_ARTICLES: list[dict[str, str]] = [
    # Wildlife — canids (confusable group)
    {"title": "Red fox", "category": "wildlife"},
    {"title": "Arctic fox", "category": "wildlife"},
    {"title": "Fennec fox", "category": "wildlife"},
    {"title": "Gray wolf", "category": "wildlife"},
    {"title": "Ethiopian wolf", "category": "wildlife"},
    {"title": "Coyote", "category": "wildlife"},
    {"title": "African wild dog", "category": "wildlife"},
    {"title": "Dhole", "category": "wildlife"},
    # Wildlife — big cats (confusable group)
    {"title": "Leopard", "category": "wildlife"},
    {"title": "Snow leopard", "category": "wildlife"},
    {"title": "Cheetah", "category": "wildlife"},
    {"title": "Jaguar", "category": "wildlife"},
    {"title": "Cougar", "category": "wildlife"},
    {"title": "Bengal tiger", "category": "wildlife"},
    {"title": "Lion", "category": "wildlife"},
    # Wildlife — corvids (confusable group)
    {"title": "Common raven", "category": "wildlife"},
    {"title": "American crow", "category": "wildlife"},
    {"title": "Eurasian magpie", "category": "wildlife"},
    {"title": "Blue jay", "category": "wildlife"},
    # Wildlife — crocodilians (confusable group)
    {"title": "American alligator", "category": "wildlife"},
    {"title": "Nile crocodile", "category": "wildlife"},
    {"title": "Saltwater crocodile", "category": "wildlife"},
    {"title": "Gharial", "category": "wildlife"},
    # Wildlife — other
    {"title": "Giant panda", "category": "wildlife"},
    {"title": "Red panda", "category": "wildlife"},
    {"title": "Polar bear", "category": "wildlife"},
    {"title": "African elephant", "category": "wildlife"},
    {"title": "Giraffe", "category": "wildlife"},
    {"title": "Emperor penguin", "category": "wildlife"},
    {"title": "Bald eagle", "category": "wildlife"},
    {"title": "Great white shark", "category": "wildlife"},
    {"title": "Bottlenose dolphin", "category": "wildlife"},
    {"title": "Monarch butterfly", "category": "wildlife"},
    {"title": "Coral reef", "category": "wildlife"},
    # Nature & Landscapes
    {"title": "Mount Everest", "category": "landscape"},
    {"title": "Grand Canyon", "category": "landscape"},
    {"title": "Amazon rainforest", "category": "landscape"},
    {"title": "Sahara", "category": "landscape"},
    {"title": "Niagara Falls", "category": "landscape"},
    {"title": "Great Barrier Reef", "category": "landscape"},
    {"title": "Aurora borealis", "category": "landscape"},
    {"title": "Victoria Falls", "category": "landscape"},
    {"title": "Yosemite Valley", "category": "landscape"},
    {"title": "Maldives", "category": "landscape"},
    {"title": "Death Valley", "category": "landscape"},
    {"title": "Yellowstone National Park", "category": "landscape"},
    {"title": "Serengeti", "category": "landscape"},
    {"title": "Antarctic ice sheet", "category": "landscape"},
    {"title": "Redwood National and State Parks", "category": "landscape"},
    {"title": "Bioluminescence", "category": "landscape"},
    {"title": "Rainbow", "category": "landscape"},
    {"title": "Volcanic lightning", "category": "landscape"},
    {"title": "Tundra", "category": "landscape"},
    {"title": "Mangrove", "category": "landscape"},
    # Urban & Architecture
    {"title": "Eiffel Tower", "category": "architecture"},
    {"title": "Colosseum", "category": "architecture"},
    {"title": "Taj Mahal", "category": "architecture"},
    {"title": "Great Wall of China", "category": "architecture"},
    {"title": "Machu Picchu", "category": "architecture"},
    {"title": "Burj Khalifa", "category": "architecture"},
    {"title": "Sagrada Família", "category": "architecture"},
    {"title": "Sydney Opera House", "category": "architecture"},
    {"title": "Angkor Wat", "category": "architecture"},
    {"title": "Parthenon", "category": "architecture"},
    {"title": "Times Square", "category": "architecture"},
    {"title": "Tokyo Tower", "category": "architecture"},
    {"title": "Brooklyn Bridge", "category": "architecture"},
    {"title": "Guggenheim Museum Bilbao", "category": "architecture"},
    {"title": "Petronas Towers", "category": "architecture"},
    # Food
    {"title": "Sushi", "category": "food"},
    {"title": "Pizza", "category": "food"},
    {"title": "Chocolate", "category": "food"},
    {"title": "French cuisine", "category": "food"},
    {"title": "Dim sum", "category": "food"},
    {"title": "Espresso", "category": "food"},
    {"title": "Sourdough", "category": "food"},
    {"title": "Indian cuisine", "category": "food"},
    {"title": "Street food", "category": "food"},
    {"title": "Molecular gastronomy", "category": "food"},
    # Technology
    {"title": "Artificial intelligence", "category": "technology"},
    {"title": "Quantum computing", "category": "technology"},
    {"title": "Internet of things", "category": "technology"},
    {"title": "Robotics", "category": "technology"},
    {"title": "3D printing", "category": "technology"},
    {"title": "Virtual reality", "category": "technology"},
    {"title": "Blockchain", "category": "technology"},
    {"title": "Autonomous car", "category": "technology"},
    {"title": "Drone (unmanned aerial vehicle)", "category": "technology"},
    {"title": "Space telescope", "category": "technology"},
    # People & Society
    {"title": "Olympic Games", "category": "people"},
    {"title": "Marathon", "category": "people"},
    {"title": "Street art", "category": "people"},
    {"title": "Jazz", "category": "people"},
    {"title": "Ballet", "category": "people"},
    {"title": "Carnival", "category": "people"},
    {"title": "Tea ceremony", "category": "people"},
    {"title": "Yoga", "category": "people"},
    {"title": "Photography", "category": "people"},
    {"title": "Surfing", "category": "people"},
    # Abstract & Science
    {"title": "Fractal", "category": "abstract"},
    {"title": "Crystal", "category": "abstract"},
    {"title": "Nebula", "category": "abstract"},
    {"title": "Black hole", "category": "abstract"},
    {"title": "Supernova", "category": "abstract"},
    {"title": "Solar eclipse", "category": "abstract"},
    {"title": "Milky Way", "category": "abstract"},
    {"title": "Saturn", "category": "abstract"},
    {"title": "Mars", "category": "abstract"},
    {"title": "Hubble Space Telescope", "category": "abstract"},
    # Flowers & Plants
    {"title": "Cherry blossom", "category": "flowers"},
    {"title": "Sunflower", "category": "flowers"},
    {"title": "Lavender", "category": "flowers"},
    {"title": "Rose", "category": "flowers"},
    {"title": "Lotus flower", "category": "flowers"},
    {"title": "Tulip", "category": "flowers"},
    {"title": "Orchidaceae", "category": "flowers"},
    {"title": "Bonsai", "category": "flowers"},
    {"title": "Venus flytrap", "category": "flowers"},
    {"title": "Baobab", "category": "flowers"},
    # Sports
    {"title": "Association football", "category": "sports"},
    {"title": "Basketball", "category": "sports"},
    {"title": "Tennis", "category": "sports"},
    {"title": "Swimming (sport)", "category": "sports"},
    {"title": "Rock climbing", "category": "sports"},
    {"title": "Skateboarding", "category": "sports"},
    {"title": "Formula One", "category": "sports"},
    {"title": "Cricket", "category": "sports"},
    {"title": "Snowboarding", "category": "sports"},
    {"title": "Archery", "category": "sports"},
    # Additional diverse topics
    {"title": "Northern lights", "category": "landscape"},
    {"title": "Coral bleaching", "category": "landscape"},
    {"title": "Lightning", "category": "landscape"},
    {"title": "Stalagmite", "category": "landscape"},
    {"title": "Oasis", "category": "landscape"},
    {"title": "Glacier", "category": "landscape"},
    {"title": "Kelp forest", "category": "landscape"},
    {"title": "Salt flat", "category": "landscape"},
    {"title": "Canyon", "category": "landscape"},
    {"title": "Atoll", "category": "landscape"},
    {"title": "Owl", "category": "wildlife"},
    {"title": "Seahorse", "category": "wildlife"},
    {"title": "Chameleon", "category": "wildlife"},
    {"title": "Flamingo", "category": "wildlife"},
    {"title": "Octopus", "category": "wildlife"},
    {"title": "Hummingbird", "category": "wildlife"},
    {"title": "Koala", "category": "wildlife"},
    {"title": "Sea turtle", "category": "wildlife"},
    {"title": "Peacock", "category": "wildlife"},
    {"title": "Pufferfish", "category": "wildlife"},
    {"title": "Origami", "category": "abstract"},
    {"title": "Kaleidoscope", "category": "abstract"},
    {"title": "Stained glass", "category": "architecture"},
    {"title": "Gothic architecture", "category": "architecture"},
    {"title": "Skyscraper", "category": "architecture"},
    {"title": "Lighthouse", "category": "architecture"},
    {"title": "Castle", "category": "architecture"},
    {"title": "Windmill", "category": "architecture"},
    {"title": "Suspension bridge", "category": "architecture"},
    {"title": "Dam", "category": "architecture"},
    {"title": "Pasta", "category": "food"},
    {"title": "Bread", "category": "food"},
    {"title": "Ice cream", "category": "food"},
    {"title": "Cheese", "category": "food"},
    {"title": "Wine", "category": "food"},
    {"title": "Telescope", "category": "technology"},
    {"title": "Solar panel", "category": "technology"},
    {"title": "Wind turbine", "category": "technology"},
    {"title": "Electric vehicle", "category": "technology"},
    {"title": "Satellite", "category": "technology"},
    # Additional subjects to reach 200+
    {"title": "Red deer", "category": "wildlife"},
    {"title": "Moose", "category": "wildlife"},
    {"title": "Brown bear", "category": "wildlife"},
    {"title": "Grizzly bear", "category": "wildlife"},
    {"title": "Sea otter", "category": "wildlife"},
    {"title": "Humpback whale", "category": "wildlife"},
    {"title": "Orca", "category": "wildlife"},
    {"title": "Zebra", "category": "wildlife"},
    {"title": "Hippopotamus", "category": "wildlife"},
    {"title": "Rhinoceros", "category": "wildlife"},
    {"title": "Kangaroo", "category": "wildlife"},
    {"title": "Platypus", "category": "wildlife"},
    {"title": "Mount Fuji", "category": "landscape"},
    {"title": "Matterhorn", "category": "landscape"},
    {"title": "Kilimanjaro", "category": "landscape"},
    {"title": "Table Mountain (Cape Town)", "category": "landscape"},
    {"title": "Loch Ness", "category": "landscape"},
    {"title": "Fjord", "category": "landscape"},
    {"title": "Sand dune", "category": "landscape"},
    {"title": "Volcano", "category": "landscape"},
    {"title": "Geyser", "category": "landscape"},
    {"title": "Cave", "category": "landscape"},
    {"title": "Louvre", "category": "architecture"},
    {"title": "Pantheon, Rome", "category": "architecture"},
    {"title": "Neuschwanstein Castle", "category": "architecture"},
    {"title": "Golden Gate Bridge", "category": "architecture"},
    {"title": "Tower Bridge", "category": "architecture"},
    {"title": "Big Ben", "category": "architecture"},
    {"title": "Pyramids of Giza", "category": "architecture"},
    {"title": "Ramen", "category": "food"},
    {"title": "Taco", "category": "food"},
    {"title": "Croissant", "category": "food"},
    {"title": "Coffee", "category": "food"},
    {"title": "Tea", "category": "food"},
    {"title": "Olive oil", "category": "food"},
    {"title": "Supercomputer", "category": "technology"},
    {"title": "Microscope", "category": "technology"},
    {"title": "Particle accelerator", "category": "technology"},
    {"title": "Smartwatch", "category": "technology"},
]

# Articles about subjects NOT in our image corpus — for testing abstention
NO_MATCH_ARTICLES: list[dict[str, str]] = [
    {"title": "Quantum entanglement", "category": "no_match"},
    {"title": "Dark matter", "category": "no_match"},
    {"title": "Theory of relativity", "category": "no_match"},
    {"title": "DNA", "category": "no_match"},
    {"title": "Higgs boson", "category": "no_match"},
    {"title": "Game theory", "category": "no_match"},
    {"title": "Inflation (cosmology)", "category": "no_match"},
    {"title": "Turing machine", "category": "no_match"},
    {"title": "P versus NP problem", "category": "no_match"},
    {"title": "Riemann hypothesis", "category": "no_match"},
    {"title": "Fermat's Last Theorem", "category": "no_match"},
    {"title": "Nash equilibrium", "category": "no_match"},
    {"title": "Supply and demand", "category": "no_match"},
    {"title": "Cognitive bias", "category": "no_match"},
    {"title": "Existentialism", "category": "no_match"},
    {"title": "Social contract", "category": "no_match"},
    {"title": "Categorical imperative", "category": "no_match"},
    {"title": "Utilitarianism", "category": "no_match"},
    {"title": "Prisoners dilemma", "category": "no_match"},
    {"title": "Bayes' theorem", "category": "no_match"},
    {"title": "Heisenberg uncertainty principle", "category": "no_match"},
    {"title": "Schrödinger equation", "category": "no_match"},
    {"title": "Wave function collapse", "category": "no_match"},
    {"title": "Arrow's impossibility theorem", "category": "no_match"},
    {"title": "Halting problem", "category": "no_match"},
    {"title": "Gödel's incompleteness theorems", "category": "no_match"},
    {"title": "Sapir-Whorf hypothesis", "category": "no_match"},
    {"title": "Occam's razor", "category": "no_match"},
    {"title": "Paradox of thrift", "category": "no_match"},
    {"title": "Tragedy of the commons", "category": "no_match"},
]


def fetch_article_intro(title: str, client: httpx.Client) -> dict[str, Any] | None:
    """Fetch the introduction of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "exintro": True,
        "explaintext": True,
        "inprop": "url",
        "format": "json",
        "redirects": 1,
    }

    try:
        resp = client.get(WIKIPEDIA_API, params=params, timeout=15.0)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("Failed to fetch '%s': %s", title, e)
        return None

    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            logger.warning("Article not found: '%s'", title)
            return None

        extract = page.get("extract", "").strip()
        if not extract or len(extract) < 100:
            logger.warning("Article too short or empty: '%s'", title)
            return None

        return {
            "source_id": f"wikipedia:{page_id}",
            "title": page.get("title", title),
            "body": extract,
            "source_url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"),
        }

    return None


def main() -> None:
    """Main article download pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Create directories
    DEV_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict[str, Any]] = []
    no_match_fetched: list[dict[str, Any]] = []

    headers = {
        "User-Agent": "CurioBot/1.0 (https://github.com/williskipsjr/Curio; curio-matching@flyrank.ai)"
    }
    with httpx.Client(headers=headers) as client:
        # Fetch matching articles
        logger.info("Fetching %d matching articles...", len(MATCH_ARTICLES))
        for i, article_spec in enumerate(MATCH_ARTICLES):
            article = fetch_article_intro(article_spec["title"], client)
            if article:
                article["category"] = article_spec["category"]
                article["is_no_match"] = False
                all_articles.append(article)
                logger.info(
                    "[%d/%d] Fetched: %s (%d chars)",
                    i + 1, len(MATCH_ARTICLES),
                    article["title"], len(article["body"]),
                )
            time.sleep(0.2)  # Be polite to Wikipedia

        # Fetch no-match articles
        logger.info("Fetching %d no-match articles...", len(NO_MATCH_ARTICLES))
        for i, article_spec in enumerate(NO_MATCH_ARTICLES):
            article = fetch_article_intro(article_spec["title"], client)
            if article:
                article["category"] = "no_match"
                article["is_no_match"] = True
                no_match_fetched.append(article)
                logger.info(
                    "[%d/%d] Fetched no-match: %s (%d chars)",
                    i + 1, len(NO_MATCH_ARTICLES),
                    article["title"], len(article["body"]),
                )
            time.sleep(0.2)

    logger.info(
        "Fetched %d matching + %d no-match = %d total articles",
        len(all_articles), len(no_match_fetched),
        len(all_articles) + len(no_match_fetched),
    )

    # Split into dev and test sets
    # Dev: first 100 matching + first 15 no-match
    # Test: remaining matching + remaining no-match
    dev_match = all_articles[:100]
    test_match = all_articles[100:]
    dev_no_match = no_match_fetched[:15]
    test_no_match = no_match_fetched[15:]

    dev_articles = dev_match + dev_no_match
    test_articles = test_match + test_no_match

    # Save individual article files
    for article in dev_articles:
        article["split"] = "dev"
        slug = article["title"].lower().replace(" ", "_").replace("/", "_")[:60]
        filepath = DEV_DIR / f"{slug}.json"
        filepath.write_text(json.dumps(article, indent=2))

    for article in test_articles:
        article["split"] = "test"
        slug = article["title"].lower().replace(" ", "_").replace("/", "_")[:60]
        filepath = TEST_DIR / f"{slug}.json"
        filepath.write_text(json.dumps(article, indent=2))

    # Save manifest
    manifest = {
        "total": len(dev_articles) + len(test_articles),
        "dev_count": len(dev_articles),
        "test_count": len(test_articles),
        "dev_match": len(dev_match),
        "dev_no_match": len(dev_no_match),
        "test_match": len(test_match),
        "test_no_match": len(test_no_match),
        "dev": dev_articles,
        "test": test_articles,
    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "Saved: dev=%d (match=%d, no_match=%d), test=%d (match=%d, no_match=%d)",
        len(dev_articles), len(dev_match), len(dev_no_match),
        len(test_articles), len(test_match), len(test_no_match),
    )
    logger.info("Manifest at %s", MANIFEST_PATH)


if __name__ == "__main__":
    main()
