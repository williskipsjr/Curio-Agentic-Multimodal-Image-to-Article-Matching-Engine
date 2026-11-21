# Curio: Agentic Multimodal Image-to-Article Matching Engine

An AI curator that finds the right image for an article, explains why it fits, and refuses when nothing does.

## Overview

Content teams waste hours digging through stock libraries. Keyword search fails in the ways that matter — a post about red foxes gets a grey wolf, a post that needs a calm landscape gets a crowded street, and nothing tells the editor when the library simply has no good option.

Curio solves this with five capabilities that keyword search can't match:

1. **Understand images, not filenames** — structured tags from a vision model, plus image embeddings
2. **Rank across a changing library** — new images are searchable within seconds of arriving
3. **Know when to say no** — a calibrated guard rejects confusable matches and explains why
4. **Adapt to each editor** — remembered preferences change the ranking
5. **Prove it with numbers** — every claim comes from a held-out test set, a fixed eval harness, and confidence intervals

## Architecture

```
Editor ──▶ Curator Agent (Gemini function calling, 8 tools)
                │
                ├── search_images ──▶ Hybrid Retrieval (BM25 + Dense + CLIP) ──▶ RRF
                ├── rank_candidates ──▶ LightGBM LambdaMART Re-ranker
                ├── check_match ──▶ Mismatch Guard (calibrated threshold)
                ├── explain_match ──▶ Grounded Explanations
                ├── get/save_preference ──▶ Preference Memory (Postgres)
                └── propose_pairing ──▶ Human Review
                        │
                        ▼
              Live Image Index (pgvector + BM25)
              Vision Tags (Gemini Flash)
              CLIP + Text Embeddings
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| API | Python 3.12+, FastAPI, Pydantic v2 |
| LLM & Vision | Gemini Flash (free tier), Ollama fallback |
| Embeddings | Gemini text / MiniLM + open_clip ViT-B/32 |
| Database | PostgreSQL 16 + pgvector (HNSW) + full-text search |
| Ranking | LightGBM (LambdaMART), scikit-learn calibration |
| Jobs | Postgres queue (FOR UPDATE SKIP LOCKED) |
| Deploy | Docker Compose locally, HF Spaces + Supabase for demo |
| CI | GitHub Actions (tests + 15-task agent smoke eval) |

## Project Status

> 🚧 **Phase 1: Design and Data** — in progress

See [DESIGN.md](DESIGN.md) for the full system design and [BUILDLOG.md](BUILDLOG.md) for progress tracking.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/williskipsjr/Curio-Agentic-Multimodal-Image-to-Article-Matching-Engine.git
cd Curio-Agentic-Multimodal-Image-to-Article-Matching-Engine

# Set up environment
cp .env.example .env
# Edit .env with your Gemini API key

# Install dependencies (using uv)
uv sync

# Start the database
docker compose up -d db

# Run database migrations
uv run alembic upgrade head

# Start the development server
uv run uvicorn curio.main:app --reload
```

## Development

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check .

# Run type checker
uv run mypy curio/
```

## License

MIT
