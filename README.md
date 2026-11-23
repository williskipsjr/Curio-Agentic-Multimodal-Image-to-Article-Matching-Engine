# Curio: Agentic Multimodal Image-to-Article Matching Engine

An autonomous AI curator that pairs articles with the right images, explains why they fit, and refuses when nothing in the library meets editorial standards.

[![CI Status](https://img.shields.io/badge/tests-80%2F80%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

---

## 1. What Curio Does

Editorial and publishing teams waste hours manually searching stock photography libraries. Conventional search engines rely on simple keyword matching or raw text embeddings that repeatedly fail in real-world publishing:
- **Confusable Lookalikes**: An article about a **Red fox** gets matched with a **Grey wolf** because both are tagged "wild canid". An article on a **Jaguar** receives an **African Leopard**.
- **Context & Mood Blindness**: An article requiring a **serene, isolated landscape** receives a **crowded tourist beach** because both match the keyword "coastline".
- **Silent Hallucinations & False Positives**: Traditional vector search *always* returns top-$k$ results, even when the library contains zero relevant images.
- **Lack of Editorial Reasoning**: Standard search cannot explain *why* an image fits or converse with an editor to adapt to specific stylistic guidelines.

**Curio** solves this with an **Agentic Multimodal Matching Architecture**:
1. **3-Way Hybrid Retrieval (RRF)**: Combines Postgres GIN keyword search (BM25), MiniLM dense text embeddings (384-d), and OpenCLIP visual embeddings (512-d) via Reciprocal Rank Fusion ($k=60$).
2. **Calibrated Mismatch Guard**: Rejects confusable lookalikes with 100% precision and abstains with `NO_CONFIDENT_MATCH` on unmatchable articles.
3. **Multimodal LLM-as-a-Judge**: Employs Gemini Flash Lite with calibrated rubric scoring ($\kappa = 0.849$ agreement with human editors).
4. **Personalised Listwise Re-Ranking**: A 17-feature LightGBM LambdaMART ranker that learns individual editor preferences from approve/reject feedback (+19.1% nDCG@5 lift).
5. **Autonomous ReAct Curator Agent**: A conversational curator equipped with 8 guardrailed tools that reasons over briefs, inspects metadata, checks editorial constraints, and provides grounded explanations.
6. **Production Work Queue & Streaming Indexing**: Resilient PostgreSQL `SKIP LOCKED` work queue with automatic retries, dead-letter routing, and a p95 image ingestion freshness latency of **1.42s**.

---

## 2. Key Empirical Metrics

All benchmarks are measured against our local PostgreSQL 16 pgvector corpus and documented in [EVIDENCE.md](EVIDENCE.md):

| Capability | Metric | Baseline | Curio Result | Impact |
|:---|:---|:---:|:---:|:---|
| **Retrieval (Stage 1)** | Recall@50 | 72.4% (BM25) | **94.7%** (3-Way RRF) | **+22.3%** lift over keyword search |
| **Retrieval Ranking** | nDCG@5 | 0.582 (BM25) | **0.754** (3-Way RRF) | Superior candidate ordering |
| **Lookalike Vetoes** | Confusable Pair Veto Rate | 0.0% (Vector top-k) | **100.0%** (15/15 vetoed) | Zero confusable animal mismatches |
| **Abstention Accuracy** | False Positive Rate (No-Match) | ~80% (Naive top-k) | **0.0%** (`NO_CONFIDENT_MATCH`) | Prevents irrelevant matches |
| **Editorial Re-ranking** | Personalised nDCG@5 | 0.721 (RRF) | **0.859** (LightGBM) | **+19.1%** average editor lift |
| **LLM Judge Alignment** | Cohen's Kappa ($\kappa$) | - | **0.849** (90% agreement) | Substantial human-level agreement |
| **Curator Agent** | Task Success Rate | - | **86.7%** (100% schema validity) | Reliable ReAct execution |
| **Streaming Indexing** | Ingestion Freshness (p95) | - | **1.42s** ($< 60s$ SLA) | Near-instant searchability |

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────┐
                    │          Human Editor / Client          │
                    └────────────────────┬────────────────────┘
                                         │
                    REST Endpoints & Conversational Agent
                                         │
             ┌───────────────────────────┴───────────────────────────┐
             ▼                                                       ▼
   ┌───────────────────┐                                   ┌───────────────────┐
   │   FastAPI API     │                                   │   Curator Agent   │
   │  (/matches, /chat)│                                   │  (ReAct Loop)     │
   └─────────┬─────────┘                                   └─────────┬─────────┘
             │                                                       │
             ├───────────────────────────────────────────────────────┤
             │ 8 Strict Guardrailed Tools (k <= 50, Confirmation)    │
             ▼                                                       ▼
┌─────────────────────────┐   ┌──────────────────────────┐   ┌─────────────────┐
│ Stage 1: Hybrid Search  │   │ Stage 2: Mismatch Guard  │   │ Stage 3: Ranker │
│ ├─ BM25 tsvector GIN    │──▶│ ├─ Lookalike Vetoes      │──▶│ ├─ LightGBM     │
│ ├─ MiniLM Dense (384d)  │   │ ├─ Must-not-have checks  │   │ ├─ 17 Features  │
│ └─ CLIP Image (512d)    │   │ └─ Isotonic Calibration  │   │ └─ Editor Prefs │
│ Reciprocal Rank Fusion  │   │                          │   │                 │
└────────────┬────────────┘   └──────────────────────────┘   └────────┬────────┘
             │                                                        │
             └───────────────────────────┬────────────────────────────┘
                                         ▼
                 ┌────────────────────────────────────────┐
                 │       PostgreSQL 16 + pgvector         │
                 │ ├─ images (clip, caption, tsv)         │
                 │ ├─ articles (brief_embedding)          │
                 │ ├─ suggestions & labels                │
                 │ ├─ cost_ledger (budget guard)          │
                 │ └─ jobs (SKIP LOCKED resilient queue)  │
                 └────────────────────────────────────────┘
```

---

## 4. Why Curio is "Agentic"

Curio is not a static retrieval script. It features an **Autonomous ReAct (Reasoning + Acting) Curator Agent** ([curio/agent/curator.py](curio/agent/curator.py)):

1. **Dynamic Tool Use**: The agent autonomously executes an 8-tool inventory:
   - `search_hybrid`: Queries PostgreSQL using multi-modal RRF.
   - `inspect_image`: Analyzes candidate metadata, dimensions, tags, and license.
   - `filter_editorial_policy`: Applies aspect-ratio constraints (16:9, square) and diversity filters.
   - `judge_candidate`: Invokes the calibrated Gemini multimodal judge.
   - `explain_rationale`: Formulates grounded visual-editorial pitches.
2. **Conversational Feedback Loop**: Editors converse in natural language (e.g., *"Find a hero photo for an AI article, but avoid clichés with glowing blue circuit boards or robot handshakes"*). The agent parses constraints, updates negative tags, and reformulates queries.
3. **Budget Guard**: Automatically tracks token and USD expenditure per session, enforcing strict quota boundaries.

---

## 5. Dataset Status & Scaling

Curio runs on real multimodal datasets stored in local PostgreSQL 16:
- **Current Indexed Assets**: **725+ local high-resolution images** across 29 categories and 12 confusable animal lookalike groups, plus **253 articles** (Wikipedia + abstract holdout topics).
- **Why 725+ vs 3,000**: The Unsplash Free / Demo API tier enforces a strict limit of **50 requests per hour**. To ensure stable bootstrapping without hitting 429 rate limit errors, Curio downloaded a balanced, representative corpus.
- **Scaling to 3,000 Images**: The ingestion script ([curio/scripts/download_images.py](curio/scripts/download_images.py)) is completely idempotent. To expand to the full 3,000+ catalog:
  ```bash
  # Option A: Resume Unsplash download across hourly windows
  uv run python -m curio.scripts.download_images

  # Option B: Add a free Pexels API key to .env (allows 200 requests/hr)
  # PEXELS_API_KEY=your_key_here
  uv run python -m curio.scripts.download_images
  ```
  It automatically picks up without duplicating existing files and updates `data/image_manifest.json` and PostgreSQL.

---

## 6. API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Service and database health check |
| `POST` | `/images` | Ingest image with pHash + SHA-256 deduplication and queue background indexing |
| `POST` | `/articles` | Ingest article and generate structured brief & embeddings |
| `POST` | `/articles/{id}/matches` | Full multimodal matching pipeline (retrieval + guard + judge + rerank) |
| `POST` | `/agent/chat` | Multi-turn conversational curator session turn |
| `POST` | `/suggestions/{id}/approve` | Approve suggestion (upserts positive human label for reranker) |
| `POST` | `/suggestions/{id}/reject` | Reject suggestion (upserts negative human label for reranker) |
| `GET` | `/suggestions/{id}/explain` | Grounded multimodal explanation citing article themes and image features |
| `GET` | `/costs` | Real-time token consumption, cost ledger, and budget status |
| `POST` | `/jobs/{queue}/process` | Trigger next background job in PostgreSQL work queue |

---

## 7. Quickstart Guide

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- `uv` package manager

### 1. Start Database
```bash
docker compose up -d curio-db
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and set your keys:
```bash
cp .env.example .env
```
```ini
GEMINI_API_KEY=your-gemini-api-key
UNSPLASH_ACCESS_KEY=your-unsplash-access-key
DATABASE_URL=postgresql://curio:curio_dev_password@localhost:5432/curio
DAILY_BUDGET_USD=0.00
```

### 4. Index the Corpus
```bash
uv run python -m curio.scripts.index_corpus
```

### 5. Launch the Server
```bash
uv run uvicorn curio.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API docs are available at **`http://localhost:8000/docs`**.

---

## 8. Running the Complete Test Suite

```bash
uv run pytest tests/ -v
```
All **80/80 tests pass** in under 35 seconds:
- `tests/test_data_scripts.py`: 9 passed
- `tests/test_health.py`: 1 passed
- `tests/test_models.py`: 10 passed
- `tests/test_phase2.py`: 7 passed
- `tests/test_phase3.py`: 11 passed
- `tests/test_phase4.py`: 6 passed
- `tests/test_phase5.py`: 11 passed
- `tests/test_phase6.py`: 12 passed
- `tests/test_pooling.py`: 3 passed
- `tests/test_schemas.py`: 10 passed
