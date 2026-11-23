# Curio: Agentic Multimodal Image-to-Article Matching Engine

An AI curator that finds the right image for an article, explains why it fits, and refuses when nothing does.

[![CI Status](https://img.shields.io/badge/tests-80%2F80%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-teal.svg)]()
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-blue.svg)]()

---

## 1. Overview

Content teams waste hours digging through stock libraries. Traditional keyword search fails in critical ways:
- A post about a **Red fox** gets a **Grey wolf** because both are "wild canids".
- A post requiring a **calm landscape** gets a **crowded tourist scene**.
- Nothing informs the editor when the library genuinely has **no good option**.

Curio solves this with five core capabilities:
1. **Multimodal Understanding**: Structured tags from Gemini Flash Lite combined with OpenCLIP ViT-B/32 and MiniLM sentence embeddings.
2. **Freshness at Scale**: Live image indexing into PostgreSQL with pgvector HNSW and weighted tsvector full-text search (p95 freshness latency: **1.42s**).
3. **Calibrated Mismatch Guard**: Calibrated precision gates that reject lookalikes and abstain when confidence is low.
4. **Editor Personalisation**: LightGBM LambdaMART re-ranker adapting to editor preferences (+19.1% nDCG@5 lift).
5. **Autonomous Curator Agent**: ReAct reasoning loop with 8 guardrailed tools, preference memory, and grounded explanations.

---

## 2. Architecture

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

## 3. Tech Stack

| Layer | Choice | Details |
|:---|:---|:---|
| **API** | Python 3.12+, FastAPI, Pydantic v2 | High-performance async REST endpoints |
| **LLM & Vision** | Gemini 3.5 Flash Lite | Structured tags, LLM judge, agent reasoning |
| **Embeddings** | MiniLM (384-d) + OpenCLIP ViT-B/32 (512-d) | Local CPU inference with L2 normalization |
| **Database** | PostgreSQL 16 + pgvector | HNSW cosine similarity + English tsvector GIN |
| **Re-Ranking** | LightGBM LambdaMART + Scikit-Learn | Listwise group ranking + Isotonic Regression |
| **Background Queue** | PostgreSQL `SELECT ... FOR UPDATE SKIP LOCKED` | Dead-letter queue, automatic retries |
| **Testing** | Pytest, pytest-asyncio, HTTPX | 80/80 comprehensive test suite |

---

## 4. API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Service health status |
| `POST` | `/images` | Ingest and index image with SHA-256 deduplication |
| `POST` | `/articles` | Create article, extract structured brief & embedding |
| `GET` | `/articles/{id}/matches?k=10` | 3-way hybrid retrieval, guard, and reranked matches |
| `POST` | `/agent/chat` | Autonomous conversational curator agent turn |
| `POST` | `/suggestions/{id}/approve` | Approve suggested pairing (records positive label) |
| `POST` | `/suggestions/{id}/reject` | Reject suggested pairing (records negative label) |
| `GET` | `/suggestions/{id}/explain` | Grounded explanation citing stored tags and scores |
| `GET` | `/costs` | Cost ledger breakdown and daily budget tracking |
| `POST` | `/jobs/{queue}/process` | Trigger next background job from queue |

---

## 5. Quickstart Guide

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
Copy `.env.example` to `.env` and provide your API keys:
```bash
cp .env.example .env
```
```ini
GEMINI_API_KEY=your-gemini-api-key
UNSPLASH_ACCESS_KEY=your-unsplash-key
DATABASE_URL=postgresql://curio:curio_dev_password@localhost:5432/curio
DAILY_BUDGET_USD=0.00
```

### 4. Index the Corpus
```bash
uv run python -m curio.scripts.index_corpus
```

### 5. Run the API Server
```bash
uv run uvicorn curio.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 6. Evaluation & Empirical Evidence

Comprehensive evaluation numbers across the 726-image and 238-article corpus are documented in [EVIDENCE.md](file:///home/williskipsjr/Documents/Willis/Internships/FlyRank%20AI/capstone/Curio/EVIDENCE.md):

- **Recall@50**: **94.7%** (3-Way RRF) vs 72.4% (Keyword BM25 alone).
- **Mismatch Guard Precision**: **100% confusable animal veto rate**; 0% false positives on unmatchable queries.
- **Personalisation Lift**: **+19.1% nDCG@5** across diverse editor profiles.
- **Streaming Freshness Latency**: **1.42s** p95 latency.
- **Inter-Annotator Agreement**: Cohen's Kappa $\kappa = 0.849$ between LLM judge and human labels.

---

## 7. Running Tests

```bash
uv run pytest tests/ -v
```
Output:
```text
============================= 80 passed in 31.09s ==============================
- tests/test_data_scripts.py:   9 passed
- tests/test_health.py:         1 passed
- tests/test_models.py:        10 passed
- tests/test_phase2.py:         7 passed
- tests/test_phase3.py:        11 passed
- tests/test_phase4.py:         6 passed
- tests/test_phase5.py:        11 passed
- tests/test_phase6.py:        12 passed
- tests/test_pooling.py:        3 passed
- tests/test_schemas.py:       10 passed
Total: 80 / 80 PASSED (100%)
```
