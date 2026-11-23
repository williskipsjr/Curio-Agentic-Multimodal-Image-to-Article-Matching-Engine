# Build Log: Curio Project Milestones

Progress tracking for each phase. All phase gates and verification suites are fully satisfied.

---

## Phase 1: Design and Data ✅ Completed (PR #1 & PR #2)

- [x] Write `DESIGN.md` (schemas, API surface, non-goals)
- [x] Download Unsplash Lite image subset with metadata manifest
- [x] Collect 238 Wikipedia articles (208 real match + 30 no-match)
- [x] Write TREC-style depth-k candidate pooling script (`curio/scripts/create_pool.py`)
- [x] Define SQLAlchemy 2.0 async models with PostgreSQL pgvector (`Image`, `Article`, `Editor`, `Suggestion`, `Label`, `CostLedger`, `Job`)
- [x] Ingest images and articles into Postgres

**Gate:** Design doc committed, 726 images and 238 articles loaded into PostgreSQL.

---

## Phase 2: Vision Pipeline and Live Index ✅ Completed (PR #3)

- [x] Structured vision tagging with Gemini Flash Lite (`curio/services/vision.py`)
- [x] Cost ledger for model calls and daily budget guard (`curio/services/cost_guard.py`)
- [x] Dense caption-text embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
- [x] Multimodal CLIP image & article embeddings (`open_clip ViT-B/32`, 512 dimensions)
- [x] PostgreSQL pgvector cosine similarity indexes and weighted full-text tsvector GIN indexes
- [x] Streaming poller for new incoming images with freshness latency tracking (`curio/services/poller.py`)
- [x] Schema-valid rate logging

**Gate:** 726 images tagged and embedded, p95 freshness latency 1.42s (< 60s target).

---

## Phase 3: Retrieval, Labels and Guard ✅ Completed (PR #4)

- [x] 3-way hybrid retrieval: Keyword BM25 + Dense Text + Multimodal CLIP (`curio/services/retrieval.py`)
- [x] Reciprocal Rank Fusion (RRF with $k=60$)
- [x] Relevance labeling and pooling harness
- [x] LLM Judge (`gemini-3.5-flash-lite`) with Cohen's Kappa validation ($\kappa = 0.849 > 0.70$)
- [x] Calibrated Mismatch Guard with confusable animal lookalike protection (`curio/services/guard.py`)
- [x] Baseline evaluations B0-B3 with bootstrap confidence intervals

**Gate:** Fox article ranks fox first, confusable wolf rejected with reason, 3-way RRF achieves 94.7% Recall@50.

---

## Phase 4: Re-ranking and Personalisation ✅ Completed (PR #5)

- [x] 17-dimensional tabular feature extractor (`curio/services/features.py`)
- [x] LightGBM LambdaMART listwise re-ranker with group ranking (`curio/services/reranker.py`)
- [x] Isotonic regression probability calibration
- [x] Gemini pairwise/listwise reranker comparison harness (`curio/services/llm_reranker.py`)
- [x] Editor preference profiles and simulated personalization evaluation (`curio/services/personalisation_eval.py`)

**Gate:** Re-ranker achieves +19.1% average nDCG@5 lift over baseline across editor profiles.

---

## Phase 5: Curator Agent and Memory ✅ Completed (PR #6)

- [x] Autonomous ReAct loop with Gemini function calling and step caps (`curio/services/agent.py`)
- [x] 8 strict agent tools with Pydantic validation and server-enforced guardrails (`curio/services/agent_tools.py`)
- [x] Persistent preference memory with explicit vs inferred confirmation gates
- [x] Hallucination detector & grounded explanation validator (`GroundingChecker`)
- [x] 60-task benchmark suite & 15-task CI smoke evaluation (`curio/services/agent_eval.py`)

**Gate:** Agent achieves 86.7% task success rate, 100% tool schema validity, 100% grounded explanations in CI.

---

## Phase 6: Production Layer & Launch ✅ Completed (PR #7 & PR #8)

- [x] Production FastAPI REST API endpoints (`curio/main.py`):
  - `POST /images` (upload, SHA-256 deduplication, live indexing)
  - `POST /articles` (creation, structured brief, MiniLM embedding)
  - `GET /articles/{id}/matches` (hybrid retrieval, guard, reranking)
  - `POST /agent/chat` (multi-turn conversational curator agent)
  - `POST /suggestions/{id}/approve` & `POST /suggestions/{id}/reject` (editorial feedback & label upsert)
  - `GET /suggestions/{id}/explain` (grounded explanation & cited tags)
  - `GET /costs` (daily budget & cost ledger summary)
  - `POST /jobs/{queue}/process` (worker trigger)
- [x] PostgreSQL background work queue with `SELECT ... FOR UPDATE SKIP LOCKED` (`curio/services/queue.py`)
- [x] Automatic retry logic with dead-letter queue routing on max attempts
- [x] 12 comprehensive unit and integration tests for Phase 6 (`tests/test_phase6.py`)
- [x] Complete test suite passing: **80 / 80 tests passing (100%)**
- [x] Artifact generation: `EVIDENCE.md`, updated `README.md`, `BUILDLOG.md`

**Gate:** All 80 unit/integration tests pass in CI, API verified, background queue handles concurrent workers safely.
