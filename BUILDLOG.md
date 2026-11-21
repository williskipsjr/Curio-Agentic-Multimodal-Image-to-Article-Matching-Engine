# Build Log

Progress tracking for each phase. Don't start a phase until the previous phase's gate passes.

## Phase 1: Design and Data ⬅️ Current

- [ ] Write DESIGN.md (schemas, API surface, non-goals)
- [ ] Download 3,000-image Unsplash Lite subset
- [ ] Collect 230 articles (200 real + 30 no-match)
- [ ] Write the pooling script
- [ ] Load images and articles into Postgres

**Gate:** Design doc is committed, images and articles are loaded into Postgres.

## Phase 2: Vision Pipeline and Live Index

- [ ] Batch vision tagging job with retries
- [ ] Cost ledger for model calls
- [ ] Caption-text embeddings (Gemini or MiniLM)
- [ ] CLIP image embeddings (open_clip ViT-B/32)
- [ ] pgvector HNSW indexes
- [ ] Keyword index (tsvector)
- [ ] Streaming poller for new images
- [ ] Schema-valid rate logging

**Gate:** Every image is tagged, schema-valid rate is logged, new image searchable in < 60s.

## Phase 3: Retrieval, Labels and Guard

- [ ] Keyword retriever (BM25)
- [ ] Dense caption retriever
- [ ] CLIP image retriever
- [ ] Reciprocal Rank Fusion (RRF)
- [ ] Label 80 articles (pooled top-20)
- [ ] LLM judge + Cohen's kappa validation
- [ ] First version of mismatch guard

**Gate:** Fox article ranks fox first, wolf rejected with reason, B0–B3 baselines reported on dev.

## Phase 4: Re-ranking and Personalisation

- [ ] LightGBM LambdaMART re-ranker
- [ ] Gemini listwise re-ranker comparison
- [ ] Preference features integration
- [ ] Guard calibration (isotonic regression)
- [ ] Quality/latency/cost comparison

**Gate:** Re-ranker beats B3 on dev nDCG@5 with non-overlapping CIs, or documented why not.

## Phase 5: Curator Agent and Memory

- [ ] 8 agent tools with Pydantic schemas
- [ ] Agent loop (capped at 8 steps)
- [ ] Session memory (short-term)
- [ ] Long-term preference memory
- [ ] Guardrails (guard check before propose)
- [ ] Grounding check for reasons
- [ ] 60-task eval harness
- [ ] 15-task CI smoke eval

**Gate:** Agent eval runs in CI, task success and tool-call validity reported.

## Phase 6: Production and Launch

- [ ] Review API endpoints
- [ ] Budget guard (daily limit)
- [ ] Load test
- [ ] Docker Compose setup
- [ ] Deploy to HF Spaces + Supabase
- [ ] Onboard GDG editors
- [ ] Write README.md (final)
- [ ] Write EVIDENCE.md
- [ ] Run test set once, log the date

**Gate:** Live URL, all FlyRank requirements met, final numbers in README.
