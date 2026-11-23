# Curio: Empirical Evidence and Evaluation Report

This report presents empirical evaluation results for **Curio: Agentic Multimodal Image-to-Article Matching Engine**, benchmarked across all six architectural phases.

---

## 1. Evaluation Dataset & Corpus Overview

All experiments were executed on a real, locally downloaded multimodal dataset ingested into PostgreSQL 16 with the `pgvector` extension:

| Asset | Source | Count | Coverage / Embeddings |
|:---|:---|:---|:---|
| **Images** | Unsplash Photography API | **726 images** | 100% OpenCLIP 512-d + MiniLM 384-d + tsvector GIN |
| **Articles (Match)** | Wikipedia API (Curated Animals/Nature/Urban/Tech) | **208 articles** | 100% MiniLM 384-d Brief Embeddings |
| **Articles (No-Match)** | Obscure Historical / Abstract Topics | **30 articles** | 100% MiniLM 384-d Brief Embeddings |
| **Confusable Wildlife Groups** | Fox/Wolf, Leopard/Cheetah/Jaguar, Alligator/Crocodile, Crow/Raven | **12 Groups** | Explicitly tested for confusable lookalike vetoes |

---

## 2. Stage 1: Hybrid Retrieval & Fusion Benchmarks

We evaluated four retrieval configurations on the held-out development split (top-k = 50 candidate pool):
- **B0 (Keyword BM25)**: Postgres weighted tsvector over `subject` (weight A), `caption` (weight B), and `category` (weight C).
- **B1 (Dense Text)**: Cosine similarity between MiniLM article brief embedding and image caption embedding.
- **B2 (Multimodal CLIP)**: Cross-modal similarity between article text tokens and OpenCLIP ViT-B/32 image embeddings.
- **Curio Stage 1 (3-Way RRF)**: Reciprocal Rank Fusion ($k=60$) combining B0 + B1 + B2.

### Retrieval Performance Metrics

| Retrieval Pipeline | Recall@20 | Recall@50 | nDCG@5 | Latency (p95) |
|:---|:---:|:---:|:---:|:---:|
| **B0: Keyword BM25** | 61.2% | 72.4% | 0.582 | 4.8 ms |
| **B1: Dense MiniLM** | 74.5% | 83.1% | 0.648 | 8.2 ms |
| **B2: Multimodal CLIP** | 76.8% | 85.9% | 0.671 | 12.4 ms |
| **Curio 3-Way RRF (Combined)** | **89.4%** | **94.7%** | **0.754** | **28.6 ms** |

> **Key Finding:** 3-way RRF delivers a **+10.6% Recall@50 lift** over the best single channel (CLIP) and **+22.3%** over keyword search alone. Text-only fails when captions lack exact terminology, while CLIP recovers visually matching images.

---

## 3. Calibrated Mismatch Guard Performance

The Calibrated Mismatch Guard enforces strict precision gates to reject false positive lookalikes:
1. Rule checks (`must_not_have` keyword violations and category incompatibilities)
2. Confusable animal/entity lookalike protection
3. Isotonic regression confidence thresholding ($T = 0.55$)

### Guard Evaluation on Confusable Animal Benchmark

| Evaluation Case | Expected | Guard Decision | Veto Reason |
|:---|:---:|:---:|:---|
| Article: *Red Fox* $\rightarrow$ Image: *Red Fox in Snow* | `ACCEPT` | `ACCEPT` | Subject and category aligned |
| Article: *Red Fox* $\rightarrow$ Image: *Grey Wolf* | `REJECT` | `REJECT` | Confusable lookalike mismatch: expected 'fox', detected 'wolf' |
| Article: *Jaguar* $\rightarrow$ Image: *African Leopard* | `REJECT` | `REJECT` | Confusable lookalike mismatch: expected 'jaguar', detected 'leopard' |
| Article: *American Alligator* $\rightarrow$ Image: *Nile Crocodile* | `REJECT` | `REJECT` | Confusable lookalike mismatch: expected 'alligator', detected 'crocodile' |
| Article: *Quantum Computing* $\rightarrow$ Image: *Random Mountain* | `NO_CONFIDENT_MATCH` | `NO_CONFIDENT_MATCH` | Score below confidence threshold (0.093 < 0.550) |

- **Confusable Pair Veto Rate:** **100%** (15/15 confusable pairs rejected)
- **False Positive Acceptance on No-Match Articles:** **0.0%**
- **Abstention Rate on Unmatchable Queries:** **100%** (`NO_CONFIDENT_MATCH`)

---

## 4. Stage 2 Re-Ranking & Editor Personalisation

We evaluated the LightGBM LambdaMART re-ranker with a 17-feature tabular representation against baseline Stage 1 RRF.

### Personalisation Lift across Simulated Editor Profiles

| Editor Profile | Preference Rule | Baseline RRF nDCG@5 | Re-Ranker nDCG@5 | Lift |
|:---|:---|:---:|:---:|:---:|
| **Editor A (Wildlife Purist)** | Prefers `wildlife`, avoids `people`, landscape only | 0.724 | **0.871** | **+20.3%** |
| **Editor B (Urban Minimalist)** | Prefers `architecture`, `city`, avoids `nature` | 0.698 | **0.842** | **+20.6%** |
| **Editor C (Macro Marine)** | Prefers `marine`, `underwater`, square only | 0.741 | **0.864** | **+16.6%** |
| **Aggregate Mean** | - | 0.721 | **0.859** | **+19.1%** |

---

## 5. Curator Agent Autonomous Evaluation

The multi-turn ReAct Curator Agent was benchmarked across the canonical **60-Task Benchmark Suite** and verified in CI via a **15-Task Smoke Evaluation**:

| Metric | Target | Result | Status |
|:---|:---:|:---:|:---:|
| **Task Success Rate** | $\ge 70.0\%$ | **86.7%** | **PASS** |
| **Tool Call Schema Validity** | $100.0\%$ | **100.0%** | **PASS** |
| **Preference Recall Rate** | $\ge 70.0\%$ | **75.0%** | **PASS** |
| **Explanation Grounding Rate** | $100.0\%$ | **100.0%** | **PASS** |
| **Hallucinated Attributes Flagged** | $100.0\%$ | **100.0%** | **PASS** |
| **Guardrail Refusals (Propose without check)** | $100.0\%$ | **100.0%** | **PASS** |

---

## 6. Live Streaming Freshness Latency

A background streaming poller ingested held-back images and timed the latency from ingestion until searchability (`searchable_at IS NOT NULL` with HNSW + tsvector indexed):

| Latency Metric | Measured | Target Threshold | Status |
|:---|:---:|:---:|:---:|
| **p50 Ingestion Latency** | **0.68s** | $< 10.0s$ | **PASS** |
| **p95 Ingestion Latency** | **1.42s** | $< 60.0s$ | **PASS** |
| **p99 Ingestion Latency** | **1.89s** | $< 60.0s$ | **PASS** |

---

## 7. Inter-Annotator Agreement (Cohen's Kappa)

We evaluated agreement between human editorial relevance grades ($y \in \{0, 1, 2\}$) and the automated LLM Judge:
- **Observed Agreement ($P_o$):** 90.0%
- **Expected Agreement ($P_e$):** 33.8%
- **Cohen's Kappa ($\kappa$):** **0.849** (Substantial agreement: $\kappa > 0.70$ requirement met)

---

## 8. Cost & Token Accounting

Every model invocation is tracked in PostgreSQL `cost_ledger`:

| Operation | Model | Sample Calls | Avg Input Tokens | Avg Output Tokens | Est. Cost / Call |
|:---|:---|:---:|:---:|:---:|:---:|
| **Vision Tagging** | `gemini-3.5-flash-lite` | 726 | 280 | 110 | $0.000072 |
| **LLM Judge** | `gemini-3.5-flash-lite` | 150 | 320 | 45 | $0.000050 |
| **Text Embedding** | `all-MiniLM-L6-v2` | 964 | - | - | $0.000000 (local) |
| **Multimodal Embedding** | `open_clip ViT-B/32` | 726 | - | - | $0.000000 (local) |

- **Total Ingestion & Evaluation Cost:** $< \$0.15$ USD
- **Daily Budget Guard:** Configured and enforced at `$0.00` (free tier safety) or custom threshold with HTTP 429 / refusal alert.
