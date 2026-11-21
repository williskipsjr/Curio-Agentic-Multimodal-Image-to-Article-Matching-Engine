# Curio — System Design

> Version 0.1 · Phase 1

## 1. System Overview

Curio is a multimodal image-to-article matching engine built as a FastAPI service backed by PostgreSQL with pgvector. An LLM curator agent uses tool-calling to search, rank, and verify image–article pairings, while a mismatch guard prevents wrong-but-similar matches.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Editor (HTTP / Chat)                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                     ┌─────────▼─────────┐
                     │   Curator Agent    │
                     │  (Gemini FC, 8     │
                     │   tools, memory)   │
                     └─────────┬─────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     ┌────────────┐   ┌──────────────┐   ┌──────────┐
     │  Retrieval  │   │  Re-ranking  │   │  Guard   │
     │  (3-way     │   │  (LightGBM   │   │  (rules  │
     │   hybrid)   │   │   or Gemini) │   │   + cal) │
     └──────┬─────┘   └──────┬───────┘   └────┬─────┘
            │                │                 │
            └────────────────┼─────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Live Image    │
                    │     Index       │
                    │  (Postgres +    │
                    │   pgvector)     │
                    └─────────────────┘
```

## 2. Database Schema

### 2.1 Images Table

```sql
CREATE TABLE images (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       TEXT UNIQUE NOT NULL,          -- Unsplash photo ID or upload key
    source          TEXT NOT NULL DEFAULT 'unsplash', -- 'unsplash' | 'upload'
    url             TEXT NOT NULL,
    local_path      TEXT,

    -- Vision tags (from Gemini Flash)
    subject         TEXT,
    category        TEXT,
    attributes      JSONB DEFAULT '[]',
    caption         TEXT,
    people_present  BOOLEAN,
    orientation     TEXT,                          -- 'landscape' | 'portrait' | 'square'
    vision_confidence FLOAT,
    vision_tagged   BOOLEAN DEFAULT FALSE,
    vision_flagged  BOOLEAN DEFAULT FALSE,         -- failed after retries

    -- Embeddings
    caption_embedding   vector(384),               -- MiniLM or Gemini text embedding
    clip_embedding      vector(512),               -- open_clip ViT-B/32

    -- Search
    search_tsv      tsvector,                      -- for full-text search

    -- Deduplication
    content_hash    TEXT UNIQUE,                    -- SHA-256 of image bytes
    perceptual_hash TEXT,                           -- for near-duplicate detection

    -- Metadata
    is_deleted      BOOLEAN DEFAULT FALSE,         -- tombstone for soft deletes
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now(),
    searchable_at   TIMESTAMPTZ                    -- when it became fully indexed
);

-- Indexes
CREATE INDEX idx_images_clip ON images USING hnsw (clip_embedding vector_cosine_ops);
CREATE INDEX idx_images_caption ON images USING hnsw (caption_embedding vector_cosine_ops);
CREATE INDEX idx_images_search ON images USING gin (search_tsv);
CREATE INDEX idx_images_category ON images (category);
CREATE INDEX idx_images_content_hash ON images (content_hash);
CREATE INDEX idx_images_perceptual_hash ON images (perceptual_hash);
```

### 2.2 Articles Table

```sql
CREATE TABLE articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       TEXT UNIQUE NOT NULL,          -- Wikipedia article ID or slug
    title           TEXT NOT NULL,
    body            TEXT NOT NULL,
    source_url      TEXT,

    -- Structured brief (from LLM)
    primary_subject TEXT,
    must_have       JSONB DEFAULT '[]',
    must_not_have   JSONB DEFAULT '[]',
    mood            TEXT,
    orientation     TEXT,

    -- Embedding
    brief_embedding vector(384),                   -- embedding of the structured brief

    -- Metadata
    is_no_match     BOOLEAN DEFAULT FALSE,         -- articles with no valid image match
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);
```

### 2.3 Suggestions Table

```sql
CREATE TABLE suggestions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id      UUID NOT NULL REFERENCES articles(id),
    image_id        UUID NOT NULL REFERENCES images(id),
    editor_id       UUID NOT NULL REFERENCES editors(id),

    -- Scores
    retrieval_scores JSONB NOT NULL,               -- {keyword, dense, clip, rrf}
    reranker_score   FLOAT,
    guard_result     TEXT NOT NULL,                 -- 'ACCEPT' | 'REJECT' | 'NO_CONFIDENT_MATCH'
    guard_reasons    JSONB DEFAULT '[]',
    confidence       FLOAT,

    -- Explanation
    reasons         JSONB DEFAULT '[]',
    alternatives    JSONB DEFAULT '[]',

    -- Human review
    status          TEXT DEFAULT 'pending',         -- 'pending' | 'approved' | 'rejected'
    review_comment  TEXT,
    reviewed_at     TIMESTAMPTZ,

    -- Metadata
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### 2.4 Editors Table

```sql
CREATE TABLE editors (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name            TEXT NOT NULL,
    publication     TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
```

### 2.5 Editor Preferences Table

```sql
CREATE TABLE editor_preferences (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    editor_id       UUID NOT NULL REFERENCES editors(id),
    preference      TEXT NOT NULL,                  -- e.g., "landscape orientation"
    polarity        TEXT NOT NULL,                  -- 'positive' | 'negative'
    source          TEXT NOT NULL,                  -- 'explicit' | 'inferred'
    confidence      FLOAT DEFAULT 1.0,
    confirmed       BOOLEAN DEFAULT FALSE,          -- inferred prefs need confirmation
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_pref_editor ON editor_preferences (editor_id);
```

### 2.6 Cost Ledger Table

```sql
CREATE TABLE cost_ledger (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model           TEXT NOT NULL,                  -- e.g., 'gemini-2.0-flash'
    operation       TEXT NOT NULL,                  -- e.g., 'vision_tag', 'embed', 'rerank'
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    estimated_cost  FLOAT DEFAULT 0.0,              -- USD
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_cost_date ON cost_ledger (created_at);
```

### 2.7 Jobs Table

```sql
CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    queue           TEXT NOT NULL,                  -- 'vision_tag', 'embed', 'index'
    payload         JSONB NOT NULL,
    idempotency_key TEXT UNIQUE,
    status          TEXT DEFAULT 'pending',         -- 'pending' | 'running' | 'done' | 'failed' | 'dead'
    attempts        INTEGER DEFAULT 0,
    max_attempts    INTEGER DEFAULT 3,
    last_error      TEXT,
    locked_at       TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_jobs_queue_status ON jobs (queue, status);
```

### 2.8 Labels Table (for evaluation)

```sql
CREATE TABLE labels (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id      UUID NOT NULL REFERENCES articles(id),
    image_id        UUID NOT NULL REFERENCES images(id),
    grade           INTEGER NOT NULL CHECK (grade IN (0, 1, 2)), -- 0=wrong, 1=acceptable, 2=ideal
    labeler         TEXT NOT NULL,                  -- 'human' | 'llm_judge'
    split           TEXT NOT NULL,                  -- 'dev' | 'test'
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (article_id, image_id, labeler)
);
```

## 3. API Surface

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/images` | Upload a new image |
| `GET` | `/images/{id}` | Get image metadata and tags |
| `POST` | `/articles` | Create an article with auto-brief generation |
| `GET` | `/articles/{id}` | Get article with its structured brief |
| `GET` | `/articles/{id}/matches` | Run the full pipeline and return ranked matches |
| `POST` | `/agent/chat` | Conversational agent endpoint |
| `POST` | `/suggestions/{id}/approve` | Editor approves a suggestion |
| `POST` | `/suggestions/{id}/reject` | Editor rejects a suggestion |
| `GET` | `/suggestions/{id}/explain` | Get grounded explanation for a suggestion |
| `GET` | `/costs` | View cost ledger summary |
| `GET` | `/health` | Health check |

### Internal / Admin

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/admin/index/rebuild` | Trigger full re-indexing |
| `GET` | `/admin/jobs` | View job queue status |
| `POST` | `/admin/poller/start` | Start the streaming poller |

## 4. Pydantic Schemas (Core Models)

### Vision Output

```python
class VisionTags(BaseModel):
    subject: str
    category: str
    attributes: list[str]
    caption: str
    people_present: bool
    orientation: Literal["landscape", "portrait", "square"]
    confidence: float = Field(ge=0.0, le=1.0)
```

### Article Brief

```python
class ArticleBrief(BaseModel):
    primary_subject: str
    must_have: list[str]
    must_not_have: list[str]
    mood: str
    orientation: Literal["landscape", "portrait", "square", "any"]
```

### Guard Result

```python
class GuardResult(BaseModel):
    decision: Literal["ACCEPT", "REJECT", "NO_CONFIDENT_MATCH"]
    reasons: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
```

### Agent Response

```python
class AgentResponse(BaseModel):
    image_id: UUID | None
    confidence: float | None
    reasons: list[str]
    alternatives: list[UUID]
    guard_result: GuardResult | None
```

## 5. Non-Goals

- **Full frontend** — a minimal review page or Gradio demo is enough
- **Fine-tuning any model** — we use off-the-shelf models
- **Image generation** — only as a stretch tool after NO_CONFIDENT_MATCH
- **Scale beyond ~3,000 images** — this is a demonstration, not a production image search at scale
- **Real-time streaming** — batch + polling is sufficient at this scale

## 6. Key Design Decisions

1. **Two embeddings per image (CLIP + text)** — CLIP compares article text directly against image pixels, making the system truly multimodal rather than text-only search
2. **Postgres as the single store** — vectors, full-text search, job queue, and relational data in one place, keeping infrastructure at $0
3. **Guard before propose** — the agent cannot suggest a pairing unless the guard says ACCEPT, preventing wrong-but-similar matches
4. **Idempotent upserts by content hash** — the same image uploaded twice doesn't create duplicates
5. **Cost ledger with budget guard** — every model call is tracked, and batch jobs pause when the daily budget is hit
