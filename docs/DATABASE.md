# Database Schema

PostgreSQL with `pgvector` is recommended.

```sql
CREATE TABLE gestures (
  id UUID PRIMARY KEY,
  gesture_id TEXT UNIQUE NOT NULL,
  word TEXT NOT NULL,
  language TEXT NOT NULL DEFAULT 'rsl',
  phrase TEXT,
  duration_seconds DOUBLE PRECISION NOT NULL,
  fps DOUBLE PRECISION NOT NULL,
  schema_version TEXT NOT NULL,
  json_object_key TEXT NOT NULL,
  glb_object_key TEXT,
  fbx_object_key TEXT,
  vrm_object_key TEXT,
  source_video_sha256 TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE gesture_variants (
  id UUID PRIMARY KEY,
  gesture_id UUID NOT NULL REFERENCES gestures(id) ON DELETE CASCADE,
  signer_profile_id UUID,
  region TEXT,
  quality_score DOUBLE PRECISION,
  notes TEXT
);

CREATE TABLE gesture_embeddings (
  id UUID PRIMARY KEY,
  gesture_id UUID NOT NULL REFERENCES gestures(id) ON DELETE CASCADE,
  model_name TEXT NOT NULL,
  embedding vector(768) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX gestures_word_idx ON gestures (word);
CREATE INDEX gesture_embeddings_vector_idx
ON gesture_embeddings USING ivfflat (embedding vector_cosine_ops);
```

## Notes

- `gesture_id` is stable and human-readable for dictionary references.
- Binary exports live in object storage, not in the database.
- Embeddings are versioned by `model_name` to allow future model upgrades.
