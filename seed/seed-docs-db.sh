#!/bin/bash
# One-shot seed for the docs-db (ParadeDB: pg_search/BM25 + pgvector).
# Runs ONLY on first init via /docker-entrypoint-initdb.d (postgres runs initdb
# scripts only on an empty data dir), so it's a no-op on every later restart and
# the persistent docs-db-data volume keeps the corpus across redeploys.
#
#   1. load the committed doc store (doc_guide + doc_chunk, FTS tsv) from the dump
#   2. enable pg_search (BM25) + vector (pgvector, Stage-3 semantic hybrid)
#   3. build the BM25 index over heading/topic/body (deep-link + classification
#      columns ride along as the stored payload)
#   4. build the HNSW (cosine) index over the bge-base embeddings — only when the
#      dump actually carries vectors, so a plain-FTS dump still seeds cleanly
#
# The dump itself stays a portable plain-FTS pg_dump (loadable into any postgres);
# BM25/pgvector/HNSW are docs-db-only enhancements added here.
set -e
DUMP=/seed/doc_store.sql.gz
PSQL="psql -v ON_ERROR_STOP=0 -U ${POSTGRES_USER} -d ${POSTGRES_DB}"

if [ ! -f "$DUMP" ]; then
  echo "[seed] WARNING: $DUMP not found — docs_db will be empty"; exit 0
fi

# Extensions FIRST — the dump is table-only (no CREATE EXTENSION) and the
# doc_chunk.embedding column is typed `vector(384)`, so the `vector` type must
# exist BEFORE the dump's CREATE TABLE runs, or the load fails. (pg_search is
# likewise needed before the dump's bm25 index is replayed.)
echo "[seed] enabling pg_search (BM25) + vector (pgvector) …"
$PSQL -c "CREATE EXTENSION IF NOT EXISTS pg_search;"
$PSQL -c "CREATE EXTENSION IF NOT EXISTS vector;" || echo "[seed] vector ext unavailable (Stage-3 only) — continuing"

echo "[seed] loading docs store from $DUMP …"
gunzip -c "$DUMP" | $PSQL

# The shipped dump already carries the 384-dim vectors (delivered via Git LFS),
# so the store comes up hybrid-ready. This ALTER is just a safety net: if you swap
# in a corpus-only dump (no embeddings), the column still exists so the docs-embed
# one-shot (embed_backfill.py) can fill any NULL rows and build HNSW → hybrid.
echo "[seed] ensuring embedding column (filled later by embed_backfill.py) …"
$PSQL -c "ALTER TABLE doc_chunk ADD COLUMN IF NOT EXISTS embedding vector(384);"

echo "[seed] ensuring BM25 index over heading/topic/body …"
$PSQL -c "CREATE INDEX IF NOT EXISTS doc_chunk_bm25 ON doc_chunk
          USING bm25 (id, heading, topic, body) WITH (key_field='id');"

# Stage-3 semantic hybrid: build the HNSW (cosine) index over the embeddings —
# but ONLY when the loaded dump actually carries vectors, so a plain-FTS dump
# (no embeddings yet) seeds cleanly without erroring.
HAS_VEC=$($PSQL -tAc "SELECT count(*) FROM doc_chunk WHERE embedding IS NOT NULL" 2>/dev/null || echo 0)
if [ "${HAS_VEC:-0}" -gt 0 ]; then
  echo "[seed] building HNSW (cosine) index over $HAS_VEC embeddings …"
  # single-threaded: parallel HNSW build needs a multi-GB /dev/shm DSM segment,
  # but the container ships the 64 MB Docker default → keep it backend-local.
  $PSQL -c "SET max_parallel_maintenance_workers = 0;
            SET maintenance_work_mem = '2GB';
            CREATE INDEX IF NOT EXISTS doc_chunk_embedding_hnsw ON doc_chunk
            USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64);"
else
  echo "[seed] no embeddings in dump — skipping HNSW (BM25-only / FTS fallback)"
fi

echo "[seed] done: $($PSQL -tAc 'SELECT count(*) FROM doc_chunk' 2>/dev/null || echo '?') chunks, BM25 indexed, vectors=${HAS_VEC:-0}"
