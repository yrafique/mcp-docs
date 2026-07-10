#!/usr/bin/env python3
"""embed_backfill.py — compute fastembed (ONNX) vectors for doc_chunk rows in
docs-db and fill the `embedding vector(384)` column, for the Stage-3 semantic
hybrid (BM25 + vector RRF).

Host-side, CPU-only (BAAI/bge-small-en-v1.5, 384-dim — a strong retrieval model
that is ~2× faster per core than bge-base on CPU and gives a smaller/faster HNSW
index + lower query-embed latency; the small quality delta vs bge-base is masked
by the BM25+RRF fusion). Designed for
a churny lab + "infinite time": a PERSISTENT multiprocessing.Pool of workers, each
loading the model ONCE (not per-chunk — fastembed's own `parallel=` respawns
workers every call, which dominated runtime), streams small shards via
imap_unordered, and the parent bulk-commits every FLUSH_ROWS. Idempotent + resumable
(only embeds rows WHERE embedding IS NULL unless --all; each flush is a checkpoint;
flushes reconnect so a docs-db restart between commits can't kill the run).

    .venv/bin/python scripts/docs/embed_backfill.py            # resume / fill NULLs
    DOCS_EMBED_PARALLEL=32 .venv/bin/python scripts/docs/embed_backfill.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import psycopg

MODEL_NAME = os.environ.get("DOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_DIM = int(os.environ.get("DOCS_EMBED_DIM", "384"))
MODEL_CACHE = os.environ.get("DOCS_EMBED_CACHE", "/models")
DSN = os.environ.get("DOCS_DATABASE_URL",
                     "postgresql://mcp:mcp@docs-db:5432/docs_db")
DOC_CHARS = 1600                                          # ~512 tokens
WORKERS = int(os.environ.get("DOCS_EMBED_PARALLEL", "8"))  # parallel embed workers; keep below free CPU cores
SHARD = int(os.environ.get("DOCS_EMBED_SHARD", "50"))    # rows per task
# bge-base on CPU is ~1.5 rows/s/worker on full 400-token chunks, so a 4000-row
# checkpoint is ~2 min of blind progress — keep it fine for visibility + resume.
FLUSH_ROWS = int(os.environ.get("DOCS_EMBED_FLUSH", "1000"))  # commit checkpoint


def doc_text(heading, topic, body) -> str:
    return " ".join(p for p in (heading, topic, body) if p)[:DOC_CHARS]


def vec_str(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ----- persistent worker: load the model ONCE per process -------------------
_W_MODEL = None


def _init_worker():
    # onnxruntime sizes its OWN intra-op pool to the whole machine and IGNORES
    # OMP_NUM_THREADS — with N workers that is N×cores of oversubscription and
    # the run thrashes (363 runnable threads on a 40-core host). fastembed's
    # `threads=` maps straight to onnxruntime intra_op_num_threads → pin to 1 so
    # each worker is exactly one core; the Pool gives us the parallelism.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    global _W_MODEL
    from fastembed import TextEmbedding
    _W_MODEL = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE, threads=1)


def _embed_shard(shard):
    ids, texts = shard
    vecs = list(_W_MODEL.embed(texts, batch_size=64))
    return [(ids[i], vec_str(vecs[i])) for i in range(len(ids))]


def ensure_hnsw(dsn: str) -> None:
    """Build the HNSW cosine index + ANALYZE once embeddings exist (idempotent) so
    the docs MCP flips to hybrid. This makes embed_backfill.py the single
    'enable hybrid' command on a fresh docs-db: it fills the column AND builds the
    index. pgvector keeps the index current as later rows are embedded."""
    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM doc_chunk WHERE embedding IS NOT NULL LIMIT 1")
            if cur.fetchone() is None:
                return
            # Single-threaded build: pgvector's PARALLEL HNSW build needs a
            # multi-GB /dev/shm DSM segment, but the docs-db container ships the
            # 64 MB Docker default → "No space left on device". Disabling parallel
            # maintenance keeps the work in backend-local memory.
            cur.execute("SET max_parallel_maintenance_workers = 0")
            cur.execute("SET maintenance_work_mem = '2GB'")
            cur.execute("CREATE INDEX IF NOT EXISTS doc_chunk_embedding_hnsw "
                        "ON doc_chunk USING hnsw (embedding vector_cosine_ops) "
                        "WITH (m=16, ef_construction=64)")
            cur.execute("ANALYZE doc_chunk")
        conn.commit()


# ----- bulk, reconnect-safe commit ------------------------------------------
def flush(dsn: str, pairs) -> None:
    if not pairs:
        return
    last = None
    for _ in range(6):
        try:
            with psycopg.connect(dsn, connect_timeout=10) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"CREATE TEMP TABLE _emb (id bigint, v vector({MODEL_DIM})) ON COMMIT DROP")
                    with cur.copy("COPY _emb (id, v) FROM STDIN") as cp:
                        for rid, v in pairs:
                            cp.write_row((rid, v))
                    cur.execute("UPDATE doc_chunk d SET embedding = _emb.v FROM _emb WHERE d.id = _emb.id")
                conn.commit()
            return
        except Exception as e:  # docs-db mid-restart — wait + retry
            last = e
            time.sleep(5)
    raise RuntimeError(f"flush failed: {type(last).__name__}: {last}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill doc_chunk embeddings")
    ap.add_argument("--dsn", default=DSN)
    ap.add_argument("--all", action="store_true", help="re-embed every row")
    args = ap.parse_args()

    where = "" if args.all else "WHERE embedding IS NULL"
    with psycopg.connect(args.dsn, connect_timeout=10) as rconn:
        with rconn.cursor() as cur:
            cur.execute(f"SELECT id, heading, topic, body FROM doc_chunk {where} ORDER BY id")
            rows = cur.fetchall()
    n = len(rows)
    print(f"[embed] {n} rows to embed with {WORKERS} workers", file=sys.stderr, flush=True)
    if n == 0:
        return 0

    # Build shards (id-list, text-list) of SHARD rows each.
    shards = []
    for i in range(0, n, SHARD):
        block = rows[i:i + SHARD]
        shards.append(([int(r[0]) for r in block],
                       [doc_text(r[1], r[2], r[3]) for r in block]))
    del rows

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    done = 0
    t0 = time.time()
    pending = []
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_embed_shard, shards):
            pending.extend(result)
            done += len(result)
            if len(pending) >= FLUSH_ROWS:
                flush(args.dsn, pending)
                pending = []
                rate = done / (time.time() - t0)
                print(f"[embed] {done}/{n} ({rate:.0f}/s, ETA {(n-done)/max(rate,1)/60:.0f} min)",
                      file=sys.stderr, flush=True)
        flush(args.dsn, pending)
    print(f"[embed] done — {n} rows in {(time.time()-t0)/60:.1f} min", file=sys.stderr, flush=True)
    print("[embed] building HNSW index (enable hybrid) …", file=sys.stderr, flush=True)
    ensure_hnsw(args.dsn)
    print("[embed] HNSW ready — docs MCP will serve hybrid after a docs-mcp restart",
          file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
