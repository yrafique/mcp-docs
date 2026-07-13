#!/usr/bin/env python3
"""Runtime release importer for the single-container mcp-docs.

Reads versions.yml + the RELEASES env (default: registry `default_releases`) and,
for each wanted release, loads its per-release dump into its own schema and builds
that schema's BM25 + HNSW indexes. Idempotent — skips a release already built, so
restarts are fast and a persisted volume is reused. This is what makes the tool
"dynamically build the release wanted": set RELEASES, boot, done.
"""
import os
import subprocess
import sys

import psycopg
import yaml

DSN = os.environ.get("DOCS_DATABASE_URL", "postgresql://mcp:mcp@localhost:5432/docs")
ROOT = os.environ.get("MCP_ROOT", "/srv")
REG = yaml.safe_load(open(os.path.join(ROOT, "versions.yml")))
BY_NAME = {r["name"]: r for r in REG["releases"]}
WANT = os.environ.get("RELEASES", "").split() or REG["default_releases"]


def log(m):
    print(m, file=sys.stderr, flush=True)


def build(conn, r):
    sch, dump = r["schema"], os.path.join(ROOT, r["dump"])
    done = conn.execute("SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname=%s",
                        (sch, sch + "_bm25")).fetchone()
    if done:
        log(f"[seed] {r['name']}: already built → skip")
        return
    if not os.path.exists(dump):
        log(f"[seed] {r['name']}: WARNING dump {dump} missing → skip")
        return
    log(f"[seed] {r['name']}: loading {os.path.basename(dump)} → schema {sch} …")
    env = {**os.environ, "PGUSER": "mcp", "PGPASSWORD": "mcp"}
    subprocess.run(f"gunzip -c '{dump}' | psql -v ON_ERROR_STOP=1 -q -h localhost -d docs",
                   shell=True, check=True, env=env)
    log(f"[seed] {r['name']}: building BM25 + HNSW …")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {sch}_bm25 ON {sch}.doc_chunk "
                 f"USING bm25 (id, heading, topic, body) WITH (key_field='id')")
    conn.execute("SET max_parallel_maintenance_workers = 0")
    conn.execute("SET maintenance_work_mem = '2GB'")
    conn.execute(f"CREATE INDEX IF NOT EXISTS {sch}_hnsw ON {sch}.doc_chunk "
                 f"USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)")
    conn.execute(f"ANALYZE {sch}.doc_chunk")
    n = conn.execute(f"SELECT count(*) FROM {sch}.doc_chunk").fetchone()[0]
    log(f"[seed] {r['name']}: done — {n} chunks, BM25 + HNSW ready")


def main():
    log(f"[seed] releases wanted: {WANT}")
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # A persisted PGDATA volume keeps whatever extension version was
        # installed at first init even after the image's base is bumped to a
        # newer paradedb build (extversion only tracks metadata, it does not
        # auto-follow a newer .so). Without this, a container recreated on top
        # of an old volume silently keeps running the old, possibly-buggy
        # pg_search behavior despite shipping a newer binary.
        c.execute("ALTER EXTENSION pg_search UPDATE")
        c.execute("ALTER EXTENSION vector UPDATE")
        for name in WANT:
            r = BY_NAME.get(name)
            if not r:
                log(f"[seed] unknown release '{name}' (not in versions.yml) → skip")
                continue
            build(c, r)
    log("[seed] all requested releases ready")


if __name__ == "__main__":
    main()
