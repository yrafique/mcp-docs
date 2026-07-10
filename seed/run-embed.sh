#!/usr/bin/env bash
# Wait for docs-db to accept TCP, then backfill embeddings + build HNSW (idempotent).
set -euo pipefail
DSN="${DOCS_DATABASE_URL:-postgresql://mcp:mcp@docs-db:5432/docs}"

echo "[embed] waiting for docs-db to accept connections …"
python3 - "$DSN" <<'PY'
import sys, time, psycopg
dsn = sys.argv[1]
for _ in range(120):                       # up to ~4 min
    try:
        psycopg.connect(dsn, connect_timeout=3).close()
        print("[embed] docs-db reachable"); break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("[embed] docs-db never became reachable")
PY

echo "[embed] backfilling embeddings + building HNSW (first run ≈10–15 min, CPU) …"
exec python3 /seed/embed_backfill.py --dsn "$DSN"
