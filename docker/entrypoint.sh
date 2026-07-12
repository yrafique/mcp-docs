#!/usr/bin/env bash
# Single-container entrypoint: Postgres (ParadeDB) + release import + MCP server,
# all in one container. Postgres runs internally; the server talks to it locally.
set -e

# 1. Start Postgres in the background (base image handles initdb + gosu postgres).
docker-entrypoint.sh postgres &

# 2. Wait until it accepts connections.
echo "[entrypoint] waiting for Postgres …"
until pg_isready -h localhost -U "${POSTGRES_USER:-mcp}" -d "${POSTGRES_DB:-docs}" >/dev/null 2>&1; do
  sleep 1
done
echo "[entrypoint] Postgres up"

# 3. Import the wanted releases into their schemas + build indexes (idempotent).
python3 /srv/docker/seed_releases.py

# 4. Start the MCP server (foreground = container PID).
echo "[entrypoint] starting mcp-docs server on :9705"
exec python3 /srv/src/server.py http --port 9705
