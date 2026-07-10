#!/usr/bin/env bash
# mcp-docs — NSP 26.4 + SR OS 26.3 product-doc full-text + hybrid search.
# Server entrypoint on port 9705. Verbs: name | port | enabled | run.
#
# Reads DOCS_DATABASE_URL from the environment (.env). In the container this is the
# Dockerfile ENTRYPOINT; it can also be run directly for local development.
set -u
NAME=mcp-docs
PORT=9705

enabled() { :; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate server.py: MCP_SRC override wins, otherwise the vendored ./src copy.
# Connects to the docs-db over DOCS_DATABASE_URL; src/doc_store.sql.gz is the seed dump.
run() {
  local src="${MCP_SRC:-}"
  [[ -z "$src" ]] && src="$HERE/src"
  while :; do
    echo "[$NAME] starting"
    python3 "$src/server.py" http --port "$PORT" 2>&1
    echo "[$NAME] exited rc=$? — restart in 3s"; sleep 3
  done
}

case "${1:-run}" in
  name)    echo "$NAME" ;;
  port)    echo "$PORT" ;;
  enabled) enabled ;;
  run)     enabled && run || { echo "$NAME: skipped (guard)"; exit 0; } ;;
  *)       echo "usage: ${0##*/} {name|port|enabled|run}" >&2; exit 2 ;;
esac
