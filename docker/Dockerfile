# mcp-docs — single self-contained container: ParadeDB (Postgres + BM25 + pgvector)
# + the MCP server + baked models + bundled per-release doc dumps, all in one image.
#
#   docker build -f docker/Dockerfile -t mcp-docs .          # from repo root
#   docker run -p 9705:9705 mcp-docs                         # defaults: nsp-26.4 + sros-26.3
#   docker run -p 9705:9705 -e RELEASES="nsp-26.4" mcp-docs  # build only one release
#
# On boot it starts Postgres, imports the wanted releases into per-version schemas
# (dynamic — set RELEASES), builds BM25 + HNSW, then serves. No external services.
#
# Pinned by digest (2026-07-13): an unpinned `:latest` had drifted onto a
# pg_search 0.24.0 build whose paradedb.boolean(should => ARRAY[...])
# combinator threw "unboxing minimum_should_match_ argument failed" on every
# hybrid search, taking docs_search fully offline with no build-time signal.
# This digest is pg_search 0.24.2, confirmed to fix paradedb.boolean() itself
# (verified directly); server.py additionally uses the equivalent
# paradedb.disjunction_max(disjuncts => ARRAY[...]) for the boosted-OR ranking,
# which also works here. Bump this digest deliberately, and re-verify
# docs_search end to end, rather than floating on :latest again.
FROM paradedb/paradedb@sha256:87fcc5658642670712e6831c6afc578927ac1ff6ef4fa77c41c8a491cc13ccb9

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Optional: trust extra root CA certs placed in ./certs (proxy). Empty by default.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

# Python deps in an isolated venv (Debian's python is externally-managed).
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /srv/requirements.txt
RUN pip install --no-cache-dir -r /srv/requirements.txt pyyaml

# Bake the embedding + reranker models so search is fully offline at runtime.
RUN python3 -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/shared/models')" \
 || echo "WARNING: embedder model download failed"
RUN python3 -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-12-v2', cache_dir='/app/shared/models')" \
 || echo "WARNING: reranker model download failed"

# App code, release registry, runtime importer, entrypoint, and bundled dumps.
COPY src/ /srv/src/
COPY versions.yml /srv/versions.yml
COPY docker/seed_releases.py /srv/docker/seed_releases.py
COPY docker/entrypoint.sh /srv/docker/entrypoint.sh
COPY data/ /srv/data/

ENV POSTGRES_USER=mcp POSTGRES_PASSWORD=mcp POSTGRES_DB=docs \
    PGDATA=/var/lib/postgresql/data/pgdata \
    DOCS_DATABASE_URL=postgresql://mcp:mcp@localhost:5432/docs \
    DOCS_EMBED_CACHE=/app/shared/models MCP_ROOT=/srv PYTHONUNBUFFERED=1

EXPOSE 9705
ENTRYPOINT ["bash", "/srv/docker/entrypoint.sh"]
