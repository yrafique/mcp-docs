# mcp-docs — standalone Nokia product-doc search MCP. Connects to a docs-db
# (Postgres/ParadeDB) via DOCS_DATABASE_URL; src/doc_store.sql.gz is the seed dump.
# For the full clone-and-run stack (docs-db + embed + server) use `make up` / compose.yml.
# No secrets baked in.
#   Build:  docker build -t mcp-docs .
#   Run:    docker run --rm -p 9705:9705 --env-file .env mcp-docs
# Compose restarts the container (restart: unless-stopped), so the entrypoint runs
# the server directly — no launch/supervisor wrapper.
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Optional: trust any extra root CA certs placed in ./certs. Empty by default → no-op.
COPY certs/ /usr/local/share/ca-certificates/extra/
RUN update-ca-certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model at build time so the container is fully self-contained
# and never needs outbound HuggingFace access at runtime.
RUN python3 -c "\
from fastembed import TextEmbedding; \
TextEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_dir='/app/shared/models')" \
 || echo "WARNING: fastembed model download failed — will retry at runtime"

COPY src/ /srv/src/

ENV PYTHONUNBUFFERED=1 DOCS_EMBED_CACHE=/app/shared/models

EXPOSE 9705
ENTRYPOINT ["python3", "/srv/src/server.py", "http", "--port", "9705"]
