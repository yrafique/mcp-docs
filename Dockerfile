# mcp-docs — standalone Nokia product-doc search MCP. Connects to a docs-db
# (Postgres/ParadeDB) via DOCS_DATABASE_URL; src/doc_store.sql.gz is the seed dump.
# For the full clone-and-run stack (docs-db + embed + server) use `make up` / compose.yml.
# No secrets baked in.
#   Build:  docker build -t mcp-docs .
#   Run:    docker run --rm -p 9705:9705 --env-file .env mcp-docs
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates \
 && rm -rf /var/lib/apt/lists/*

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
COPY launch.sh /srv/launch.sh

ENV PYTHONUNBUFFERED=1 MCP_SRC=/srv/src \
    DOCS_EMBED_CACHE=/app/shared/models

EXPOSE 9705
ENTRYPOINT ["bash", "/srv/launch.sh", "run"]
