# mcp-docs seed/embed one-shot — fills the 384-dim bge-small embeddings + builds HNSW
# so the docs store serves FULL HYBRID (BM25 + vector RRF). Built by compose.yml's
# docs-embed service; exits 0 when done. Idempotent (only embeds NULL rows).
# Built from the repo ROOT context (so it can share ./certs) — see compose.yml.
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

RUN pip install --no-cache-dir "psycopg[binary]>=3.1" "fastembed>=0.3"
COPY seed/embed_backfill.py /seed/embed_backfill.py
COPY seed/run-embed.sh      /seed/run-embed.sh
ENV PYTHONUNBUFFERED=1 DOCS_EMBED_CACHE=/models
ENTRYPOINT ["bash", "/seed/run-embed.sh"]
