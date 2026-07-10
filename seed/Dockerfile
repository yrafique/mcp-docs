# mcp-docs seed/embed one-shot — fills the 384-dim bge-small embeddings + builds HNSW
# so the docs store serves FULL HYBRID (BM25 + vector RRF). Built by compose.yml's
# docs-embed service; exits 0 when done. Idempotent (only embeds NULL rows).
FROM python:3.12-slim
RUN pip install --no-cache-dir "psycopg[binary]>=3.1" "fastembed>=0.3"
COPY embed_backfill.py /seed/embed_backfill.py
COPY run-embed.sh      /seed/run-embed.sh
ENV PYTHONUNBUFFERED=1 DOCS_EMBED_CACHE=/models
ENTRYPOINT ["bash", "/seed/run-embed.sh"]
