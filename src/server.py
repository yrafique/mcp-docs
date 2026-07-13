#!/usr/bin/env python3
"""server.py — mcp local-docs MCP server.

Hybrid search over Nokia product documentation held in a ParadeDB doc store
(Postgres + pg_search/BM25 + pgvector).

Retrieval fuses two signals over the ONE ParadeDB store:
  • BM25 (ParadeDB pg_search) — true IDF + length normalization, precise on the
    exact CLI/REST/alarm/YANG tokens these docs are full of.
  • vector cosine NN (pgvector HNSW, bge-small, embedded in-process) — paraphrase
    and synonym recall.
They are combined with Reciprocal Rank Fusion (ranking="hybrid"); if the vector
half is unavailable the server serves BM25 alone (ranking="bm25").

Standalone server — it does NOT need NSP credentials, only a read connection to
the docs-db (DOCS_DATABASE_URL). Runs over stdio or HTTP:

    python3 server.py            # stdio (default — for Claude Code)
    python3 server.py http        # streamable-HTTP on port 9705
    python3 server.py http --port N

Tools:
    docs.search(query, version?, guide?, limit?)  — ranked passages + citations
    docs.get_chunk(chunk_id, context?)            — full passage + neighbours
    docs.list_guides(version?)                    — what's ingested
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import sys

import psycopg
from psycopg.rows import dict_row
from mcp.server.fastmcp import FastMCP

# Stage-3 semantic hybrid (optional): if the doc store has populated vector
# embeddings + the fastembed model is available in-container, search fuses BM25
# with vector cosine via Reciprocal Rank Fusion. Config via env.
#
# bge-small (384-d, fastembed, baked into the image, CPU) is the embedder. A/B
# tested against bge-m3 (1024-d, Ollama/GPU) and arctic-embed-s: both scored
# lower on an LLM-judged deep-question eval once the reranker (below) is in the
# pipeline, and both drop portability (Ollama needs a host GPU service; neither
# ships in a fastembed-baked image the same way). Not worth it — keep bge-small.
EMBED_MODEL = os.environ.get("DOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.environ.get("DOCS_EMBED_CACHE", "/app/shared/models")
HYBRID_ENABLED = os.environ.get("DOCS_HYBRID", "1") not in ("0", "false", "")
VEC_COLUMN = "embedding"
# mxbai/bge retrieval query instruction (passages embedded without it at ingest).
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

# Stage-4 cross-encoder rerank (optional): re-order the fused top-N with a small
# CPU cross-encoder before returning. On our deep multi-domain question set this
# lifted KPI ~+10 pts vs hybrid alone; ms-marco-MiniLM-L-12 (QA-trained, 120 MB)
# beat both a 1 GB reranker and the small L-6 in A/B, so it is the default.
RERANK_ENABLED = os.environ.get("DOCS_RERANK", "1") not in ("0", "false", "")
RERANK_MODEL = os.environ.get("DOCS_RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-12-v2")
# Candidate pool the cross-encoder re-orders = limit×RERANK_POOL, capped. A tight
# pool matters: a bigger one feeds near-duplicate/tangential distractors the
# cross-encoder happily promotes — A/B showed ~15 beats ~25 on our question set.
RERANK_POOL = int(os.environ.get("DOCS_RERANK_POOL", "3"))
RERANK_POOL_MAX = int(os.environ.get("DOCS_RERANK_POOL_MAX", "18"))
# Prose bias (routing): a conceptual question ("why is NRC-P slow…") shouldn't be
# answered by an SR OS CLI command-reference entry that merely shares a token. When
# the query isn't itself a CLI lookup, demote command-reference chunks in the rerank
# so the conceptual prose guides win. Reference chunks stay available, just lower.
PROSE_BIAS = os.environ.get("DOCS_RERANK_PROSE_BIAS", "1") not in ("0", "false", "")
PROSE_PENALTY = float(os.environ.get("DOCS_RERANK_PROSE_PENALTY", "0.5"))
_CLI_REF_GUIDES = frozenset({
    "md-cli-command-reference", "classic-cli-command-reference",
    "clear-monitor-show-tools-commands"})
_CLI_WORDS = frozenset({"show", "configure", "config", "command", "commands",
                        "cli", "clear", "monitor", "tools", "admin", "syntax"})


def _is_cli_ref(guide: str) -> bool:
    g = (guide or "").lower()
    return g in _CLI_REF_GUIDES or g.endswith("command-reference") or g.endswith("-commands")


def _is_cli_query(query: str) -> bool:
    """True if the query looks like a CLI command lookup (short/imperative or names
    a CLI verb) rather than a conceptual question — then don't apply prose bias."""
    toks = (query or "").lower().split()
    return bool(_CLI_WORDS & set(toks)) or len(toks) <= 3

DSN = os.environ.get(
    "DOCS_DATABASE_URL",
    # Read-only role by default; falls back to mcp/mcp if _ro absent.
    os.environ.get("DATABASE_URL", "postgresql://mcp:mcp@localhost:5432/aggregation_db"),
)
# Multi-release: each (product, version) lives in its OWN schema (nsp_2604,
# sros_263, …), seeded at runtime from its per-release dump. The registry
# (versions.yml) maps product/version → schema; search routes to the right one(s).
MCP_ROOT = os.environ.get(
    "MCP_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@functools.lru_cache(maxsize=1)
def _registry() -> list[dict]:
    """Release registry from versions.yml: [{name, product, version, schema}]."""
    try:
        import yaml
        reg = yaml.safe_load(open(os.path.join(MCP_ROOT, "versions.yml"), encoding="utf-8"))
        return reg.get("releases", [])
    except Exception as e:  # noqa: BLE001
        print(f"[docs] registry unavailable: {type(e).__name__}: {e}", file=sys.stderr)
        return []


mcp = FastMCP("mcp-docs-mcp", host="0.0.0.0", port=9705)


def _q(sql: str, params: tuple = (), schema: str | None = None) -> list[dict]:
    """Run a SELECT with a short bounded retry. If `schema` is given, set the
    search_path first so unqualified doc_chunk/doc_guide resolve to that release."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            with psycopg.connect(DSN, connect_timeout=5, row_factory=dict_row) as c:
                with c.cursor() as cur:
                    if schema:
                        cur.execute(f'SET search_path TO "{schema}", public')
                    cur.execute(sql, params)
                    return cur.fetchall()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"docs-db query failed: {type(last).__name__}: {last}")


_SEEDED: set | None = None


def _seeded_schemas() -> set:
    """Schemas that actually exist in the DB (were seeded). Cached."""
    global _SEEDED
    if _SEEDED is None:
        try:
            _SEEDED = {r["schema_name"] for r in
                       _q("SELECT schema_name FROM information_schema.schemata")}
        except Exception:  # noqa: BLE001
            _SEEDED = set()
    return _SEEDED


def _resolve(product: str = "all", version: str = "") -> list[dict]:
    """Registry entries to search — filtered by product/version and to schemas that
    are actually seeded. Each entry: {name, product, version, schema}."""
    p = (product or "all").strip().lower()
    v = (version or "").strip()
    seeded = _seeded_schemas()
    out = []
    for r in _registry():
        if r["schema"] not in seeded:
            continue
        if p in ("nsp", "sros") and r["product"] != p:
            continue
        if v and str(r["version"]) != v:
            continue
        out.append(r)
    return out


_BM25: bool | None = None


def _has_bm25() -> bool:
    """True if the doc store runs ParadeDB pg_search (BM25). Detected once and
    cached. BM25 (true IDF + length normalization) is the base ranker for search;
    the vector half is fused on top when available."""
    global _BM25
    if _BM25 is None:
        try:
            _BM25 = bool(_q("SELECT 1 FROM pg_extension WHERE extname='pg_search'"))
        except Exception:
            _BM25 = False
    return _BM25


_VEC: bool | None = None
_EMBEDDER = None


def _has_vectors() -> bool:
    """True if the query embedder is available (each seeded release schema is
    guaranteed embeddings + an HNSW index by the runtime seed). Cached."""
    global _VEC
    if _VEC is None:
        _VEC = False
        if not HYBRID_ENABLED:
            return False
        _VEC = bool(_get_embedder() is not None and _seeded_schemas())
    return _VEC


def _get_embedder():
    """Lazy-load the fastembed model once (offline; ~15-30 s first call)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            from fastembed import TextEmbedding
            _EMBEDDER = TextEmbedding(model_name=EMBED_MODEL, cache_dir=EMBED_CACHE)
        except Exception as e:  # noqa: BLE001
            print(f"[docs] embedder unavailable: {type(e).__name__}: {e}",
                  file=sys.stderr)
            _EMBEDDER = False
    return _EMBEDDER or None


@functools.lru_cache(maxsize=512)
def _embed_query(text: str) -> str | None:
    """Embed a QUERY (with the retrieval instruction) → pgvector literal. Cached
    so repeated queries skip the ~20 ms CPU embed."""
    emb = _get_embedder()
    if emb is None:
        return None
    vec = next(iter(emb.embed([QUERY_PROMPT + text])))
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


_RERANKER = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker once (offline; baked into the image)."""
    global _RERANKER
    if _RERANKER is None:
        if not RERANK_ENABLED:
            _RERANKER = False
        else:
            try:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                _RERANKER = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=EMBED_CACHE)
            except Exception as e:  # noqa: BLE001
                print(f"[docs] reranker unavailable: {type(e).__name__}: {e}",
                      file=sys.stderr)
                _RERANKER = False
    return _RERANKER or None


def _rerank_ce(query: str, rows: list[dict], limit: int) -> list[dict]:
    """Re-order the fused candidate pool with the cross-encoder, return top `limit`.
    Scores (query, heading + body-prefix) pairs — a true relevance model, unlike the
    BM25/vector recall scores. Falls back to the given order on any failure."""
    ce = _get_reranker()
    if ce is None or len(rows) <= 1:
        return rows[:limit]
    docs = [((r.get("heading") or "") + ". " + (r.get("body") or "")[:512]) for r in rows]
    try:
        scores = list(ce.rerank(query, docs))
    except Exception:  # noqa: BLE001
        return rows[:limit]
    # Prose-bias routing: for a conceptual question, demote CLI command-reference
    # chunks so shared-token noise doesn't outrank the conceptual guide.
    if PROSE_BIAS and not _is_cli_query(query):
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        scores = [((s - lo) / rng) * (1.0 - PROSE_PENALTY if _is_cli_ref(rows[i].get("guide_slug")) else 1.0)
                  for i, s in enumerate(scores)]
    order = sorted(range(len(rows)), key=lambda i: -scores[i])
    return [rows[i] for i in order][:limit]


def _vector_rows(schema: str, qvec: str, guide: str, cli_mode: str,
                 mgmt_domain: str, limit: int) -> list[dict]:
    """HNSW cosine-NN search within one release schema (sub-ms ANN)."""
    clauses = ""
    tail: list = []
    if guide:
        clauses += " AND c.guide_slug = %s"; tail.append(guide)
    if cli_mode:
        clauses += " AND c.cli_mode = %s"; tail.append(cli_mode)
    if mgmt_domain:
        clauses += " AND c.mgmt_domain = %s"; tail.append(mgmt_domain)
    sql = f"""
        SELECT c.id, c.guide_slug, dg.title, c.page_no, c.heading, c.body,
               c.deep_url, c.topic, c.version, c.cli_mode, c.mgmt_domain,
               c.context_tree, dg.landing_url, dg.pdf_url,
               1 - (c.{VEC_COLUMN} <=> %s::vector) AS rank
        FROM doc_chunk c
        JOIN doc_guide dg ON dg.slug = c.guide_slug AND dg.version = c.version
        WHERE c.{VEC_COLUMN} IS NOT NULL{clauses}
        ORDER BY c.{VEC_COLUMN} <=> %s::vector
        LIMIT %s
    """
    return _q(sql, tuple([qvec] + tail + [qvec, limit]), schema=schema)


def _rrf(lists: list[list[dict]], limit: int, k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion of several ranked result lists (BM25 + vector)."""
    score: dict = {}
    meta: dict = {}
    for rows in lists:
        for rank, r in enumerate(rows):
            rid = r["id"]
            score[rid] = score.get(rid, 0.0) + 1.0 / (k + rank + 1)
            meta.setdefault(rid, r)
    top = sorted(score, key=lambda i: -score[i])[:limit]
    out = []
    for rid in top:
        r = dict(meta[rid])
        r["rank"] = round(score[rid], 6)
        out.append(r)
    return out


# Nokia domain-vocabulary aliases the small embedding model can't bridge on its
# own (pure-vector "pseudowire" returns PPPoE, never Epipe). Additive query
# expansion that lifts BOTH halves (BM25 exact-term recall + embedding concept).
# Curated in a maintainable data file so it can be tuned without touching code;
# override the path with DOCS_SYNONYMS. Missing/bad file → no expansion (safe).
_SYN_FILE = os.environ.get(
    "DOCS_SYNONYMS", os.path.join(os.path.dirname(__file__), "synonyms.json"))


def _load_synonyms() -> dict:
    try:
        import json
        with open(_SYN_FILE, encoding="utf-8") as f:
            return {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    except Exception as e:  # noqa: BLE001
        print(f"[docs] synonyms unavailable ({_SYN_FILE}): {type(e).__name__}: {e}",
              file=sys.stderr)
        return {}


_SYNONYMS = _load_synonyms()


def _expand(query: str) -> str:
    """Append Nokia-native synonyms for any domain term in the query (additive)."""
    ql = query.lower()
    extra = [exp for term, exp in _SYNONYMS.items() if term in ql]
    return query + " " + " ".join(extra) if extra else query


def _cite(row: dict) -> str:
    title = row.get("title") or row.get("guide_slug")
    page = row.get("page_no")
    pg = f", p.{page}" if page else ""
    return f"{title}{pg}"


def _bm25_rows(schema: str, query: str, guide: str, cli_mode: str,
               mgmt_domain: str, limit: int) -> list[dict]:
    """ParadeDB BM25 ranking over heading/topic/body (heading boosted) within one
    release schema. True IDF + length normalization. Returns final-ordered rows."""
    clauses = ""
    # heading^3, topic^2, body — boost titles without losing body recall.
    args: list = [query, query, query]
    if guide:
        clauses += " AND c.guide_slug = %s"
        args.append(guide)
    if cli_mode:
        clauses += " AND c.cli_mode = %s"
        args.append(cli_mode)
    if mgmt_domain:
        clauses += " AND c.mgmt_domain = %s"
        args.append(mgmt_domain)
    args.append(limit)
    sql = f"""
        SELECT c.id, c.guide_slug, dg.title, c.page_no, c.heading, c.body,
               c.deep_url, c.topic, c.version, c.cli_mode, c.mgmt_domain,
               c.context_tree, dg.landing_url, dg.pdf_url,
               paradedb.score(c.id) AS rank
        FROM doc_chunk c
        JOIN doc_guide dg ON dg.slug = c.guide_slug AND dg.version = c.version
        WHERE c.id @@@ paradedb.disjunction_max(disjuncts => ARRAY[
                  paradedb.boost(3.0, paradedb.match('heading', %s)),
                  paradedb.boost(2.0, paradedb.match('topic',   %s)),
                  paradedb.match('body', %s)]){clauses}
        ORDER BY rank DESC, c.page_no
        LIMIT %s
    """
    return _q(sql, tuple(args), schema=schema)


@mcp.tool()
def docs_search(query: str, product: str = "all", version: str = "",
                guide: str = "", limit: int = 8, cli_mode: str = "",
                mgmt_domain: str = "") -> dict:
    """Hybrid-search the ingested docs; returns ranked passages + citations.

    query       : natural words or exact tokens (CLI verbs, REST paths, alarms).
    product     : doc family — "nsp", "sros", or "all" (default, both).
    version     : pin an exact doc version (overrides `product`).
    guide       : optional guide slug to restrict to.
    limit       : max passages (default 8).
    cli_mode    : SR OS CLI filter — "md-cli" or "classic-cli" (stored in
                  separate fields). Empty = both + all prose.
    mgmt_domain : management-plane PAIRING filter — "classic" pairs SR OS
                  classic-CLI with NSP NFM-P (the "Classic Management" User_Guide);
                  "model-driven" pairs SR OS MD-CLI with NSP MDM. Use this to get
                  BOTH sides of a management plane in one search. Empty = all.

    Ranking: ParadeDB BM25 fused with pgvector cosine NN via Reciprocal Rank
    Fusion (ranking="hybrid"); falls back to BM25 alone if the vector half is
    unavailable (ranking="bm25"). The query is expanded with Nokia-native
    synonyms first (additive).
    """
    limit = max(1, min(int(limit), 25))
    releases = _resolve(product, version)
    versions_out = [r["version"] for r in releases]
    cli_mode = (cli_mode or "").strip().lower()
    if cli_mode and cli_mode not in ("md-cli", "classic-cli"):
        cli_mode = ""  # ignore junk filter rather than zero out results
    mgmt_domain = (mgmt_domain or "").strip().lower()
    if mgmt_domain and mgmt_domain not in ("classic", "model-driven"):
        mgmt_domain = ""
    if not re.search(r"[A-Za-z0-9]", query or ""):
        return {"query": query, "versions": versions_out, "count": 0, "results": [],
                "note": "empty/unsearchable query"}
    if not releases:
        return {"query": query, "versions": [], "count": 0, "results": [],
                "note": "no matching release is seeded — see docs_list_versions"}
    if not _has_bm25():
        return {"query": query, "versions": versions_out, "count": 0, "results": [],
                "note": "docs-db has no pg_search (BM25) — check the ParadeDB store"}

    # Per-release candidate pool → fuse (BM25+vector) → merge across releases →
    # cross-encoder rerank the union → top `limit`. Pool is split across releases
    # so the rerank input stays tight (a big pool feeds distractors).
    full = min(max(limit * RERANK_POOL, 12), RERANK_POOL_MAX) if RERANK_ENABLED else limit
    per = full if len(releases) == 1 else max(full // len(releases) + 2, 6)
    cand = max(per, limit * 3, 24)
    eq = _expand(query)   # query + Nokia-native synonyms (additive; both halves)
    qvec = _embed_query(eq) if _has_vectors() else None

    fused_all: list[dict] = []
    ranking = "bm25"
    for rel in releases:
        sch = rel["schema"]
        try:
            bm25 = _bm25_rows(sch, eq, guide, cli_mode, mgmt_domain, cand)
        except Exception:
            bm25 = []
        vec = []
        if qvec:
            try:
                vec = _vector_rows(sch, qvec, guide, cli_mode, mgmt_domain, cand)
            except Exception:
                vec = []
        if bm25 and vec:
            part, ranking = _rrf([bm25, vec], per), "hybrid"
        else:
            part = bm25[:per]
        for r in part:
            r["_release"] = rel["name"]
        fused_all.extend(part)

    if RERANK_ENABLED and fused_all and _get_reranker() is not None:
        rows = _rerank_ce(query, fused_all, limit)   # reranks on the raw question
        ranking += "+rerank"
    else:
        rows = sorted(fused_all, key=lambda r: -float(r.get("rank") or 0))[:limit]

    results = []
    for r in rows:
        snippet = r["body"]
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
        results.append({
            "chunk_id": r["id"],
            "citation": _cite(r),
            "release": r.get("_release"),
            "guide": r["guide_slug"],
            "version": r.get("version"),
            "cli_mode": r.get("cli_mode"),
            "mgmt_domain": r.get("mgmt_domain"),
            "topic": r.get("topic"),
            "page": r["page_no"],
            "heading": r["heading"],
            "rank": round(float(r["rank"]), 5),
            "snippet": snippet,
            "url": r.get("deep_url") or r["landing_url"] or r["pdf_url"],
        })
    return {"query": query, "releases": [r["name"] for r in releases],
            "versions": versions_out, "guide": guide or None,
            "cli_mode": cli_mode or None, "mgmt_domain": mgmt_domain or None,
            "ranking": ranking, "count": len(results), "results": results}


def _schema_for_chunk(chunk_id: int) -> tuple:
    """Find which seeded release schema holds a chunk id (ids are globally unique)."""
    for rel in _resolve():
        got = _q("SELECT guide_slug, version, seq FROM doc_chunk WHERE id=%s",
                 (chunk_id,), schema=rel["schema"])
        if got:
            return rel["schema"], got[0]
    return None, None


@mcp.tool()
def docs_get_chunk(chunk_id: int, context: int = 1) -> dict:
    """Return a passage's full text plus `context` neighbouring chunks for more
    surrounding detail. Use after docs.search when a snippet looks relevant."""
    context = max(0, min(int(context), 5))
    schema, b = _schema_for_chunk(chunk_id)
    if not b:
        return {"error": f"chunk {chunk_id} not found"}
    rows = _q(
        """SELECT c.id, c.page_no, c.seq, c.heading, c.body, c.deep_url,
                  c.topic, dg.title, dg.landing_url, dg.pdf_url
           FROM doc_chunk c JOIN doc_guide dg ON dg.slug = c.guide_slug
           WHERE c.guide_slug=%s AND c.version=%s AND c.seq BETWEEN %s AND %s
           ORDER BY c.seq""",
        (b["guide_slug"], b["version"], b["seq"] - context, b["seq"] + context),
        schema=schema,
    )
    # Anchor the citation/URL on the requested chunk itself, not just rows[0].
    focus = next((r for r in rows if r["id"] == chunk_id), rows[0])
    return {
        "chunk_id": chunk_id,
        "guide": b["guide_slug"],
        "topic": focus.get("topic"),
        "citation": _cite({**focus, "guide_slug": b["guide_slug"],
                           "page_no": focus["page_no"]}),
        "url": focus.get("deep_url") or focus["landing_url"] or focus["pdf_url"],
        "passages": [{"chunk_id": r["id"], "page": r["page_no"],
                      "topic": r.get("topic"), "heading": r["heading"],
                      "body": r["body"], "url": r.get("deep_url")}
                     for r in rows],
    }


@mcp.tool()
def docs_list_guides(product: str = "all", version: str = "") -> dict:
    """List ingested guides (slug, title, version, pages, chunks) across the
    matching releases. product : "nsp"/"sros"/"all"; version pins an exact one.
    """
    guides = []
    for rel in _resolve(product, version):
        rows = _q(
            """SELECT slug, title, version, n_pages, n_chunks, landing_url,
                      fetched_at::text AS fetched_at
               FROM doc_guide ORDER BY slug""", schema=rel["schema"])
        for r in rows:
            r["release"] = rel["name"]
        guides.extend(rows)
    return {"releases": [r["name"] for r in _resolve(product, version)],
            "count": len(guides), "guides": guides}


@mcp.tool()
def docs_list_versions() -> dict:
    """List the product-doc releases this server has built and can search
    (NSP + SR OS versions), plus which the query defaults to."""
    seeded = _seeded_schemas()
    out = []
    for r in _registry():
        out.append({"name": r["name"], "product": r["product"],
                    "version": r["version"], "schema": r["schema"],
                    "ready": r["schema"] in seeded})
    return {"count": sum(1 for r in out if r["ready"]), "releases": out}


def main() -> int:
    ap = argparse.ArgumentParser(description="mcp local-docs MCP server")
    ap.add_argument("transport", nargs="?", default="stdio",
                    choices=["stdio", "sse", "http"])
    ap.add_argument("--port", type=int, default=9705)
    args = ap.parse_args()

    # Report which release schemas are seeded and ready at startup.
    try:
        rel = _resolve()
        total = 0
        for r in rel:
            total += _q("SELECT count(*) AS n FROM doc_chunk", schema=r["schema"])[0]["n"]
        names = ", ".join(r["name"] for r in rel) or "none"
        print(f"[mcp-docs-mcp] ready — releases: {names} ({total} chunks)", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[mcp-docs-mcp] WARNING: docs-db not ready: {e}", file=sys.stderr)

    if args.transport in ("sse", "http"):
        mcp.settings.port = args.port
        mcp.settings.host = "0.0.0.0"
        from mcp.server.transport_security import TransportSecuritySettings
        mcp.settings.transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=False)
        _t = "streamable-http" if args.transport == "http" else args.transport
        print(f"[mcp-docs-mcp] {_t} on 0.0.0.0:{args.port}", file=sys.stderr)
        mcp.run(transport=_t)
    else:
        mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
