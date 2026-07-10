#!/usr/bin/env python3
"""server.py — mcp local-docs MCP server.

Full-text search over Nokia product documentation ingested into the entitydb
doc store (migration 013, populated by scripts/docs/ingest_nsp_docs.py).

Retrieval is Postgres-native FTS (tsvector + GIN) ranked with ts_rank_cd —
precise on the exact CLI/REST/alarm/YANG tokens these docs are full of, and
zero new infra (entitydb is the lab's existing postgres:16-alpine). No vectors,
no embeddings, no second datastore.

Standalone server — it does NOT need NSP credentials (unlike the nsp MCP), only
a read connection to entitydb. Runs host-side so Claude Code can launch it over
stdio:

    python3 server.py            # stdio (default — for Claude Code)
    python3 server.py sse        # HTTP/SSE on port 9710
    python3 server.py sse --port N

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
EMBED_MODEL = os.environ.get("DOCS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_CACHE = os.environ.get("DOCS_EMBED_CACHE", "/app/shared/models")
HYBRID_ENABLED = os.environ.get("DOCS_HYBRID", "1") not in ("0", "false", "")
# mxbai/bge retrieval query instruction (passages embedded without it at ingest).
QUERY_PROMPT = "Represent this sentence for searching relevant passages: "

DSN = os.environ.get(
    "DOCS_DATABASE_URL",
    # Read-only role by default; falls back to mcp/mcp if _ro absent.
    os.environ.get("DATABASE_URL", "postgresql://mcp:mcp@localhost:5432/aggregation_db"),
)
# The doc store holds two product families, distinguished by their version
# string: NSP guides (e.g. 26-4) and SROS/TiMOS CLI references (e.g. 26-3).
# Both are configurable; by default search resolves to BOTH.
DOCS_NSP_VERSION = os.environ.get("DOCS_NSP_VERSION") or os.environ.get("DOCS_VERSION", "26-4")
DOCS_SROS_VERSION = os.environ.get("DOCS_SROS_VERSION", "26-3")


def _versions(version: str = "", product: str = "all") -> list[str]:
    """Resolve which doc versions to query. Explicit `version` wins; otherwise
    `product` ∈ {nsp, sros, all} maps to the configured version(s)."""
    if version:
        return [version]
    p = (product or "all").strip().lower()
    if p == "nsp":
        return [DOCS_NSP_VERSION]
    if p == "sros":
        return [DOCS_SROS_VERSION]
    return [DOCS_NSP_VERSION, DOCS_SROS_VERSION]


mcp = FastMCP("mcp-docs-mcp", host="0.0.0.0", port=9710)


def _q(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT against entitydb with a short bounded retry."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            with psycopg.connect(DSN, connect_timeout=5, row_factory=dict_row) as c:
                with c.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"entitydb query failed: {type(last).__name__}: {last}")


_BM25: bool | None = None


def _has_bm25() -> bool:
    """True if the doc store runs ParadeDB pg_search (BM25). Detected once and
    cached; when present, search uses BM25 (true IDF + length normalization)
    instead of ts_rank_cd + the lexical reranker."""
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
    """True if the store has populated embeddings + an HNSW index AND the
    fastembed model can be loaded in-process (for query embedding). Cached."""
    global _VEC
    if _VEC is None:
        _VEC = False
        if not HYBRID_ENABLED:
            return False
        try:
            ok = _q("""SELECT 1 FROM doc_chunk WHERE embedding IS NOT NULL LIMIT 1""")
            # NB: avoid a literal '%hnsw%' here — psycopg reads the bare % as a
            # parameter placeholder and raises, which the except below would
            # swallow into _VEC=False (hybrid silently never activates). strpos
            # keeps the check %-free.
            idx = _q("""SELECT 1 FROM pg_indexes
                        WHERE tablename='doc_chunk'
                          AND strpos(lower(indexdef), 'hnsw') > 0""")
            _VEC = bool(ok and idx and _get_embedder() is not None)
        except Exception:
            _VEC = False
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
    so repeated queries skip the ~30 ms CPU embed."""
    emb = _get_embedder()
    if emb is None:
        return None
    vec = next(iter(emb.embed([QUERY_PROMPT + text])))
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def _vector_rows(qvec: str, versions: list, guide: str, cli_mode: str,
                 mgmt_domain: str, limit: int) -> list[dict]:
    """HNSW cosine-NN search (fast: sub-ms ANN over the half-precision index)."""
    clauses = ""
    args: list = [qvec]
    if guide:
        clauses += " AND c.guide_slug = %s"; args.append(guide)
    if cli_mode:
        clauses += " AND c.cli_mode = %s"; args.append(cli_mode)
    if mgmt_domain:
        clauses += " AND c.mgmt_domain = %s"; args.append(mgmt_domain)
    args.append(versions)
    args.append(limit)
    sql = f"""
        SELECT c.id, c.guide_slug, dg.title, c.page_no, c.heading, c.body,
               c.deep_url, c.topic, c.version, c.cli_mode, c.mgmt_domain,
               c.context_tree, dg.landing_url, dg.pdf_url,
               1 - (c.embedding <=> %s::vector) AS rank
        FROM doc_chunk c
        JOIN doc_guide dg ON dg.slug = c.guide_slug AND dg.version = c.version
        WHERE c.embedding IS NOT NULL{clauses}
          AND c.version = ANY(%s)
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """
    # qvec appears twice (select distance + order); rebuild args accordingly.
    a = [qvec]
    tail: list = []
    if guide:
        tail.append(guide)
    if cli_mode:
        tail.append(cli_mode)
    if mgmt_domain:
        tail.append(mgmt_domain)
    return _q(sql, tuple([qvec] + tail + [versions, qvec, limit]))


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


# Nokia domain-vocabulary the small embedding model (bge-small) does NOT bridge
# on its own — pure-vector search for "pseudowire" returns PPPoE/CUPS, never
# Epipe. Each key is a lowercased trigger phrase; its value is a few Nokia-native
# terms appended to the query, lifting both the BM25 half (exact-term recall) and
# the embedding (it shifts toward the right concept). Additive only — never
# removes the user's own terms. Keep expansions short + high-precision.
_SYNONYMS = {
    "pseudowire": "epipe vll spoke-sdp vc-type",
    "pseudo-wire": "epipe vll spoke-sdp",
    "point-to-point service": "epipe vll",
    "point to point service": "epipe vll",
    "l2vpn": "epipe vpls vll",
    "l2 vpn": "epipe vpls vll",
    "layer 2 vpn": "epipe vpls vll",
    "layer-2 vpn": "epipe vpls vll",
    "layer 2 service": "epipe vpls",
    "multipoint service": "vpls",
    "lan service": "vpls",
    "l3vpn": "vprn",
    "l3 vpn": "vprn",
    "layer 3 vpn": "vprn",
    "routing domain": "isis ospf igp",
    "backbone routing": "isis ospf igp",
    "join the backbone": "isis ospf igp adjacency",
    "bring up a new router": "router-id system interface isis ospf commissioning",
    "commission a router": "router-id system interface bof",
    "prefix-sid": "node-sid segment-routing",
    "prefix sid": "node-sid segment-routing",
    "label switched path": "lsp mpls",
}


def _expand(query: str) -> str:
    """Append Nokia-native synonyms for any domain term in the query (additive)."""
    ql = query.lower()
    extra = [exp for term, exp in _SYNONYMS.items() if term in ql]
    return query + " " + " ".join(extra) if extra else query


def _query_terms(query: str) -> tuple[str | None, list[str]]:
    """Return (websearch_passthrough | None, content_tokens).

    Quoted/operator queries pass through to websearch_to_tsquery verbatim;
    otherwise we get the bare content tokens and decide AND-vs-OR in docs_search.
    """
    if '"' in query or " OR " in query or " or " in query:
        toks = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 1]
        return (query, toks)
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 1]
    return (None, toks)


def _rerank(rows: list[dict], qterms: list[str]) -> list[dict]:
    """Lexical rerank over the FTS candidate set. ts_rank_cd has no IDF or
    document-length normalization, so a keyword-stuffed chunk floats. We blend it
    with: distinct query-term COVERAGE (AND-quality precision), a heading/topic
    boost, and a length-normalization proxy that demotes very long bodies (the
    residual density-spam). BM25 (Stage 2) supersedes this; until then it fixes
    the worst mis-rankings cheaply.
    """
    if not qterms:
        return rows
    qset = set(qterms)
    n = len(qset)

    def score(r: dict) -> float:
        base = float(r.get("rank") or 0.0)
        h = (r.get("heading") or "").lower()
        t = (r.get("topic") or "").lower()
        body = r.get("body") or ""
        text = (h + " " + t + " " + body[:3000]).lower()
        covered = sum(1 for w in qset if w in text) / n        # 0..1 term coverage
        head_hit = sum(1 for w in qset if w in h or w in t)     # title relevance
        length_pen = 4000.0 / (4000.0 + len(body))             # ↓ for long bodies
        return base * (0.4 + covered) * (1.0 + 0.4 * head_hit) * (0.6 + 0.4 * length_pen)

    return sorted(rows, key=score, reverse=True)


def _cite(row: dict) -> str:
    title = row.get("title") or row.get("guide_slug")
    page = row.get("page_no")
    pg = f", p.{page}" if page else ""
    return f"{title}{pg}"


def _bm25_rows(query: str, versions: list, guide: str, cli_mode: str,
               mgmt_domain: str, limit: int) -> list[dict]:
    """ParadeDB BM25 ranking over heading/topic/body (heading boosted). True IDF
    + length normalization — a keyword-stuffed chunk (e.g. `shutdown`) no longer
    floats on a single repeated term. Returns final-ordered rows (no rerank)."""
    clauses = ""
    # heading^3, topic^2, body — boost titles without losing body recall.
    args: list = [query, query, query]
    args.append(versions)
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
        WHERE c.id @@@ paradedb.boolean(should => ARRAY[
                  paradedb.boost(3.0, paradedb.match('heading', %s)),
                  paradedb.boost(2.0, paradedb.match('topic',   %s)),
                  paradedb.match('body', %s)])
          AND c.version = ANY(%s){clauses}
        ORDER BY rank DESC, c.page_no
        LIMIT %s
    """
    return _q(sql, tuple(args))


def _fts_rows(fn: str, qtext: str, versions: list, guide: str,
              cli_mode: str, mgmt_domain: str, cand_limit: int) -> list[dict]:
    """Run one FTS pass and return up to cand_limit candidate rows (pre-rerank)."""
    clauses = ""
    args: list = [qtext, versions]
    if guide:
        clauses += " AND c.guide_slug = %s"
        args.append(guide)
    if cli_mode:
        clauses += " AND c.cli_mode = %s"
        args.append(cli_mode)
    if mgmt_domain:
        clauses += " AND c.mgmt_domain = %s"
        args.append(mgmt_domain)
    args.append(cand_limit)
    sql = f"""
        WITH g AS (SELECT {fn}('english', %s) AS q)
        SELECT c.id, c.guide_slug, dg.title, c.page_no, c.heading, c.body,
               c.deep_url, c.topic, c.version, c.cli_mode, c.mgmt_domain,
               c.context_tree, dg.landing_url, dg.pdf_url,
               ts_rank_cd(c.tsv, g.q) AS rank
        FROM doc_chunk c
        JOIN g ON TRUE
        JOIN doc_guide dg ON dg.slug = c.guide_slug AND dg.version = c.version
        WHERE c.version = ANY(%s)
          AND g.q IS NOT NULL AND numnode(g.q) > 0
          AND c.tsv @@ g.q{clauses}
        ORDER BY rank DESC, c.page_no
        LIMIT %s
    """
    return _q(sql, tuple(args))


@mcp.tool()
def docs_search(query: str, product: str = "all", version: str = "",
                guide: str = "", limit: int = 8, cli_mode: str = "",
                mgmt_domain: str = "") -> dict:
    """Full-text search the ingested docs; returns ranked passages + citations.

    query       : natural words or exact tokens (CLI verbs, REST paths, alarms).
                  Quote a phrase or use OR/-term for websearch semantics.
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

    Ranking: AND-first (precise) with OR fallback (recall), then a lexical rerank
    (term-coverage + heading boost + length normalization) over the candidates.
    """
    limit = max(1, min(int(limit), 25))
    versions = _versions(version, product)
    cli_mode = (cli_mode or "").strip().lower()
    if cli_mode and cli_mode not in ("md-cli", "classic-cli"):
        cli_mode = ""  # ignore junk filter rather than zero out results
    mgmt_domain = (mgmt_domain or "").strip().lower()
    if mgmt_domain and mgmt_domain not in ("classic", "model-driven"):
        mgmt_domain = ""
    web, toks = _query_terms(query)
    if web is None and not toks:
        return {"query": query, "versions": versions, "guide": guide or None,
                "count": 0, "results": [], "note": "empty/unsearchable query"}

    ranking = "fts"
    rows: list[dict] = []
    eq = _expand(query)   # query + Nokia-native synonyms (additive; for both halves)
    if _has_bm25():
        # ParadeDB BM25 — true IDF + length normalization.
        cand = max(limit * 3, 24)
        try:
            bm25 = _bm25_rows(eq, versions, guide, cli_mode, mgmt_domain, cand)
        except Exception:
            bm25 = []
        # Semantic hybrid: fuse BM25 with vector cosine (RRF) when embeddings +
        # the in-container model are available — adds paraphrase/synonym recall.
        if _has_vectors():
            qvec = _embed_query(eq)
            vec = []
            if qvec:
                try:
                    vec = _vector_rows(qvec, versions, guide, cli_mode, mgmt_domain, cand)
                except Exception:
                    vec = []
            if bm25 and vec:
                rows = _rrf([bm25, vec], limit)
                ranking = "hybrid"
            elif bm25:
                rows, ranking = bm25[:limit], "bm25"
        elif bm25:
            rows, ranking = bm25[:limit], "bm25"

    if not rows:
        ranking = "fts"
        cand_limit = max(limit * 5, 40)
        if web is not None:
            rows = _fts_rows("websearch_to_tsquery", web, versions, guide,
                             cli_mode, mgmt_domain, cand_limit)
        else:
            # AND-first for precision; fall back to OR for recall if too few hits.
            rows = _fts_rows("to_tsquery", " & ".join(toks), versions, guide,
                             cli_mode, mgmt_domain, cand_limit)
            if len(rows) < min(limit, 5):
                rows = _fts_rows("to_tsquery", " | ".join(toks), versions, guide,
                                 cli_mode, mgmt_domain, cand_limit)
        rows = _rerank(rows, toks)[:limit]
    results = []
    for r in rows:
        snippet = r["body"]
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
        results.append({
            "chunk_id": r["id"],
            "citation": _cite(r),
            "guide": r["guide_slug"],
            "version": r.get("version"),
            "cli_mode": r.get("cli_mode"),
            "mgmt_domain": r.get("mgmt_domain"),
            "topic": r.get("topic"),
            "page": r["page_no"],
            "heading": r["heading"],
            "rank": round(float(r["rank"]), 5),
            "snippet": snippet,
            # Prefer the exact per-topic deep-link; fall back to the guide
            # landing page, then the source PDF, for legacy PDF-ingested rows.
            "url": r.get("deep_url") or r["landing_url"] or r["pdf_url"],
        })
    return {"query": query, "versions": versions, "guide": guide or None,
            "cli_mode": cli_mode or None, "mgmt_domain": mgmt_domain or None,
            "ranking": ranking, "count": len(results), "results": results}


@mcp.tool()
def docs_get_chunk(chunk_id: int, context: int = 1) -> dict:
    """Return a passage's full text plus `context` neighbouring chunks for more
    surrounding detail. Use after docs.search when a snippet looks relevant."""
    context = max(0, min(int(context), 5))
    base = _q("SELECT guide_slug, version, seq FROM doc_chunk WHERE id=%s",
              (chunk_id,))
    if not base:
        return {"error": f"chunk {chunk_id} not found"}
    b = base[0]
    rows = _q(
        """SELECT c.id, c.page_no, c.seq, c.heading, c.body, c.deep_url,
                  c.topic, dg.title, dg.landing_url, dg.pdf_url
           FROM doc_chunk c JOIN doc_guide dg ON dg.slug = c.guide_slug
           WHERE c.guide_slug=%s AND c.version=%s AND c.seq BETWEEN %s AND %s
           ORDER BY c.seq""",
        (b["guide_slug"], b["version"], b["seq"] - context, b["seq"] + context),
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
    """List ingested guides (slug, title, version, pages, chunks, when fetched).

    product : "nsp", "sros", or "all" (default). `version` pins an exact one.
    """
    versions = _versions(version, product)
    rows = _q(
        """SELECT slug, title, version, n_pages, n_chunks, landing_url,
                  fetched_at::text AS fetched_at
           FROM doc_guide WHERE version = ANY(%s) ORDER BY version, slug""",
        (versions,))
    return {"versions": versions, "count": len(rows), "guides": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description="mcp local-docs MCP server")
    ap.add_argument("transport", nargs="?", default="stdio",
                    choices=["stdio", "sse", "http"])
    ap.add_argument("--port", type=int, default=9710)
    args = ap.parse_args()

    # Fail fast on a bad DB connection so misconfig surfaces at startup.
    try:
        n = _q("SELECT count(*) AS n FROM doc_chunk")[0]["n"]
        print(f"[mcp-docs-mcp] entitydb ok — {n} chunks indexed", file=sys.stderr)
    except Exception as e:  # noqa: BLE001
        print(f"[mcp-docs-mcp] WARNING: entitydb not ready: {e}", file=sys.stderr)

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
