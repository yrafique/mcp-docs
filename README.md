<div align="center">

![mcp-docs banner](docs/assets/banner.png)

**Hybrid BM25 + vector documentation search with deep-link citations.**

[![Quick Start](https://img.shields.io/badge/%E2%96%B6%20Quick%20Start-005AFF?style=for-the-badge)](#-quick-start)
[![Architecture](https://img.shields.io/badge/%F0%9F%A7%AD%20Architecture-001135?style=for-the-badge)](#-architecture)
[![Tools](https://img.shields.io/badge/%F0%9F%A7%B0%20Tools-7D33F2?style=for-the-badge)](#-tools)
[![Connect](https://img.shields.io/badge/%F0%9F%94%8C%20Connect-23ABB6?style=for-the-badge)](#-connect)
[![Configuration](https://img.shields.io/badge/%E2%9A%99%EF%B8%8F%20Configuration-37CC73?style=for-the-badge)](#%EF%B8%8F-configuration)

![transport](https://img.shields.io/badge/transport-streamable--HTTP-005AFF?style=flat-square)
![port](https://img.shields.io/badge/port-9705-23ABB6?style=flat-square)
![ranking](https://img.shields.io/badge/ranking-BM25%20%2B%20vector%20RRF-7D33F2?style=flat-square)
![backend](https://img.shields.io/badge/backend-ParadeDB-37CC73?style=flat-square)
![Nokia](https://img.shields.io/badge/Nokia-2026-001135?style=flat-square&logo=nokia&logoColor=white)

</div>

---

> [!WARNING]
> **Experimental proof-of-concept — not an official Nokia product.** This is a personal/community experiment for evaluating MCP-based documentation search. It is **not** built, supported, endorsed, or maintained by Nokia, carries **no warranty**, and is not covered by any Nokia support agreement. Nokia product names and documentation are the property of Nokia and are referenced here for interoperability and evaluation only. Do not use it in production. Use at your own risk.

**mcp-docs** is a Model Context Protocol server that gives AI agents and MCP-aware IDEs a grounded, citable view of Nokia product documentation — NSP guides and SR OS books. It fuses BM25 lexical ranking (ParadeDB) with pgvector semantic nearest-neighbour search via Reciprocal Rank Fusion, and returns a **section- and step-precise `deep_url`** with every hit, so any answer cites straight back to the exact page, heading, or step. The corpus and the embedding model ship inside the stack, no upstream credentials, no live network dependency.

---

## Quick start

**Runs anywhere Docker does — Linux, macOS, and Windows.** The whole stack is containerized (the scripts execute inside Linux containers), so the host OS never matters. You only need three tools on the host:

| Tool | Linux | macOS | Windows |
|------|-------|-------|---------|
| **Docker + Compose v2** | Docker Engine | Docker Desktop | Docker Desktop (WSL 2 backend) |
| **Git** | `dnf`/`apt install git` | `brew install git` | Git for Windows |
| **Git LFS** | `dnf`/`apt install git-lfs` | `brew install git-lfs` | bundled with Git for Windows |

Clone and run:

```bash
git lfs install                                                    # once per machine
git clone https://github.com/yrafique/mcp-docs.git && cd mcp-docs
make up          # or, without make: git lfs pull && docker compose up -d --build
```

`make up` runs `git lfs pull` to fetch the seed dump (corpus **plus** pre-computed `bge-small` embeddings, ~133 MB, stored via Git LFS) then brings the stack up. Because the vectors ship inside the dump, the docs-db comes up **hybrid-ready** and the one-shot embed step is a near-instant no-op.

> **No `make`?** It's only a shortcut. On Windows (or any host without `make`) run the two commands directly: `git lfs pull` then `docker compose up -d --build`. Stop with `docker compose down`, follow logs with `docker compose logs -f mcp-docs`.
>
> **Git LFS is required.** The corpus lives in Git LFS — run `git lfs install` once per machine *before* cloning. Without it the clone pulls only a small pointer file and the db seeds empty.
>
> **Windows line endings.** `.gitattributes` pins all scripts to LF, so a Windows checkout keeps them container-safe. Just leave Git's `core.autocrlf` at its default.

The corpus, vectors, and model persist on named volumes, so every later `up` is instant. Once up, the server speaks streamable-HTTP at:

```
http://<host>:9705/mcp
```

---

## Architecture

![mcp-docs architecture diagram](docs/assets/architecture.png)

Docs are ingested and chunked, each chunk keeps its deep-link, page, heading, `cli_mode`, and `mgmt_domain`, into a ParadeDB store. An idempotent embedding backfill fills the 384-dim `bge-small` vectors and builds an HNSW (cosine) index. Queries are answered by a 3-tier ranker and exposed through three tools, each result reporting the `ranking` tier it came from.

| Tier | What it does |
|------|--------------|
| **`hybrid`** | Reciprocal-Rank-Fusion of ParadeDB **BM25** + **pgvector** cosine NN (`bge-small`, HNSW) with Nokia domain-synonym query expansion (pseudowire→Epipe, routing-domain→IGP…) |
| **`bm25`** | ParadeDB BM25 only, fallback before vectors are built |
| **`fts`** | Plain Postgres full-text (`ts_rank_cd` + lexical rerank), final fallback |

---

## Tools

| Family | Key tools | What |
|--------|-----------|------|
| **Search** | `docs_search` | Hybrid / BM25 / FTS search across the corpus. Filter by `product` (`nsp`/`sros`), `version`, `guide`, `cli_mode` (`md-cli`/`classic-cli`), and `mgmt_domain` (`classic`/`model-driven`) to pair both sides of a management plane. Returns ranked snippets + a citable `deep_url`. |
| **Catalogue** | `docs_list_guides` | List the ingested guides and books with their slug, title, version, page count, chunk count, and fetch time. |
| **Context** | `docs_get_chunk` | Fetch one chunk by id together with its neighbouring chunks for the full surrounding context, used after a `docs_search` hit looks relevant. |

---

## Connect

Point your MCP client at the streamable-HTTP endpoint. This repo serves a single connection:

```json
{
  "mcpServers": {
    "mcp-docs": {
      "type": "http",
      "url": "http://<host>:9705/mcp"
    }
  }
}
```

No credentials and no secrets, the docs-db uses generic local defaults on an internal compose network, and the documentation corpus + embedding model are self-contained. Hand the whole repo to anyone and `docker compose up -d --build` just works.

---

## Configuration

Runtime config is read from `.env`. Copy `.env.example` to `.env` and adjust as needed, all values below have safe defaults for the bundled standalone stack.

| Variable | Purpose | Example |
|----------|---------|---------|
| `DOCS_DATABASE_URL` | DSN of the ParadeDB docs-db (defaults to the bundled `docs-db` service) | `postgresql://mcp:mcp@docs-db:5432/docs` |
| `DOCS_NSP_VERSION` | NSP documentation version to resolve for `product="nsp"` | `26-4` |
| `DOCS_SROS_VERSION` | SR OS / TiMOS documentation version for `product="sros"` | `26-3` |
| `DOCS_HYBRID` | Enable the vector half of the hybrid (`0`/`false` = BM25/FTS only) | `1` |
| `DOCS_EMBED_MODEL` | Query-side embedding model (must match the corpus vectors) | `BAAI/bge-small-en-v1.5` |
| `DOCS_EMBED_CACHE` | Where the query-side embedding model is cached in-container (baked into the image at build time) | `/app/shared/models` |

Never commit a filled-in `.env`. No real secrets are needed for the bundled stack; if you point `DOCS_DATABASE_URL` at an external ParadeDB, seed it first from `src/doc_store.sql.gz` and supply your own credentials there.

---

## Layout

```
mcp-docs/
├── Makefile               # make up / make down / make logs
├── compose.yml            # the stack: docs-db + docs-embed + mcp-docs
├── Dockerfile             # MCP server image
├── requirements.txt       # server Python dependencies
├── .env.example           # copy to .env and adjust
├── src/
│   ├── doc_store.sql.gz   # Git LFS: corpus + bge-small embeddings (~133 MB, hybrid-ready)
│   └── server.py          # the MCP server — search / list-guides / get-chunk tools
├── seed/
│   ├── Dockerfile         # one-shot embed/index image (docs-embed service)
│   ├── seed-docs-db.sh    # first-init DB seed: load dump, enable BM25 + vector, index
│   ├── embed_backfill.py  # fills any NULL embeddings + builds HNSW (idempotent)
│   └── run-embed.sh       # waits for docs-db, then runs the backfill
└── README.md
```

The three stack services:

| Service | Role |
|---------|------|
| `docs-db` | ParadeDB; auto-seeds the committed corpus `src/doc_store.sql.gz` on first init |
| `docs-embed` | one-shot: fills any **missing** `bge-small` embeddings + builds the HNSW index, then exits. The shipped dump is already embedded, so this is a near-instant no-op; it only does real work (downloading the ~130 MB model, ~10-15 min on CPU) if you swap in a corpus-only dump |
| `mcp-docs` | the MCP server on port `9705`; embeds the query side of the hybrid in-process using the `bge-small` model baked into its image |

---

<div align="center">

<sub>Section-precise hybrid search over the Nokia NSP &amp; SR OS documentation set, with a citable deep_url on every hit.</sub>

<sub>Experimental proof-of-concept · not an official Nokia product · no warranty · provided as-is.</sub>

</div>
