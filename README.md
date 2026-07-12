<div align="center">

![mcp-docs banner](docs/assets/banner.png)

# mcp-docs

### Intelligent, fully-offline documentation search for Nokia **NSP** & **SR OS**, in a single container.

Ask a real engineering question. Get the exact passage back, ranked by a purpose-built
retrieval stack, with a deep-link citation to the precise page and heading. **With zero
internet, zero credentials, and one `docker run`.**

![offline](https://img.shields.io/badge/100%25-offline%20%2F%20air--gapped-001135?style=for-the-badge)
![single container](https://img.shields.io/badge/single-container-005AFF?style=for-the-badge)
![MCP](https://img.shields.io/badge/MCP-native-7D33F2?style=for-the-badge)

![search](https://img.shields.io/badge/search-BM25%20%2B%20vector%20%2B%20reranker-23ABB6?style=flat-square)
![multi-version](https://img.shields.io/badge/multi--release-schema%20per%20version-37CC73?style=flat-square)
![citations](https://img.shields.io/badge/every%20hit-deep--link%20cited-005AFF?style=flat-square)
![backend](https://img.shields.io/badge/engine-ParadeDB-001135?style=flat-square)

</div>

---

> [!WARNING]
> **Experimental proof-of-concept: not an official Nokia product.** A personal/community
> experiment for evaluating MCP-based documentation search. Not built, supported, or endorsed
> by Nokia; no warranty; not for production. Nokia product names/docs are the property of Nokia,
> referenced here for interoperability and evaluation only.

**mcp-docs** turns the sprawling Nokia NSP and SR OS documentation set into a single, intelligent,
air-gapped search endpoint that AI agents and MCP-aware IDEs can ground their answers on. The
corpus, the search engine, and the AI models all live **inside one container**. Hand it to anyone,
run it on a laptop or a locked-down NOC, and it just works.

---

## Why it's different

| | |
|---|---|
| **Completely offline** | Corpus, embedding model, reranker, and database are all baked in. Runs `--network none`, air-gapped. No internet, no Hugging Face, no NSP connection, no credentials. |
| **One self-contained container** | ParadeDB + the MCP server + models + docs in a single image. `docker run` → ready. Nothing else to stand up. |
| **Multi-release, on demand** | Each product release lives in its own schema (`nsp_2604`, `sros_263`, …), built at boot from its own optimized dump. Pick what you want with `RELEASES=`; defaults to **NSP 26.4 + SR OS 26.3**. Add a release = drop a dump + one registry line. |
| **Genuinely intelligent search** | Not keyword grep. A hybrid **BM25 + semantic-vector** retriever fused by Reciprocal Rank Fusion, re-ordered by a **cross-encoder reranker**, with Nokia-aware query expansion and conceptual-vs-CLI routing. |
| **Deep-link citations** | Every hit comes back with a **section-precise `deep_url`** that goes straight to the exact page, heading, or CLI command. |
| **Measured, not vibes** | The retrieval stack was A/B-tested on deep, multi-domain engineer questions and LLM-judged: **+12 quality points** over plain hybrid search. |
| **MCP-native** | Four clean tools any MCP client (Claude Code, IDEs, agents) can call directly. |

---

## Quick start

```bash
docker run -p 9705:9705 mcp-docs          # defaults: NSP 26.4 + SR OS 26.3
```

On boot it starts its internal database, imports the requested releases into per-version schemas,
builds the search indexes, and serves the MCP endpoint at **`http://<host>:9705/mcp`**. Air-gapped?

```bash
docker run --network none -p 9705:9705 mcp-docs     # fully offline
```

Only want one release, or want to add more:

```bash
docker run -p 9705:9705 -e RELEASES="nsp-26.4" mcp-docs            # just NSP 26.4
docker run -p 9705:9705 -e RELEASES="nsp-26.4 sros-26.3" mcp-docs  # both (default)
```

Build it yourself: `make up` (or `docker build -f docker/Dockerfile -t mcp-docs .`).

---

## How it works: from raw docs to intelligent answers

**Ingest pipeline (offline, done once per release → ships as an optimized dump):**

```
 Nokia docs ──▶ parse & chunk ──▶ classify & deep-link ──▶ embed & index ──▶ optimized dump
               section-aware       tag every chunk with       BM25 (ParadeDB)   per-release
               passages that       page, heading, CLI mode,    + vector (HNSW)   .sql.gz, ready
               keep their context  mgmt domain, deep URL       baked in          to load at runtime
```

Each chunk keeps *where it came from*: page, heading, whether it's MD-CLI vs classic-CLI, which
management plane it belongs to, and a link straight back to the source. That metadata is what makes
the answers precise and citable.

**Search pipeline (at query time, in the container):**

```
 your question ──▶ Nokia synonym expansion ──▶ ┌ BM25 (exact terms) ┐ ──▶ RRF fuse ──▶ cross-encoder ──▶ CLI-vs-concept ──▶ cited
                    (pseudowire → Epipe…)        └ vector (meaning)   ┘                 rerank            routing            answer
```

- **BM25** nails the exact CLI/REST/alarm/YANG tokens these docs are full of.
- **Semantic vectors** catch paraphrases and intent ("bring up a router" → commissioning).
- **Reciprocal Rank Fusion** blends both signals.
- **A cross-encoder reranker** then reads the candidates and re-orders by true relevance, the single
  biggest quality lever.
- **Conceptual-vs-CLI routing** keeps a "why is X slow?" question from being answered by a raw command
  reference.

The result: the *right passage*, ranked first, cited to the exact spot.

---

## Multi-release

One container can hold many NSP and SR OS releases side by side, each isolated in its own schema and
searchable independently or together.

```bash
docker run -p 9705:9705 -e RELEASES="nsp-26.4 nsp-25.11 sros-26.3" mcp-docs
```

Adding a release is a two-line change: drop `data/nsp-25.11.sql.gz` and register it in `versions.yml`:

```yaml
- name: nsp-25.11
  product: nsp
  version: "25.11"
  schema: nsp_2511
  dump: data/nsp-25.11.sql.gz
```

Search then routes automatically: `product="nsp"`, `version="26.4"`, or `"all"` (both families).

---

## Tools

| Tool | What it does |
|------|--------------|
| **`docs_search`** | Hybrid search across the selected release(s). Filter by `product`, `version`, `guide`, `cli_mode`, `mgmt_domain`. Returns ranked passages, each with a citable `deep_url` and its source release. |
| **`docs_get_chunk`** | Fetch a passage plus its neighbours for full surrounding context. |
| **`docs_list_guides`** | Catalogue the ingested guides/books per release. |
| **`docs_list_versions`** | Show which NSP / SR OS releases are built and searchable. |

---

## Connect

Point any MCP client at the streamable-HTTP endpoint:

```json
{
  "mcpServers": {
    "mcp-docs": { "type": "http", "url": "http://<host>:9705/mcp" }
  }
}
```

No credentials, no secrets, no external services. Hand over the container and it just works.

---

## Under the hood

![mcp-docs architecture diagram](docs/assets/architecture.png)

- **Engine:** ParadeDB (Postgres + `pg_search` BM25 + `pgvector` HNSW): top-tier lexical + vector search in one embedded store.
- **Models:** `bge-small` embeddings + `ms-marco-MiniLM-L-12` cross-encoder reranker, both baked in, CPU-only (~19 ms + rerank).
- **Portable:** everything is bundled and runs on CPU: no GPU, no cloud, no network.

<sub>Optional roadmap: a live-NSP mode that auto-detects your running release and routes docs to it (requires network + a read-only token), kept strictly optional so the default stays fully offline.</sub>

---

<div align="center">
<sub>Section-precise, intelligent hybrid search over the Nokia NSP &amp; SR OS documentation set: offline, in one container.</sub>

<sub>Experimental proof-of-concept · not an official Nokia product · no warranty · provided as-is.</sub>
</div>
