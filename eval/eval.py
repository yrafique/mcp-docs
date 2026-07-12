#!/usr/bin/env python3
"""eval.py — retrieval quality harness for mcp-docs.

Runs a fixed set of deep, multi-domain customer questions through docs_search and
scores the RETURNED ranking against a heuristic gold (expected guide(s) + key
concepts). No human labels / no external LLM — the gold is a stable yardstick so
each improvement (reranker, routing, embedder) is measured on the same ruler.

It measures docs_search itself, toggling server-module flags between configs, so
whatever the server actually does in production is what gets scored.

Metrics over all questions (K=5 unless noted):
  Hit@5  — fraction with a relevant hit in the top 5
  P@5    — mean precision in the top 5
  MRR    — mean 1/rank of the first relevant hit (top 10)

Run inside the mcp-docs container:
  cat eval/eval.py | docker compose -f docker/compose.yml exec -T mcp-docs python3 -
"""
import sys
import time

sys.path.insert(0, "/srv/src")
import server

K = 5

# (query, gold_guides, gold_terms). gold_guides = guide slugs that should hold the
# closest material; gold_terms = concepts a truly relevant passage should mention.
QUESTIONS = [
 ("At 30K+ NEs, how do I split the network across Assurance Maps regions without a hard object limit becoming a soft performance cliff?",
  ["NSP_Planning_Guide","NSP_Network_and_Service_Assurance_Guide","User_Guide"], ["map","region","scale","object","telemetry"]),
 ("What is the MDM scale ceiling in NSP and what drives it?",
  ["NSP_Planning_Guide","NSP_Device_Management_Guide"], ["mdm","scale","mediation","node"]),
 ("What sizing and scale characterization exists for a standalone CLM split at 5K+ nodes?",
  ["NSP_Planning_Guide","NSP_Installation_and_Upgrade_Guide"], ["clm","scale","node","sizing"]),
 ("How do I size a multi-node NSP cluster to carry EOL 7250 nodes through an NFM-P to NSP migration in parallel?",
  ["NSP_Installation_and_Upgrade_Guide","NSP_Planning_Guide"], ["cluster","node","migration","resource"]),
 ("At 30K LSPs with telemetry-based PCC-init tunnels, how many can NRC-P track before reaction time and GC degrade?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["lsp","pcc","telemetry","reaction"]),
 ("What is the CPU, memory and disk profile difference between co-hosted NSP plus CLM and a standalone CLM?",
  ["NSP_Planning_Guide","NSP_Installation_and_Upgrade_Guide"], ["cpu","memory","disk","resource"]),
 ("Why is NRC-P reaction time slow at 30K LSPs and which knobs (telemetry cadence, resignal timers, PCE recompute) move it?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["reaction","resignal","pce","recompute","timer"]),
 ("How do I stop PCE from auto-rerouting controller-inserted SR Policy / SR-TE LSPs while still letting congestion optimization reroute others?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["pce","sr policy","reroute","optimization"]),
 ("When a link congests, what is the blast radius — which LSPs reroute, in what order, and how long until the TED reconverges?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["congestion","reroute","ted","lsp"]),
 ("How do I prove telemetry-driven auto-reroute end to end in a lab, from congestion to NRC-P recompute to path change?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["telemetry","reroute","optimization","path"]),
 ("What is the difference in reaction-time tuning between a PCC-init and a PCE-init LSP model?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["pcc","pce","lsp","reaction"]),
 ("Which path-control reaction-time control knobs are safe to tune at hyperscaler WAN scale?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["reaction","timer","optimization"]),
 ("During NFM-P to model-driven NSP cutover, how do I keep adapter-managed nodes and native model-driven nodes coherent in one topology?",
  ["NSP_Installation_and_Upgrade_Guide","NSP_Device_Management_Guide"], ["migration","model-driven","adapter","nfm-p"]),
 ("What breaks when migrating the northbound API from XML (NFM-P) to REST (NSP) and what is the dependency chain?",
  ["NSP_Installation_and_Upgrade_Guide","NSP_System_Administrator_Guide"], ["rest","api","migration","northbound"]),
 ("For a 7250 adapter migration, which services and intents auto-associate on brownfield import vs need manual stitching?",
  ["NSP_Device_Management_Guide","NSP_Service_Management_Guide"], ["adapter","intent","service","migration"]),
 ("What is the rollback story if a model-driven migration goes wrong mid-flight while the network is live?",
  ["NSP_Installation_and_Upgrade_Guide","NSP_Device_Management_Guide","NSP_usecase_samples"], ["rollback","migration","restore"]),
 ("What exactly does NSP define as an unhealthy service, what is the filter looking at, and can the criteria change?",
  ["NSP_Network_and_Service_Assurance_Guide"], ["unhealthy","service","assurance","health"]),
 ("Can I correlate an optical port alarm (DDM Rx power, LinkDown) to the impacted service name and destination in one view?",
  ["NSP_Network_and_Service_Assurance_Guide","Troubleshooting_Guide","User_Guide"], ["alarm","correlate","service","port"]),
 ("At what telemetry subscription density does the assurance pipeline (collector, Kafka, OAM-PM) start dropping or lagging?",
  ["NSP_Data_Collection_and_Analysis_Guide","NSP_Network_and_Service_Assurance_Guide","NSP_Planning_Guide"], ["telemetry","subscription","collector","oam","kafka"]),
 ("How do I keep out-of-box telemetry subscriptions (OAM-PM, ACT) enabled at scale without capping NE count?",
  ["NSP_Data_Collection_and_Analysis_Guide"], ["telemetry","subscription","oam","scale"]),
 ("Why can a non-admin user with two NE groups not see the interconnecting links unless Access to all Equipment is enabled?",
  ["NSP_System_Administrator_Guide"], ["role","resource","equipment","access"]),
 ("Can I grant a CLM admin role that lets a user edit CLM function without full NSP admin rights?",
  ["NSP_System_Administrator_Guide"], ["role","admin","clm","user"]),
 ("How does RBAC resource-group scoping interact with the map and topology layer?",
  ["NSP_System_Administrator_Guide","NSP_Path_Control_and_Simulation_Guide"], ["resource","group","map","topology"]),
 ("Given an active alarm, the affected LSP, the link utilization and the service intent, walk from symptom to root cause across all domains.",
  ["NSP_Network_and_Service_Assurance_Guide","Troubleshooting_Guide","NSP_Path_Control_and_Simulation_Guide"], ["alarm","lsp","root cause","service"]),
 ("Which NSP features are gated behind CLM licensing versus base, for a customer-safe licensing posture?",
  ["NSP_Planning_Guide","NSP_System_Administrator_Guide"], ["license","clm","feature"]),
 ("What changed for Path Control and assurance between recent NSP releases that a large customer would feel?",
  ["NSP_Path_Control_and_Simulation_Guide","NSP_Network_and_Service_Assurance_Guide"], ["path control","assurance","release"]),
 ("If IGP link latency history shows drift on a core link, how do I tell a real transport issue from a telemetry artifact or reroute side-effect?",
  ["NSP_Network_and_Service_Assurance_Guide","NSP_Path_Control_and_Simulation_Guide"], ["latency","igp","link","telemetry"]),
 ("How do I quantify oversubscription and headroom across the LSP fabric to say how long before more capacity is needed?",
  ["NSP_Path_Control_and_Simulation_Guide","NSP_Planning_Guide","Analytics"], ["utilization","capacity","headroom","lsp"]),
 ("For a customer running both PCE and PCC-init LSPs, how do I audit which controller owns each tunnel and catch ones out of managed state?",
  ["NSP_Path_Control_and_Simulation_Guide"], ["pce","pcc","lsp","controller","managed"]),
 ("When the customer asks is my network healthy right now, what quantified answer spans alarms, down LSPs, reroute churn, link utilization and telemetry health?",
  ["NSP_Network_and_Service_Assurance_Guide","Troubleshooting_Guide"], ["health","alarm","assurance","utilization"]),
]


def grade(res, guides, terms) -> int:
    g = res.get("guide") in guides
    text = ((res.get("heading") or "") + " " + (res.get("snippet") or "")).lower()
    t = sum(1 for term in terms if term.lower() in text)
    if g and t >= 1:
        return 2
    if g or t >= 2:
        return 1
    return 0


def measure(label):
    hit = p5 = mrr = 0.0
    t0 = time.time()
    for q, guides, terms in QUESTIONS:
        res = server.docs_search(q, limit=10).get("results", [])
        grades = [grade(r, guides, terms) for r in res]
        top5 = grades[:K]
        hit += 1.0 if any(g >= 2 for g in top5) else 0.0
        p5 += sum(1 for g in top5 if g >= 2) / K
        rank = next((i + 1 for i, g in enumerate(grades) if g >= 2), 0)
        mrr += (1.0 / rank) if rank else 0.0
    n = len(QUESTIONS)
    print(f"{label:34} Hit@5={hit/n*100:5.1f}%   P@5={p5/n*100:5.1f}%   "
          f"MRR={mrr/n:5.3f}   ({time.time()-t0:4.1f}s)")


def _emb_bge_small():
    server.EMBED_BACKEND = "local"
    server.VEC_COLUMN = "embedding"
    server._VEC = None
    server._embed_query.cache_clear()


def _emb_bge_m3():
    server.EMBED_BACKEND = "ollama"
    server.VEC_COLUMN = "embedding_m3"
    server._VEC = None
    server._embed_query.cache_clear()


def cfg_baseline():
    _emb_bge_small(); server.RERANK_ENABLED = False


def cfg_rerank():
    _emb_bge_small(); server.RERANK_ENABLED = True; server._RERANKER = None; server.PROSE_BIAS = False


def cfg_rerank_routing():
    _emb_bge_small(); server.RERANK_ENABLED = True; server._RERANKER = None; server.PROSE_BIAS = True


def cfg_rerank_routing_m3():
    _emb_bge_m3(); server.RERANK_ENABLED = True; server._RERANKER = None; server.PROSE_BIAS = True


CONFIGS = {
    "baseline (hybrid, no rerank)": cfg_baseline,
    "+ rerank (ms-marco-L12)": cfg_rerank,
    "+ rerank + routing (bge-small)": cfg_rerank_routing,
    "+ rerank + routing + bge-m3 (GPU)": cfg_rerank_routing_m3,
}

if __name__ == "__main__":
    only = sys.argv[1:]
    print(f"{len(QUESTIONS)} questions | metrics over top-{K} (Hit/P) and top-10 (MRR)\n")
    for label, setup in CONFIGS.items():
        if only and not any(o in label for o in only):
            continue
        setup()
        measure(label)
