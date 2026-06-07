"""
Fair-shot test: give the structural methods TOXICITY-AWARE variants and check whether
any of them beats the graph-free producer signals (TC, Vol) on the rate-based
downstream-toxicity metrics. Answers the objection that plain betweenness/closeness
are content-blind, so the comparison is unfair.

All on the proper directed reply graph (B), in ORIGINAL-author space, against the same
reply-forest ground truth (Direct/Down counts, Elicit/Subtree rates).

Methods:
  Toxicity-BLIND structural (reference): WtdDegree (w in [1,2]), Betweenness, Closeness
  Toxicity-AWARE structural:
    ToxDegree     : weighted degree using per-edge TOXIC-reply COUNTS (not capped at 2x)
    ToxPPR        : personalized PageRank, restart distribution proportional to a user's
                    toxic-comment count (random walk biased toward toxic regions)
    ToxSub-PR     : PageRank on the toxic-interaction subgraph (edges with >=1 toxic reply)
    ToxSub-Btw    : betweenness on the toxic-interaction subgraph
    ToxSub-Clo    : closeness on the toxic-interaction subgraph
  Producer reference: ToxicCount, Volume
"""

import json
import os
from collections import Counter, defaultdict
from statistics import mean

import networkx as nx

from .. import config
from ..forest import build_reply_forest, downstream_metrics
from ..io import load_jsonl

RESULTS_DIR = config.RESULTS_DIR
TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
OUT = os.path.join(RESULTS_DIR, "table3_toxicity_aware.json")
TOP_K = 10
MIN_DIRECT = 5
MIN_DESC = 5


def main():
    toxic = load_jsonl(TOXIC)
    non = load_jsonl(NONTOXIC)
    posts = toxic + non
    author, tox, parent = build_reply_forest(posts)  # keyed by comment id, ORIGINAL authors
    d_cnt, s_cnt = downstream_metrics(author, tox, parent)  # toxic direct / distinct toxic descendants

    # producer signals
    tc = Counter()
    vol = Counter()
    for p in posts:
        a = p.get("author", "unknown")
        vol[a] += 1
        if p.get("prediction") == "toxic":
            tc[a] += 1
    toxic_users = [u for u in vol if tc[u] >= 1]
    print(f"users={len(vol)}  toxic users={len(toxic_users)}", flush=True)

    # build directed reply graph in author space, with per-edge total & toxic counts
    e_tot = Counter()
    e_tox = Counter()
    for cid in author:
        p = parent.get(cid)
        if p is None or p not in author:
            continue
        pu, rv = author[p], author[cid]
        if pu == rv:
            continue
        e_tot[(pu, rv)] += 1
        if tox[cid]:
            e_tox[(pu, rv)] += 1

    G = nx.DiGraph()
    for (u, v), t in e_tot.items():
        prop = 1.0 + e_tox[(u, v)] / t  # the w in [1,2] used in the paper
        G.add_edge(u, v, prop=prop, tox=float(e_tox[(u, v)]), toxw=1.0 + e_tox[(u, v)])
    Gt = nx.DiGraph()  # toxic-interaction subgraph
    for (u, v), tx in e_tox.items():
        if tx >= 1:
            Gt.add_edge(u, v, tox=float(tx))
    print(
        f"G: {G.number_of_nodes()} nodes {G.number_of_edges()} edges | "
        f"Gtox: {Gt.number_of_nodes()} nodes {Gt.number_of_edges()} edges",
        flush=True,
    )

    # ---- scores (dict author -> value) ----
    def wdeg(weight):
        d = defaultdict(float)
        for u, v, data in G.edges(data=True):
            d[u] += data[weight]
            d[v] += data[weight]
        return d

    scores = {}
    scores["WtdDegree(blind)"] = wdeg("prop")
    scores["ToxDegree"] = wdeg("tox")
    scores["Betweenness(blind)"] = nx.betweenness_centrality(G, weight=None)
    scores["Closeness(blind)"] = nx.closeness_centrality(G)
    pers = {n: float(tc.get(n, 0)) for n in G.nodes()}
    if sum(pers.values()) > 0:
        scores["ToxPPR"] = nx.pagerank(G, personalization=pers, weight="prop")
    scores["ToxSub-PR"] = nx.pagerank(Gt, weight="tox") if Gt.number_of_edges() else {}
    print("toxic-subgraph betweenness/closeness...", flush=True)
    scores["ToxSub-Btw"] = nx.betweenness_centrality(Gt, weight=None) if Gt.number_of_edges() else {}
    scores["ToxSub-Clo"] = nx.closeness_centrality(Gt) if Gt.number_of_edges() else {}
    scores["ToxicCount"] = {u: tc[u] for u in vol}
    scores["Volume"] = {u: vol[u] for u in vol}

    # ---- validation ----
    dt_tot, dt_tox, ds_tot, ds_tox = Counter(), Counter(), Counter(), Counter()
    for cid in author:
        p = parent.get(cid)
        if p is not None and p in author and author[p] != author[cid]:
            dt_tot[author[p]] += 1
            if tox[cid]:
                dt_tox[author[p]] += 1
    for cid in author:
        ca = author[cid]
        seen = set()
        node = parent.get(cid)
        hops = 0
        while node is not None and hops < 500:
            a = author.get(node)
            if a is not None and a != ca and a not in seen:
                seen.add(a)
                ds_tot[a] += 1
                if tox[cid]:
                    ds_tox[a] += 1
            node = parent.get(node)
            hops += 1
    base_e = sum(dt_tox.values()) / sum(dt_tot.values())
    base_s = sum(ds_tox.values()) / sum(ds_tot.values())

    def elicit(o):
        return dt_tox[o] / dt_tot[o] if dt_tot[o] >= MIN_DIRECT else None

    def subrate(o):
        return ds_tox[o] / ds_tot[o] if ds_tot[o] >= MIN_DESC else None

    def evaluate(scoremap):
        ranked = sorted((u for u in toxic_users if u in scoremap), key=lambda u: scoremap[u], reverse=True)[
            :TOP_K
        ]
        e = [x for x in (elicit(o) for o in ranked) if x is not None]
        s = [x for x in (subrate(o) for o in ranked) if x is not None]
        return {
            "direct": round(mean([d_cnt.get(o, 0) for o in ranked]), 2),
            "downstream": round(mean([s_cnt.get(o, 0) for o in ranked]), 2),
            "elicit_pct": round(100 * mean(e), 1) if e else None,
            "subtree_pct": round(100 * mean(s), 1) if s else None,
            "n_rate": len(s),
        }

    order = [
        "ToxicCount",
        "Volume",
        "WtdDegree(blind)",
        "Betweenness(blind)",
        "Closeness(blind)",
        "ToxDegree",
        "ToxPPR",
        "ToxSub-PR",
        "ToxSub-Btw",
        "ToxSub-Clo",
    ]
    results = {m: evaluate(scores[m]) for m in order if m in scores}
    results["_base"] = {"elicit_pct": round(100 * base_e, 1), "subtree_pct": round(100 * base_s, 1)}
    json.dump(results, open(OUT, "w"), indent=2)

    print(f"\nbase rates: elicit={100*base_e:.1f}%  subtree={100*base_s:.1f}%\n")
    print(f"{'method':<20}{'direct':>7}{'down':>7}{'elicit%':>9}{'subtree%':>10}")
    for m in order:
        if m not in results:
            continue
        r = results[m]
        tag = (
            "  <- producer (graph-free)"
            if m in ("ToxicCount", "Volume")
            else ("  <- toxicity-BLIND" if "blind" in m else "  <- toxicity-AWARE")
        )
        print(
            f"{m:<20}{r['direct']:>7}{r['downstream']:>7}"
            f"{str(r['elicit_pct']):>9}{str(r['subtree_pct']):>10}{tag}"
        )
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
