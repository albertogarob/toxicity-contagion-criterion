"""
Disambiguation: separate toxicity-aware structural methods into
  (a) CIRCULAR  -- use toxic-REPLY interaction labels (overlap with the outcome):
                   ToxDegree, ToxSub-PR
  (b) CLEAN     -- use ONLY the user's own toxic-comment count (the same independent
                   signal ToxicCount uses) propagated through the topology:
                   PPR-clean  : personalized PageRank, restart ~ toxic_count, UNWEIGHTED edges
                   NbrToxIn   : sum of neighbors' toxic_count over in-edges (people u replied to)
                   NbrToxOut  : sum of neighbors' toxic_count over out-edges (people who replied to u)
                   DiffuseTC  : 2-step diffusion of toxic_count over the undirected reply graph

If a CLEAN method beats ToxicCount on the rate metric, the graph genuinely adds value
beyond the producer record. If only CIRCULAR methods do, the apparent structural win is
label leakage. Graph B (reply/all), original-author space, same rate ground truth.
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
OUT = os.path.join(RESULTS_DIR, "table3_diffusion.json")
TOP_K = 10
MIN_DIRECT = 5
MIN_DESC = 5


def main():
    toxic = load_jsonl(TOXIC)
    non = load_jsonl(NONTOXIC)
    posts = toxic + non
    author, tox, parent = build_reply_forest(posts)
    d_cnt, s_cnt = downstream_metrics(author, tox, parent)

    tc = Counter()
    vol = Counter()
    for p in posts:
        a = p.get("author", "unknown")
        vol[a] += 1
        if p.get("prediction") == "toxic":
            tc[a] += 1
    toxic_users = [u for u in vol if tc[u] >= 1]

    # reply graph in author space (+ per-edge toxic counts for the circular refs)
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
    for (u, v), _t in e_tot.items():
        G.add_edge(u, v, tox=float(e_tox[(u, v)]))
    UG = G.to_undirected()

    scores = {}
    # --- CLEAN: own toxic_count propagated through topology ---
    pers = {n: float(tc.get(n, 0)) for n in G.nodes()}
    scores["PPR-clean"] = nx.pagerank(G, personalization=pers, weight=None)  # UNWEIGHTED edges
    nin = defaultdict(float)
    nout = defaultdict(float)
    for u, v in G.edges():  # edge pu->rv : pu received reply from rv
        nout[u] += tc.get(v, 0)  # u's repliers' toxicity
        nin[v] += tc.get(u, 0)  # the people u replied to, their toxicity
    scores["NbrToxOut"] = nin  # toxicity of users u replied to
    scores["NbrToxIn"] = nout  # toxicity of users who replied to u
    diff = defaultdict(float)
    for u in UG.nodes():
        diff[u] = tc.get(u, 0) + 0.5 * sum(tc.get(w, 0) for w in UG.neighbors(u))
    scores["DiffuseTC"] = diff
    # --- CIRCULAR refs (use toxic-reply labels) ---
    twd = defaultdict(float)
    for u, v, data in G.edges(data=True):
        twd[u] += data["tox"]
        twd[v] += data["tox"]
    scores["ToxDegree(circ)"] = twd
    Gt = nx.DiGraph()
    for (u, v), tx in e_tox.items():
        if tx >= 1:
            Gt.add_edge(u, v, tox=float(tx))
    scores["ToxSub-PR(circ)"] = nx.pagerank(Gt, weight="tox") if Gt.number_of_edges() else {}
    # --- producer ref ---
    scores["ToxicCount"] = {u: tc[u] for u in vol}
    scores["Volume"] = {u: vol[u] for u in vol}

    # validation rates
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

    def evaluate(sm):
        ranked = sorted((u for u in toxic_users if u in sm), key=lambda u: sm[u], reverse=True)[:TOP_K]
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
        "PPR-clean",
        "NbrToxIn",
        "NbrToxOut",
        "DiffuseTC",
        "ToxDegree(circ)",
        "ToxSub-PR(circ)",
    ]
    results = {m: evaluate(scores[m]) for m in order}
    results["_base"] = {"elicit_pct": round(100 * base_e, 1), "subtree_pct": round(100 * base_s, 1)}
    json.dump(results, open(OUT, "w"), indent=2)

    print(f"base: elicit={100*base_e:.1f}% subtree={100*base_s:.1f}%\n")
    print(f"{'method':<18}{'direct':>7}{'down':>7}{'elicit%':>9}{'subtree%':>10}{'n':>4}")
    for m in order:
        r = results[m]
        tag = "producer" if m in ("ToxicCount", "Volume") else "CIRCULAR" if "circ" in m else "CLEAN"
        print(
            f"{m:<18}{r['direct']:>7}{r['downstream']:>7}{str(r['elicit_pct']):>9}"
            f"{str(r['subtree_pct']):>10}{r['n_rate']:>4}  {tag}"
        )
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
