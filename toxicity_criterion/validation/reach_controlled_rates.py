"""
Non-circular validation. The earlier ground truth (downstream toxic *count*) is
mechanically correlated with reach/degree, so degree-like scores predict it
tautologically. Here we use rate-based outcomes that control for reach:

  elicitation_rate(u) = toxic direct replies / total direct replies   (provocativeness)
  subtree_rate(u)     = toxic descendants / total descendants          (conversation toxicity)

compared against the global base rates. A method's selection is a genuine
amplifier set only if its users elicit toxicity at an ABOVE-baseline rate, not
merely because they are popular. Rates require a minimum support (replies /
descendants) to be meaningful.

Reproducible (PYTHONHASHSEED pinned); IM reported as mean +/- std over runs.
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import json
import random
from collections import Counter, defaultdict
from statistics import mean, pstdev

import networkx as nx

from .. import config
from ..forest import build_reply_forest
from ..graph import build_reply_graph
from ..influence import CELFPlusPlus, IndependentCascadeModel
from ..io import load_jsonl

RESULTS_DIR = config.RESULTS_DIR
TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
OUT = os.path.join(RESULTS_DIR, "table2_reach_controlled_rates.json")
TOP_K = 10
P0 = 0.15
MAX_CAND = 200
R = 200
N_IM = 10
N_RANDOM = 50
MIN_DIRECT = 5  # min direct replies to compute elicitation rate
MIN_DESC = 5  # min descendants to compute subtree rate


def rate_metrics(author, toxic, parent):
    direct_tot, direct_tox = Counter(), Counter()
    for c in author:
        p = parent.get(c)
        if p is not None and p in author and author[p] != author[c]:
            direct_tot[author[p]] += 1
            if toxic[c]:
                direct_tox[author[p]] += 1
    desc_tot, desc_tox = Counter(), Counter()
    for c in author:
        ca = author[c]
        seen = set()
        node = parent.get(c)
        hops = 0
        while node is not None and hops < 500:
            a = author.get(node)
            if a is not None and a != ca and a not in seen:
                seen.add(a)
                desc_tot[a] += 1
                if toxic[c]:
                    desc_tox[a] += 1
            node = parent.get(node)
            hops += 1
    return direct_tot, direct_tox, desc_tot, desc_tox


def main():
    toxic_posts = load_jsonl(TOXIC)
    nontoxic_posts = load_jsonl(NONTOXIC)
    allposts = toxic_posts + nontoxic_posts
    author, toxic, parent = build_reply_forest(allposts)
    direct_tot, direct_tox, desc_tot, desc_tox = rate_metrics(author, toxic, parent)

    # global base rates
    base_direct = sum(direct_tox.values()) / sum(direct_tot.values())
    base_desc = sum(desc_tox.values()) / sum(desc_tot.values())
    print(
        f"base elicitation rate = {base_direct:.3f}; base subtree toxicity rate = {base_desc:.3f}", flush=True
    )

    g = build_reply_graph(toxic_posts, nontoxic_posts, weighting="proportion")
    anon2orig = g.anonymizer.anon_to_original
    toxic_set = {u for u in g.users if g.users[u].toxic_count >= 1}
    print(
        f"graph: {len(g.users)} nodes, {len(g.edge_weights)} edges; toxic users: {len(toxic_set)}", flush=True
    )

    def elicit(o):
        return direct_tox[o] / direct_tot[o] if direct_tot[o] >= MIN_DIRECT else None

    def subrate(o):
        return desc_tox[o] / desc_tot[o] if desc_tot[o] >= MIN_DESC else None

    def score(top):
        origs = [anon2orig.get(u, u) for u in top]
        e = [elicit(o) for o in origs]
        e = [x for x in e if x is not None]
        s = [subrate(o) for o in origs]
        s = [x for x in s if x is not None]
        return (mean(e) if e else float("nan"), len(e), mean(s) if s else float("nan"), len(s))

    def top_toxic(ranked_pairs):
        out = []
        for u, _ in ranked_pairs:
            if u in toxic_set:
                out.append(u)
            if len(out) == TOP_K:
                break
        return out

    breadth = sorted(((u, g.get_out_degree(u)) for u in g.users), key=lambda x: x[1], reverse=True)
    wd = defaultdict(float)
    for (u, v), w in g.edge_weights.items():
        wd[u] += w
        wd[v] += w
    G = nx.DiGraph()
    for (u, v), w in g.edge_weights.items():
        G.add_edge(u, v, weight=w)
    pr = sorted(nx.pagerank(G, weight="weight").items(), key=lambda x: x[1], reverse=True)
    print("betweenness/closeness...", flush=True)
    btw = sorted(nx.betweenness_centrality(G, weight=None).items(), key=lambda x: x[1], reverse=True)
    clo = sorted(nx.closeness_centrality(G).items(), key=lambda x: x[1], reverse=True)
    tc = sorted(((n, u.toxic_count) for n, u in g.users.items()), key=lambda x: x[1], reverse=True)
    vol = sorted(((n, u.total_posts) for n, u in g.users.items()), key=lambda x: x[1], reverse=True)

    methods = {
        "ReplyBreadth": top_toxic(breadth),
        "WtdDegree": top_toxic(sorted(wd.items(), key=lambda x: x[1], reverse=True)),
        "Closeness": top_toxic(clo),
        "Betweenness": top_toxic(btw),
        "PageRank": top_toxic(pr),
        "ToxicCount": top_toxic(tc),
        "Volume": top_toxic(vol),
    }
    results = {}
    for m, top in methods.items():
        er, en, sr, sn = score(top)
        results[m] = {
            "elicit_rate": round(er, 3),
            "elicit_n": en,
            "subtree_rate": round(sr, 3),
            "subtree_n": sn,
        }

    # IM stochastic
    def celf_once(seed):
        random.seed(seed)
        model = IndependentCascadeModel(g, P0)
        celf = CELFPlusPlus(g, model, num_simulations=R)
        cand = g.get_candidate_users(min_toxic_posts=1, max_candidates=MAX_CAND)
        return [n for n, _ in celf.find_top_k_influential(TOP_K, candidates=cand, verbose=False)]

    im_e, im_s = [], []
    for i in range(N_IM):
        er, en, sr, sn = score(celf_once(1000 * (i + 1)))
        im_e.append(er)
        im_s.append(sr)
    results["IM"] = {
        "elicit_rate": round(mean(im_e), 3),
        "elicit_std": round(pstdev(im_e), 3),
        "subtree_rate": round(mean(im_s), 3),
        "subtree_std": round(pstdev(im_s), 3),
    }

    # random
    rng = random.Random(42)
    tox_list = sorted(toxic_set)
    re_, rs_ = [], []
    for _ in range(N_RANDOM):
        er, en, sr, sn = score(rng.sample(tox_list, TOP_K))
        if en:
            re_.append(er)
        if sn:
            rs_.append(sr)
    results["Random(toxic)"] = {
        "elicit_rate": round(mean(re_), 3),
        "elicit_std": round(pstdev(re_), 3),
        "subtree_rate": round(mean(rs_), 3),
        "subtree_std": round(pstdev(rs_), 3),
    }
    results["_base"] = {"elicit_rate": round(base_direct, 3), "subtree_rate": round(base_desc, 3)}

    print(f"\n{'Method':<16} {'elicit_rate':>12} {'subtree_rate':>14}", flush=True)
    for m in [
        "ReplyBreadth",
        "WtdDegree",
        "Closeness",
        "Betweenness",
        "PageRank",
        "IM",
        "ToxicCount",
        "Volume",
        "Random(toxic)",
    ]:
        r = results[m]
        print(f"{m:<16} {r['elicit_rate']:>12} {r['subtree_rate']:>14}", flush=True)
    print(f"{'BASE RATE':<16} {base_direct:>12.3f} {base_desc:>14.3f}", flush=True)

    json.dump(
        {"config": {"MIN_DIRECT": MIN_DIRECT, "MIN_DESC": MIN_DESC, "R": R}, "results": results},
        open(OUT, "w"),
        indent=2,
    )
    print(f"\nSaved: {OUT}", flush=True)


if __name__ == "__main__":
    main()
