"""
T1.1 downstream-impact validation (observational, non-circular).

Question: on the repaired reply graph, which method of selecting "amplifiers"
picks the users who actually sit above the most toxic discourse? We compare
several selection methods against ground truth taken from the REAL reply trees
(not the IC model, so influence maximization cannot win by construction).

Selection methods (top-10 each), computed from the all-users reply graph / activity:
  IM            CELF++ frequency ranking on graph_v2
  ReplyBreadth  out-degree in the reply graph (# distinct users who replied to u)  [the cheap amplifier metric]
  WtdDegree     weighted degree centrality
  PageRank      PageRank
  ToxicCount    # toxic comments authored (the "producer" baseline)
  Volume        # comments authored
  Random        mean over random draws (control)

Validation metrics per user, from the actual comment reply forest:
  direct_toxic  # toxic comments posted as a DIRECT reply to one of u's comments
  subtree_toxic # distinct toxic comments anywhere DOWNSTREAM of u's comments
A method is "better" if its selected users score higher on these.
"""

import json
import os
import random
from collections import Counter, defaultdict

import networkx as nx

from .. import config
from ..forest import build_reply_forest, downstream_metrics
from ..graph import build_reply_graph
from ..influence import CELFPlusPlus, IndependentCascadeModel
from ..io import load_jsonl

RESULTS_DIR = config.RESULTS_DIR
TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
OUT = os.path.join(RESULTS_DIR, "table2_reach_controlled_counts.json")
TOP_K = 10
P0 = 0.15
MAX_CAND = 200
R = 200
N_MC = 5
N_RANDOM = 20


def main():
    toxic_posts = load_jsonl(TOXIC)
    nontoxic_posts = load_jsonl(NONTOXIC)
    allposts = toxic_posts + nontoxic_posts

    # ground truth from real reply forest
    author, toxic, parent = build_reply_forest(allposts)
    direct, subtree = downstream_metrics(author, toxic, parent)
    print(f"users with >=1 direct toxic reply received: {len(direct)}")
    print(f"users with >=1 toxic comment downstream: {len(subtree)}")

    # repaired reply graph
    g = build_reply_graph(toxic_posts, nontoxic_posts, weighting="proportion")
    print(f"graph_v2: {len(g.users)} nodes, {len(g.edge_weights)} edges")

    # ---- selection methods (top-10), all in anonymized User_ ids ----
    # IM frequency ranking
    def celf(seed):
        random.seed(seed)
        m = IndependentCascadeModel(g, P0)
        c = CELFPlusPlus(g, m, num_simulations=R)
        cand = g.get_candidate_users(min_toxic_posts=1, max_candidates=MAX_CAND)
        return [n for n, _ in c.find_top_k_influential(TOP_K, candidates=cand, verbose=False)]

    freq = Counter()
    for i in range(N_MC):
        for u in celf(1000 * (i + 1))[:TOP_K]:
            freq[u] += 1
    im_top = [u for u, _ in freq.most_common(TOP_K)]

    # reply breadth = out-degree (distinct repliers) in graph_v2
    breadth = {u: g.get_out_degree(u) for u in g.users}
    rb_top = [u for u, _ in sorted(breadth.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

    # weighted degree
    wd = defaultdict(float)
    for (u, v), w in g.edge_weights.items():
        wd[u] += w
        wd[v] += w
    wd_top = [u for u, _ in sorted(wd.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

    # pagerank
    G = nx.DiGraph()
    for (u, v), w in g.edge_weights.items():
        G.add_edge(u, v, weight=w)
    pr = nx.pagerank(G, weight="weight")
    pr_top = [u for u, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:TOP_K]]

    # producers: toxic count, volume
    tc_top = [u for u, _ in sorted(g.users.items(), key=lambda x: x[1].toxic_count, reverse=True)[:TOP_K]]
    vol_top = [u for u, _ in sorted(g.users.items(), key=lambda x: x[1].total_posts, reverse=True)[:TOP_K]]

    # map anonymized User_ ids back to original usernames to look up ground truth
    anon2orig = g.anonymizer.anon_to_original

    def score(method_top):
        origs = [anon2orig.get(u, u) for u in method_top]
        d = [direct.get(o, 0) for o in origs]
        s = [subtree.get(o, 0) for o in origs]
        return {
            "direct_sum": sum(d),
            "direct_mean": round(sum(d) / len(d), 2),
            "subtree_sum": sum(s),
            "subtree_mean": round(sum(s) / len(s), 2),
        }

    methods = {
        "IM": im_top,
        "ReplyBreadth": rb_top,
        "WtdDegree": wd_top,
        "PageRank": pr_top,
        "ToxicCount": tc_top,
        "Volume": vol_top,
    }
    results = {m: score(top) for m, top in methods.items()}

    # random control
    rng = random.Random(42)
    cand_all = g.get_candidate_users(min_toxic_posts=1, max_candidates=None)
    rd_direct, rd_sub = [], []
    for _ in range(N_RANDOM):
        pick = rng.sample(cand_all, TOP_K)
        sc = score(pick)
        rd_direct.append(sc["direct_mean"])
        rd_sub.append(sc["subtree_mean"])
    results["Random(toxic-cand)"] = {
        "direct_mean": round(sum(rd_direct) / N_RANDOM, 2),
        "subtree_mean": round(sum(rd_sub) / N_RANDOM, 2),
    }

    print(f"\n{'Method':<20} {'direct_sum':>11} {'direct_mean':>12} {'subtree_sum':>12} {'subtree_mean':>13}")
    for m in ["IM", "ReplyBreadth", "WtdDegree", "PageRank", "ToxicCount", "Volume", "Random(toxic-cand)"]:
        r = results[m]
        print(
            f"{m:<20} {r.get('direct_sum','-'):>11} {r['direct_mean']:>12} "
            f"{r.get('subtree_sum','-'):>12} {r['subtree_mean']:>13}"
        )

    # Note: the per-method top-k user lists (im_top, rb_top) are deliberately NOT written to the
    # committed results file. Even as anonymized User_<n> pseudonyms they single out individual
    # users; only the aggregate metrics are persisted.
    json.dump(
        {"config": {"R": R, "N_MC": N_MC, "top_k": TOP_K}, "results": results},
        open(OUT, "w"),
        indent=2,
    )
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
