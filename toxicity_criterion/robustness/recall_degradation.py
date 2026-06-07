"""
Recall-degradation robustness of the headline validation.

The classifier validation puts Sonnet's in-domain recall in a range (~0.32 vs the LLM
panel, up to ~0.77 on clear-cut toxicity). This script asks the question the downstream
paper actually needs answered: does the headline conclusion -- producer-count is the
strongest amplifier-selection method on the reach-controlled (subtree-rate) validation --
survive if Sonnet's recall were materially WORSE than it is?

We simulate worse recall by randomly DROPPING a fraction f of the toxic labels (toxic ->
non-toxic), which degrades BOTH the producer signal (ToxicCount) AND the downstream-toxicity
ground truth (subtree rate) together -- exactly how a lower-recall classifier would distort
the corpus. We then re-run the selection-vs-outcome comparison and check whether
ToxicCount / Volume still top the structural methods (weighted degree, PageRank,
reply-breadth) and the pool base rate.

Constructions mirror the paper: directed reply graph over all users, toxicity-graded edge
weights w in [1,2]; outcome = subtree rate (toxic share of a user's downstream comments),
with a minimum-support filter. IM/CELF++ is omitted (expensive, and already shown to fail
the rate check); structure is otherwise label-near-invariant, which is the point.

Reproducible: PYTHONHASHSEED pinned; f>0 averaged over SEEDS perturbations (mean +/- sd).
"""

import os
import random
import sys
from collections import Counter, defaultdict
from statistics import mean, pstdev

import networkx as nx

from .. import config
from ..forest import build_reply_forest
from ..io import load_jsonl

TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
MIN_DESC = 5
TOP_K = 10
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = 20
PRODUCERS = {"ToxicCount", "Volume"}


def desc_rate(author, toxic, parent):
    """dtot[u], dtox[u]: distinct downstream comments below u, and the toxic share."""
    dtot, dtox = Counter(), Counter()
    for c in author:
        ca = author[c]
        seen = set()
        node = parent.get(c)
        hops = 0
        while node is not None and hops < 500:
            a = author.get(node)
            if a is not None and a != ca and a not in seen:
                seen.add(a)
                dtot[a] += 1
                if toxic[c]:
                    dtox[a] += 1
            node = parent.get(node)
            hops += 1
    return dtot, dtox


def method_scores(author, toxic, parent):
    """ToxicCount, Volume (producers) + WtdDegree, PageRank, ReplyBreadth (structure)."""
    tc, vol = Counter(), Counter()
    for c, a in author.items():
        vol[a] += 1
        if toxic[c]:
            tc[a] += 1
    # directed reply graph author(parent)->author(child); weight 1 + toxic share of replies
    tot, tox = Counter(), Counter()
    for c in author:
        p = parent.get(c)
        if p is None or p not in author:
            continue
        u, v = author[p], author[c]
        if u == v:
            continue
        tot[(u, v)] += 1
        if toxic[c]:
            tox[(u, v)] += 1
    G = nx.DiGraph()
    for (u, v), n in tot.items():
        G.add_edge(u, v, weight=1.0 + tox[(u, v)] / n)
    wd = defaultdict(float)
    for u, v, w in G.edges(data="weight"):
        wd[u] += w
        wd[v] += w
    pr = nx.pagerank(G, weight="weight") if G.number_of_edges() else {}
    breadth = {u: G.out_degree(u) for u in G.nodes()}
    return {
        "ToxicCount": dict(tc),
        "Volume": dict(vol),
        "WtdDegree": dict(wd),
        "PageRank": pr,
        "ReplyBreadth": breadth,
    }


def evaluate(toxic, author, parent):
    """Return per-method top-10 mean subtree rate (%), and the evaluable-pool base rate."""
    scores = method_scores(author, toxic, parent)
    dtot, dtox = desc_rate(author, toxic, parent)
    rate = {u: dtox[u] / dtot[u] for u in dtot if dtot[u] >= MIN_DESC}
    tox_users = {u for u in scores["ToxicCount"] if scores["ToxicCount"][u] >= 1}
    pool = [rate[u] for u in rate if u in tox_users]
    if len(pool) < TOP_K:
        return None
    out = {}
    for m, sc in scores.items():
        ranked = [u for u in sorted(sc, key=lambda x: sc[x], reverse=True) if u in tox_users]
        topk = [rate[u] for u in ranked if u in rate][:TOP_K]
        out[m] = 100 * mean(topk) if topk else float("nan")
    out["_pool"] = 100 * mean(pool)
    return out


def degrade(base_toxic, f, seed):
    """Keep each toxic label with prob (1-f); simulates recall scaled by (1-f)."""
    rng = random.Random(seed)
    return {c: (t and rng.random() >= f) for c, t in base_toxic.items()}


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})
    posts = load_jsonl(TOXIC) + load_jsonl(NONTOXIC)
    author, base_toxic, parent = build_reply_forest(posts)
    n_tox = sum(base_toxic.values())
    print(f"{len(author)} comments, {n_tox} toxic; degrading recall by dropping toxic labels.\n")

    methods = ["ToxicCount", "Volume", "WtdDegree", "PageRank", "ReplyBreadth"]
    print(
        f"Top-{TOP_K} mean subtree rate (%) vs simulated recall loss "
        f"(f = fraction of toxic labels dropped; effective recall ~ (1-f)x current):"
    )
    header = "  f      eff.recall  " + "".join(f"{m:>13}" for m in methods) + f"{'pool':>9}  winner"
    print(header)
    for f in FRACS:
        seeds = [0] if f == 0 else list(range(SEEDS))
        rows = []
        for s in seeds:
            tox = base_toxic if f == 0 else degrade(base_toxic, f, s)
            r = evaluate(tox, author, parent)
            if r:
                rows.append(r)
        agg = {m: mean([r[m] for r in rows]) for m in methods + ["_pool"]}
        sd = {m: (pstdev([r[m] for r in rows]) if len(rows) > 1 else 0.0) for m in methods}
        best = max(methods, key=lambda m: agg[m])
        prod_best = max(PRODUCERS, key=lambda m: agg[m])
        win = "producer" if best in PRODUCERS else f"** {best} **"
        cells = "".join(f"{agg[m]:>8.1f}±{sd[m]:>3.1f}" for m in methods)
        print(
            f"  {f:<4.1f}   {1-f:>6.2f}     {cells}{agg['_pool']:>9.1f}  {win}"
            + ("" if best in PRODUCERS else f" (>{prod_best} {agg[prod_best]:.1f})")
        )
    print("\nReading: if 'winner' stays 'producer' as f grows, the headline conclusion does")
    print("not hinge on Sonnet's recall -- worse recall attenuates all methods together and")
    print("producer-count keeps its lead over the structural methods and the pool base rate.")


if __name__ == "__main__":
    main()
