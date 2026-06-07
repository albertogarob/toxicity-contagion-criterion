"""
Decisive test: does a CLEAN (non-circular) toxicity-diffused graph index robustly beat
the producer baselines (ToxicCount, Volume) on the reach-controlled outcome?

Outcome (reach-controlled): per-user SUBTREE TOXICITY RATE = toxic descendants / total
descendants (support >= 5). Controls for reach (unlike the raw count, which is tautological).

Clean diffusion methods use ONLY each user's own toxic-comment count propagated through the
reply topology (no toxic-reply labels -> no outcome leakage):
  PPR_toxic  : personalized PageRank, restart proportional to toxic_count, UNWEIGHTED edges
  NbrTox     : sum of neighbors' toxic_count (in+out)
  DiffuseTC  : toxic_count(u) + 0.5 * sum_neighbors toxic_count

Tests (ruthless, not designed to force a positive):
  1. Spearman rho(score, subtree_rate) over all evaluable toxic users, with bootstrap 95% CI.
  2. The contrast best-diffusion vs best-producer: bootstrap CI of the rho DIFFERENCE
     (does it exclude 0?). This is the headline.
  3. Top-k subtree rate at k = 10, 20, 50.
  4. Per-subreddit: which family wins on rho.
  5. Label-perturbation robustness: flip 15% of toxicity labels, recompute; fraction of
     runs where diffusion still beats producer.
"""

import glob
import math
import os
import random
from collections import Counter, defaultdict
from statistics import mean

import networkx as nx

from .. import config
from ..forest import build_reply_forest
from ..io import load_jsonl

TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
RAW = config.RAW_DIR
MIN_DESC = 5
NBOOT = 1000
RNG = random.Random(0)


def spearman(xs, ys):
    n = len(xs)
    if n < 5:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            a = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = a
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def desc_rate(author, tox, parent):
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
                if tox[c]:
                    desc_tox[a] += 1
            node = parent.get(node)
            hops += 1
    return desc_tot, desc_tox


def build_scores(author, tox, parent, posts):
    tc, vol = Counter(), Counter()
    for p in posts:
        a = p.get("author", "unknown")
        vol[a] += 1
        if p.get("prediction") == "toxic":
            tc[a] += 1
    G = nx.DiGraph()
    for cid in author:
        pp = parent.get(cid)
        if pp is None or pp not in author:
            continue
        u, v = author[pp], author[cid]
        if u != v:
            G.add_edge(u, v)
    UG = G.to_undirected()
    pers = {n: float(tc.get(n, 0)) for n in G.nodes()}
    ppr = (
        nx.pagerank(G, personalization=pers, weight=None)
        if (G.number_of_edges() and sum(pers.values()))
        else {}
    )
    nbr = defaultdict(float)
    for u in UG.nodes():
        nbr[u] = sum(tc.get(w, 0) for w in UG.neighbors(u))
    diff = {u: tc.get(u, 0) + 0.5 * nbr[u] for u in set(list(tc) + list(nbr))}
    deg = {u: G.in_degree(u) + G.out_degree(u) for u in G.nodes()}
    breadth = {u: G.out_degree(u) for u in G.nodes()}
    prn = nx.pagerank(G, weight=None) if G.number_of_edges() else {}
    return {
        "ToxicCount": tc,
        "Volume": vol,  # producer
        "PPR_toxic": ppr,
        "NbrTox": nbr,
        "DiffuseTC": diff,  # clean diffusion
        "Degree": deg,
        "ReplyBreadth": breadth,
        "PageRank": prn,  # naive structural
    }, tc


PRODUCER = ["ToxicCount", "Volume"]
DIFFUSION = ["PPR_toxic", "NbrTox", "DiffuseTC"]
STRUCT = ["Degree", "ReplyBreadth", "PageRank"]


def evaluate(author, tox, parent, posts):
    scores, tc = build_scores(author, tox, parent, posts)
    dtot, dtox = desc_rate(author, tox, parent)
    users = [u for u in tc if tc[u] >= 1 and dtot[u] >= MIN_DESC]
    if len(users) < 20:
        return None
    outcome = {u: dtox[u] / dtot[u] for u in users}
    return scores, outcome, users


def rho_all(scores, outcome, users):
    y = [outcome[u] for u in users]
    out = {}
    for m, sc in scores.items():
        out[m] = spearman([sc.get(u, 0.0) for u in users], y)
    return out


def main():
    posts = load_jsonl(TOXIC) + load_jsonl(NONTOXIC)
    author, tox, parent = build_reply_forest(posts)
    scores, outcome, users = evaluate(author, tox, parent, posts)
    print(f"evaluable toxic users (support>={MIN_DESC}): {len(users)}\n")

    # 1. point rho
    rhos = rho_all(scores, outcome, users)
    print("Spearman rho(score, subtree_rate)  [reach-controlled]:")
    for m in sorted(rhos, key=lambda k: -(rhos[k] or -9)):
        fam = "producer" if m in PRODUCER else "DIFFUSION" if m in DIFFUSION else "structural"
        print(f"  {m:<13} rho={rhos[m]:+.3f}   {fam}")

    best_prod = max(PRODUCER, key=lambda m: rhos[m])
    best_diff = max(DIFFUSION, key=lambda m: rhos[m])
    print(
        f"\nbest producer = {best_prod} (rho={rhos[best_prod]:.3f});  "
        f"best diffusion = {best_diff} (rho={rhos[best_diff]:.3f})"
    )

    # 2. bootstrap CI of the difference (best_diff - best_prod)
    diffs = []
    for _ in range(NBOOT):
        idx = [RNG.randrange(len(users)) for _ in range(len(users))]
        bu = [users[i] for i in idx]
        by = [outcome[u] for u in bu]
        rd = spearman([scores[best_diff].get(u, 0.0) for u in bu], by)
        rp = spearman([scores[best_prod].get(u, 0.0) for u in bu], by)
        if rd is not None and rp is not None:
            diffs.append(rd - rp)
    diffs.sort()
    lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    point = rhos[best_diff] - rhos[best_prod]
    verdict = (
        "ROBUST WIN (CI excludes 0)"
        if lo > 0
        else ("ROBUST LOSS" if hi < 0 else "NOT SIGNIFICANT (CI spans 0)")
    )
    print(
        f"\n[HEADLINE] rho({best_diff}) - rho({best_prod}) = {point:+.3f}  "
        f"95% CI [{lo:+.3f}, {hi:+.3f}]  -> {verdict}"
    )

    # 3. top-k subtree rate
    print("\nTop-k subtree toxicity rate (%):")
    print(f"  {'method':<13}{'k=10':>8}{'k=20':>8}{'k=50':>8}")
    for m in PRODUCER + DIFFUSION:
        row = []
        ranked = sorted(
            users + [u for u in scores[m] if u not in users],
            key=lambda u: scores[m].get(u, 0.0),
            reverse=True,
        )
        for k in (10, 20, 50):
            topk = [u for u in ranked if u in outcome][:k]
            row.append(100 * mean([outcome[u] for u in topk]) if topk else float("nan"))
        print(f"  {m:<13}{row[0]:>8.1f}{row[1]:>8.1f}{row[2]:>8.1f}")
    base = 100 * sum(d for d in (outcome[u] for u in users)) / len(users)
    print(f"  {'(eval pool)':<13}{'':>8}{'':>8}{'':>8}  mean rate over pool = {base:.1f}%")

    # 4. per-subreddit
    id2sub = {}
    for f in glob.glob(os.path.join(RAW, "*_dataset.jsonl")):
        sub = os.path.basename(f).replace("_dataset.jsonl", "")
        for r in load_jsonl(f):
            id2sub[r["id"]] = sub
    print("\nPer-subreddit rho (best diffusion vs best producer):")
    wins = 0
    total = 0
    for sub in sorted(set(id2sub.values())):
        sp = [p for p in posts if id2sub.get(p["id"]) == sub]
        a2, t2, p2 = build_reply_forest(sp)
        ev = evaluate(a2, t2, p2, sp)
        if ev is None:
            print(f"  {sub:<20} (too few evaluable users, skipped)")
            continue
        sc2, oc2, us2 = ev
        rr = rho_all(sc2, oc2, us2)
        bp = max(PRODUCER, key=lambda m: rr[m])
        bd = max(DIFFUSION, key=lambda m: rr[m])
        total += 1
        win = rr[bd] > rr[bp]
        wins += win
        print(
            f"  {sub:<20} diffusion {rr[bd]:+.3f} ({bd})  vs producer {rr[bp]:+.3f} ({bp})  "
            f"-> {'diffusion' if win else 'producer'} (n={len(us2)})"
        )
    print(f"  diffusion wins in {wins}/{total} subreddits")

    # 5. label-perturbation robustness
    print("\nLabel-perturbation robustness (flip 15% of toxicity labels, 20 runs):")
    pr = random.Random(7)
    keep = 0
    for _ in range(20):
        tox2 = {c: (not tox[c]) if pr.random() < 0.15 else tox[c] for c in tox}
        posts2 = [dict(p, prediction=("toxic" if tox2[p["id"]] else "non-toxic")) for p in posts]
        sc2, _ = build_scores(author, tox2, parent, posts2)
        dt2, dx2 = desc_rate(author, tox2, parent)
        us2 = [u for u in sc2["ToxicCount"] if sc2["ToxicCount"][u] >= 1 and dt2[u] >= MIN_DESC]
        if len(us2) < 20:
            continue
        oc2 = {u: dx2[u] / dt2[u] for u in us2}
        rd = spearman([sc2[best_diff].get(u, 0.0) for u in us2], [oc2[u] for u in us2])
        rp = spearman([sc2[best_prod].get(u, 0.0) for u in us2], [oc2[u] for u in us2])
        if rd is not None and rp is not None and rd > rp:
            keep += 1
    print(f"  diffusion still beats producer in {keep}/20 noisy-label runs")


if __name__ == "__main__":
    main()
