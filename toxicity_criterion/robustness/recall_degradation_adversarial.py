"""
Non-random (adversarial) recall-degradation robustness.

recall_degradation.py dropped toxic labels at random. But Sonnet's real misses concentrate
on borderline *offensive language*, not hate speech; if that flavour were concentrated among
particular users or graph positions, a *structured* miss pattern could behave differently
from random. Here we drop the SAME budget of toxic labels under adversarial, clustered
patterns and re-check whether producer-count still tops the rate-based validation.

Patterns (each drops ~f of all toxic labels, but non-uniformly):
  top_producer : concentrate ALL dropped labels on the HEAVIEST toxic producers (greedily
                 strip the top producers' toxicity until ~f is removed). This is the worst
                 case for ToxicCount: it sabotages exactly the signal producer-count uses.
  user_cluster : drop every toxic label of a random subset of toxic users (whole-user
                 dropout) until ~f removed -- misses clustered by author, not by comment.
  hub_targeted : drop toxic replies preferentially where the *recipient* is a high-degree
                 user, shifting the missed toxicity toward structural hubs (favours the
                 structural methods, against producer-count).

If producer-count still wins under these, the headline conclusion is robust not just to
random under-detection but to plausibly biased under-detection.
"""

import os
import random
import sys
from collections import Counter
from statistics import mean, pstdev

from ..forest import build_reply_forest
from ..io import load_jsonl
from .recall_degradation import FRACS, NONTOXIC, PRODUCERS, SEEDS, TOP_K, TOXIC, evaluate

METHODS = ["ToxicCount", "Volume", "WtdDegree", "PageRank", "ReplyBreadth"]


def toxic_comment_ids(author, toxic):
    return [c for c, t in toxic.items() if t]


def drop_top_producer(author, base_toxic, f, seed):
    """Strip whole authors' toxicity, heaviest producers first, until ~f of toxic removed."""
    tc = Counter()
    for c, t in base_toxic.items():
        if t:
            tc[author[c]] += 1
    total = sum(tc.values())
    budget = int(round(f * total))
    drop_authors, removed = set(), 0
    for a, n in tc.most_common():  # heaviest producers first
        if removed >= budget:
            break
        drop_authors.add(a)
        removed += n
    return {c: (t and author[c] not in drop_authors) for c, t in base_toxic.items()}


def drop_user_cluster(author, base_toxic, f, seed):
    """Drop every toxic label of a RANDOM subset of toxic users until ~f removed."""
    rng = random.Random(seed)
    tc = Counter()
    for c, t in base_toxic.items():
        if t:
            tc[author[c]] += 1
    authors = list(tc)
    rng.shuffle(authors)
    total = sum(tc.values())
    budget = int(round(f * total))
    drop_authors, removed = set(), 0
    for a in authors:
        if removed >= budget:
            break
        drop_authors.add(a)
        removed += tc[a]
    return {c: (t and author[c] not in drop_authors) for c, t in base_toxic.items()}


def drop_hub_targeted(author, base_toxic, parent, f, seed):
    """Drop toxic replies preferentially where the parent's author is high reply-degree."""
    rng = random.Random(seed)
    indeg = Counter()
    for c in author:
        p = parent.get(c)
        if p is not None and p in author and author[p] != author[c]:
            indeg[author[p]] += 1
    tox_ids = toxic_comment_ids(author, base_toxic)

    # weight each toxic comment by the recipient(parent-author) degree; drop highest-weight first
    def w(c):
        p = parent.get(c)
        return indeg[author[p]] if (p in author) else 0

    ordered = sorted(tox_ids, key=lambda c: (w(c), rng.random()), reverse=True)
    budget = int(round(f * len(tox_ids)))
    drop = set(ordered[:budget])
    return {c: (t and c not in drop) for c, t in base_toxic.items()}


def run(mode, author, base_toxic, parent):
    print(f"\n=== mode: {mode} ===")
    print("  f      " + "".join(f"{m:>13}" for m in METHODS) + f"{'pool':>9}  winner")
    for f in FRACS:
        seeds = [0] if (f == 0 or mode == "top_producer") else list(range(SEEDS))
        rows = []
        for s in seeds:
            if f == 0:
                tox = base_toxic
            elif mode == "top_producer":
                tox = drop_top_producer(author, base_toxic, f, s)
            elif mode == "user_cluster":
                tox = drop_user_cluster(author, base_toxic, f, s)
            else:
                tox = drop_hub_targeted(author, base_toxic, parent, f, s)
            r = evaluate(tox, author, parent)
            if r:
                rows.append(r)
        if not rows:
            continue
        agg = {m: mean([r[m] for r in rows]) for m in METHODS + ["_pool"]}
        sd = {m: (pstdev([r[m] for r in rows]) if len(rows) > 1 else 0.0) for m in METHODS}
        best = max(METHODS, key=lambda m: agg[m])
        prod_best = max(PRODUCERS, key=lambda m: agg[m])
        win = "producer" if best in PRODUCERS else f"** {best} **"
        cells = "".join(f"{agg[m]:>8.1f}±{sd[m]:>3.1f}" for m in METHODS)
        extra = "" if best in PRODUCERS else f"  (>{prod_best} {agg[prod_best]:.1f})"
        print(f"  {f:<4.1f}  {cells}{agg['_pool']:>9.1f}  {win}{extra}")


def main():
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execvpe(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})
    posts = load_jsonl(TOXIC) + load_jsonl(NONTOXIC)
    author, base_toxic, parent = build_reply_forest(posts)
    print(
        f"{len(author)} comments, {sum(base_toxic.values())} toxic. "
        f"Adversarial/clustered recall loss; top-{TOP_K} mean subtree rate (%)."
    )
    for mode in ("top_producer", "user_cluster", "hub_targeted"):
        run(mode, author, base_toxic, parent)
    print("\nIf 'winner' stays 'producer' across all three adversarial patterns, the")
    print("downstream conclusion is robust to biased (not merely random) under-detection.")


if __name__ == "__main__":
    main()
