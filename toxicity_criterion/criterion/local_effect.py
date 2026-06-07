"""Criterion Step 1 , is there any local (parent->child) toxic effect?

Tests whether toxicity correlates across reply edges against a label-permutation null
(relabel which comments are toxic, keeping the count and the reply structure fixed):

  * conditional risk ``P(child toxic | parent toxic)`` and the relative risk over the base rate;
  * the Pearson assortativity of ``(parent_toxic, child_toxic)`` over reply edges;
  * excess toxicity by hop distance to a toxic ancestor (descriptive);
  * the largest connected component of users joined by toxic exchanges (shared with Step 3).

Run: ``python -m toxicity_criterion.criterion.local_effect``
(reproduces the paper's r=+0.046, RR 3.35x, largest component 7).
"""

from __future__ import annotations

import random
from statistics import mean

from .. import config
from ..forest import build_reply_forest, largest_toxic_component, reply_edges
from ..io import load_corpus
from ..stats import null_summary, pearson

NPERM = 300
SEED = 0


def _cond_and_r(edges, toxd):
    pt = [c for p, c in edges if toxd[p]]
    cond = mean([1 if toxd[c] else 0 for c in pt]) if pt else 0.0
    r = pearson([1 if toxd[p] else 0 for p, _ in edges], [1 if toxd[c] else 0 for _, c in edges])
    return cond, r


def run(
    toxic_path: str = config.SONNET_TOXIC,
    nontoxic_path: str = config.SONNET_NONTOXIC,
    n_perm: int = NPERM,
    seed: int = SEED,
) -> dict:
    """Compute Step 1 statistics and their permutation-null summaries; return a result dict."""
    rng = random.Random(seed)
    author, tox, parent = build_reply_forest(load_corpus(toxic_path, nontoxic_path))
    comments = list(author)
    n_tox = sum(1 for c in comments if tox[c])
    base = n_tox / len(comments)
    edges = reply_edges(author, parent)

    cond_obs, r_obs = _cond_and_r(edges, tox)
    comp_obs = largest_toxic_component(author, tox, parent)

    null_cond: list[float] = []
    null_r: list[float] = []
    null_comp: list[float] = []
    for _ in range(n_perm):
        perm = set(rng.sample(comments, n_tox))
        toxp = {c: (c in perm) for c in comments}
        c_, r_ = _cond_and_r(edges, toxp)
        null_cond.append(c_)
        null_r.append(r_)
        null_comp.append(largest_toxic_component(author, toxp, parent))

    # descriptive hop-distance decay
    def ancestor(c, d):
        x = c
        for _ in range(d):
            x = parent.get(x)
            if x is None or x not in author:
                return None
        return x

    hops = {}
    for d in (1, 2, 3, 4):
        ta = [c for c in comments if (a := ancestor(c, d)) is not None and tox[a]]
        if ta:
            rate = mean([1 if tox[c] else 0 for c in ta])
            hops[d] = {"rate": rate, "ratio": rate / base, "n": len(ta)}

    return {
        "comments": len(comments),
        "toxic": n_tox,
        "base_rate": base,
        "reply_edges": len(edges),
        "P_child_given_parent": cond_obs,
        "relative_risk": cond_obs / base,
        "P_child_given_parent_null": null_summary(cond_obs, null_cond),
        "assortativity": r_obs,
        "assortativity_null": null_summary(r_obs, null_r),
        "largest_component": comp_obs,
        "largest_component_null": null_summary(comp_obs, null_comp),
        "hop_decay": hops,
    }


def main() -> None:
    res = run()
    print(
        f"comments={res['comments']}  reply-edges={res['reply_edges']}  "
        f"base toxic rate={res['base_rate']:.4f} ({100 * res['base_rate']:.2f}%)\n"
    )
    print("(1) Edge assortativity / contagion:")
    a = res["P_child_given_parent_null"]
    print(
        f"  P(child toxic | parent toxic) = {res['P_child_given_parent']:.4f} "
        f"({100 * res['P_child_given_parent']:.2f}%)  => relative risk {res['relative_risk']:.2f}x"
    )
    print(f"    null = {100 * a['mean']:.2f}% +/- {100 * a['sd']:.2f}   z={a['z']:.1f}   p={a['p']:.4f}")
    r = res["assortativity_null"]
    print(
        f"  toxic assortativity (Pearson parent~child) = {res['assortativity']:+.3f}   "
        f"null {r['mean']:+.3f}+/-{r['sd']:.3f}  z={r['z']:.1f}  p={r['p']:.4f}"
    )
    print("\n(2) Excess toxicity by hop distance to a toxic ancestor:")
    for d, h in res["hop_decay"].items():
        print(
            f"  hop {d}: P(toxic | ancestor@{d} toxic) = {100 * h['rate']:.2f}%  "
            f"({h['ratio']:.2f}x base)  n={h['n']}"
        )
    print("\n(3) Toxic-reply subgraph clustering:")
    c = res["largest_component_null"]
    print(
        f"  largest toxic-reply component = {res['largest_component']}   "
        f"null {c['mean']:.1f}+/-{c['sd']:.1f}  z={c['z']:.1f}  p={c['p']:.4f}"
    )


if __name__ == "__main__":
    main()
