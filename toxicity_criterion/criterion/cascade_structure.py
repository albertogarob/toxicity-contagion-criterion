"""Criterion Step 3 , does toxicity aggregate into cascades?

A second, independent operationalization of "no cascades", complementary to the component-size
test in :mod:`~toxicity_criterion.criterion.local_effect`. This measures cascade *depth*: the
longest run of consecutive toxic comments along any root-to-leaf reply path, against a
label-permutation null. A genuine cascade would show toxic chains deeper than chance.

Run: ``python -m toxicity_criterion.criterion.cascade_structure``
(reproduces the paper's deepest chain = 3, null 2.02+/-0.18; 582 singletons / 17 pairs / 1 triple).
"""

from __future__ import annotations

import random

from .. import config
from ..forest import build_reply_forest, chain_histogram, longest_toxic_chain, topo_order
from ..io import load_corpus
from ..stats import null_summary

NPERM = 300
SEED = 0


def run(
    toxic_path: str = config.SONNET_TOXIC,
    nontoxic_path: str = config.SONNET_NONTOXIC,
    n_perm: int = NPERM,
    seed: int = SEED,
) -> dict:
    """Compute the deepest consecutive-toxic chain vs the permutation null; return a result dict."""
    rng = random.Random(seed)
    author, tox, parent = build_reply_forest(load_corpus(toxic_path, nontoxic_path))
    comments = list(author)
    order = topo_order(author, parent)
    n_tox = sum(1 for c in comments if tox[c])

    obs = longest_toxic_chain(order, parent, tox)
    hist = chain_histogram(order, parent, tox)

    null: list[float] = []
    for _ in range(n_perm):
        perm = set(rng.sample(comments, n_tox))
        toxp = {c: (c in perm) for c in comments}
        null.append(longest_toxic_chain(order, parent, toxp))

    return {
        "comments": len(comments),
        "toxic": n_tox,
        "deepest_chain": obs,
        "deepest_chain_null": null_summary(obs, null),
        "chain_histogram": {int(k): int(v) for k, v in sorted(hist.items())},
    }


def main() -> None:
    res = run()
    s = res["deepest_chain_null"]
    print(f"comments={res['comments']}  toxic={res['toxic']}\n")
    print("Longest consecutive-toxic reply chain (cascade depth):")
    print(
        f"  observed = {res['deepest_chain']}   null = {s['mean']:.2f} +/- {s['sd']:.2f}   "
        f"z = {s['z']:.1f}   empirical p = {s['p']:.3f}"
    )
    print(f"  -> toxic chains run {'DEEPER' if s['z'] > 2 else 'no deeper'} than chance.\n")
    print("Observed maximal toxic-chain lengths (length: count):")
    for length, count in res["chain_histogram"].items():
        print(f"  length {length}: {count}")


if __name__ == "__main__":
    main()
