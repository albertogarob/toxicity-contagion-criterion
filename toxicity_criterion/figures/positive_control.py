"""Positive-control figure for the criterion (paper Fig. 3).

Reuses the generative sweep of :mod:`~toxicity_criterion.criterion.disattenuation`: on the real
reply forest, plant a known parent->child contagion (dial: true RR), corrupt labels at the
classifier's measured (Se, Sp), and measure the largest toxic component the criterion recovers.
The figure shows the criterion lights up when contagion is planted yet the real corpus sits at
the no-contagion floor (largest component 7, within the permutation null): a true negative.

Output: ``figures/fig_poscontrol.pdf``. Run with ``PYTHONHASHSEED=0`` for reproducibility:
``PYTHONHASHSEED=0 python -m toxicity_criterion.figures.positive_control``.
"""

from __future__ import annotations

import os
import sys
from statistics import mean, pstdev

# Pin the hash seed before anything stochastic, mirroring the disattenuation sweep.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.execvpe(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .. import config  # noqa: E402
from ..criterion.disattenuation import (  # noqa: E402
    assortativity,
    classifier_rates,
    references,
    simulate,
)
from ..forest import build_reply_forest, largest_toxic_component, reply_edges, topo_order  # noqa: E402
from ..io import load_corpus  # noqa: E402

# Observed-on-real and permutation null for the largest toxic component (from local_effect).
OBS_MAXCOMP = 7
NULL_MEAN, NULL_SD = 7.5, 2.0

RR_GRID = [1, 2, 4, 8, 16, 32, 64]
SIMS = 12
SCENARIOS = [
    ("Higher recall (Se=0.77)", "unanimous(3of3)", "#2c6fb3", "o"),
    ("Lower recall (Se=0.32)", "panel-majority(2of3)", "#c0504d", "s"),
]


def sweep(order, author, parent, edges, Se, Sp, p):
    xs, ys, yerr, obs_a = [], [], [], []
    for rr in RR_GRID:
        mc, oa = [], []
        for s in range(SIMS):
            _, obs = simulate(order, author, parent, p, rr, Se, Sp, 100 * rr + s)
            mc.append(largest_toxic_component(author, obs, parent))
            oa.append(assortativity(edges, obs))
        xs.append(rr)
        ys.append(mean(mc))
        yerr.append(pstdev(mc))
        obs_a.append(mean(oa))
    return xs, ys, yerr, obs_a


def main() -> None:
    author, tox, parent = build_reply_forest(load_corpus(config.SONNET_TOXIC, config.SONNET_NONTOXIC))
    edges = reply_edges(author, parent)
    order = topo_order(author, parent)
    refs = references()

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.axhspan(
        NULL_MEAN - NULL_SD,
        NULL_MEAN + NULL_SD,
        color="0.85",
        zorder=0,
        label="Permutation null ($7.5\\pm2.0$)",
    )
    ax.axhline(OBS_MAXCOMP, color="black", ls="--", lw=1.3, zorder=5, label="Observed (real corpus) = 7")

    for name, refkey, color, marker in SCENARIOS:
        Se, Sp, p, _ = classifier_rates(refs[refkey])
        xs, ys, yerr, _ = sweep(order, author, parent, edges, Se, Sp, p)
        lo = [max(0.0, y - e) for y, e in zip(ys, yerr)]
        hi = [y + e for y, e in zip(ys, yerr)]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, zorder=8, linewidth=0)
        ax.plot(xs, ys, marker=marker, ms=4.5, lw=1.8, color=color, label=name, zorder=10)

    ax.set_xscale("log")
    ax.set_xticks(RR_GRID)
    ax.set_xticklabels([str(r) for r in RR_GRID])
    ax.set_xlabel("Introduced relative risk  $\\mathrm{RR}$ (parent$\\to$child)")
    ax.set_ylabel("Largest toxic component  $S$ (users)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    out = os.path.join(config.FIG_DIR, "fig_poscontrol.pdf")
    fig.savefig(out)
    print("wrote", out)

    for name, refkey, _, _ in SCENARIOS:
        Se, Sp, p, _ = classifier_rates(refs[refkey])
        xs, ys, yerr, obs_a = sweep(order, author, parent, edges, Se, Sp, p)
        print(f"\n{name}: Se={Se:.2f} Sp={Sp:.3f}")
        for x, y, e, oa in zip(xs, ys, yerr, obs_a):
            print(f"  true_RR={x:>5}  obs_assort={oa:.3f}  maxcomp={y:.1f}+-{e:.1f}")


if __name__ == "__main__":
    main()
