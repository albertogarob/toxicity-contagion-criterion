"""Criterion Step 2 , correct the local effect for classifier error.

A noisy classifier attenuates any measured assortativity toward zero, so a corpus can look
non-contagious merely because the labels are imperfect. This module removes that artifact two
ways and shows the corrected effect stays small either way:

  1. **Classical disattenuation.** The correction-for-attenuation identity
     ``r_true = r_obs / kappa^2`` with ``kappa = (Se + Sp - 1) * sqrt(p(1-p) / (q(1-q)))``,
     where (Se, Sp, true prevalence p, observed positive rate q) are estimated by
     Horvitz-Thompson reweighting over the validation strata under several reference labellings
     (panel majority / unanimous / human).
  2. **Generative sweep (positive control).** Plant a known parent->child contagion at relative
     risk RR on the real forest, corrupt labels at the classifier's measured (Se, Sp), and find
     the true RR whose *observed* assortativity matches r_obs.

Run: ``PYTHONHASHSEED=0 python -m toxicity_criterion.criterion.disattenuation``
(reproduces the paper's closed-form r_true in [0.09, 0.31] and generative 0.07-0.19).
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
from statistics import mean

from .. import config
from ..forest import build_reply_forest, largest_toxic_component, reply_edges, topo_order
from ..io import load_corpus
from ..stats import pearson

R_OBS = 0.046  # observed parent~child assortativity (from local_effect.py)
SIMS = 8
RR_GRID = [1, 2, 3, 4, 6, 8, 10, 14, 20]

CELLS = ["agree_tox", "gpt_tox_son_non", "gpt_non_son_tox", "agree_non"]
SON_PRED = {"agree_tox": 1, "gpt_non_son_tox": 1, "gpt_tox_son_non": 0, "agree_non": 0}

_KEY_CSV = os.path.join(config.VAL_PRIVATE_DIR, "validation_key.csv")
_POPS_JSON = os.path.join(config.VAL_PUBLIC_DIR, "validation_populations.json")


def assortativity(edges, toxd) -> float:
    return pearson([1 if toxd[p] else 0 for p, _ in edges], [1 if toxd[c] else 0 for _, c in edges])


def _norm(x: str) -> str:
    s = (x or "").strip().lower()
    return "toxic" if s == "toxic" else ("non-toxic" if s in ("non-toxic", "nontoxic") else "")


def load_col(path: str, col: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            v = _norm(r.get(col))
            if v:
                out[r["sample_id"]] = v
    return out


def classifier_rates(reference: dict[str, str]):
    """Return (Se, Sp, true prevalence p, observed positive rate q) via HT over the strata."""
    key = {r["sample_id"]: r for r in csv.DictReader(open(_KEY_CSV))}
    pops = json.load(open(_POPS_JSON))["cell_populations"]
    share = {}
    for c in CELLS:
        ids = [s for s in key if key[s]["stratum"] == c and reference.get(s)]
        share[c] = (sum(1 for s in ids if reference[s] == "toxic") / len(ids)) if ids else 0.0
    TP = FP = FN = TN = 0.0
    for c in CELLS:
        N = pops[c]
        t = share[c] * N
        nt = N - t
        if SON_PRED[c] == 1:
            TP += t
            FP += nt
        else:
            FN += t
            TN += nt
    N = TP + FP + FN + TN
    Se = TP / (TP + FN) if (TP + FN) else float("nan")
    Sp = TN / (TN + FP) if (TN + FP) else float("nan")
    return Se, Sp, (TP + FN) / N, (TP + FP) / N


def kappa(Se: float, Sp: float, p: float, q: float) -> float:
    if not (0 < p < 1 and 0 < q < 1):
        return float("nan")
    return (Se + Sp - 1) * math.sqrt(p * (1 - p) / (q * (1 - q)))


def references() -> dict[str, dict[str, str]]:
    opus = load_col(os.path.join(config.VAL_PUBLIC_DIR, "panel_opus.csv"), "opus_label")
    gpt = load_col(os.path.join(config.VAL_PUBLIC_DIR, "panel_gpt.csv"), "gpt_label")
    gem = load_col(os.path.join(config.VAL_PUBLIC_DIR, "panel_gemini.csv"), "gemini_label")
    human = load_col(os.path.join(config.VAL_PRIVATE_DIR, "human_labels.csv"), "human_label")
    sids = set(opus) | set(gpt) | set(gem)

    def majority(thr):
        out = {}
        for s in sids:
            vs = [d.get(s) for d in (opus, gpt, gem)]
            if all(vs):
                out[s] = "toxic" if sum(v == "toxic" for v in vs) >= thr else "non-toxic"
        return out

    return {"panel-majority(2of3)": majority(2), "unanimous(3of3)": majority(3), "human": human}


def simulate(order, author, parent, p_base, rr_true, Se, Sp, seed):
    """Plant contagion at rr_true on the real forest, then corrupt labels at (Se, Sp)."""
    rng = random.Random(seed)
    true = {}
    r0, r1 = p_base, min(1.0, rr_true * p_base)
    for c in order:
        par = parent.get(c)
        pr = r1 if (par in true and true[par]) else r0
        true[c] = rng.random() < pr
    obs = {c: (rng.random() < Se) if true[c] else (rng.random() < (1 - Sp)) for c in author}
    return true, obs


def run(toxic_path: str = config.SONNET_TOXIC, nontoxic_path: str = config.SONNET_NONTOXIC) -> dict:
    """Closed-form + generative disattenuation. Returns the per-scenario kappa/r_true and ranges."""
    author, tox, parent = build_reply_forest(load_corpus(toxic_path, nontoxic_path))
    edges = reply_edges(author, parent)
    refs = references()

    closed = {}
    for name, ref in refs.items():
        Se, Sp, p, q = classifier_rates(ref)
        k = kappa(Se, Sp, p, q)
        rt = R_OBS / (k * k) if k == k and k != 0 else float("nan")
        closed[name] = {"Se": Se, "Sp": Sp, "p_true": p, "kappa": k, "r_true": rt}

    # agree_non -> 0 variant (the fragile cell zeroed)
    maj = refs["panel-majority(2of3)"]
    key = {r["sample_id"]: r["stratum"] for r in csv.DictReader(open(_KEY_CSV))}
    maj0 = {s: ("non-toxic" if key.get(s) == "agree_non" else v) for s, v in maj.items()}
    Se, Sp, p, q = classifier_rates(maj0)
    k = kappa(Se, Sp, p, q)
    closed["agree_non->0"] = {"Se": Se, "Sp": Sp, "p_true": p, "kappa": k, "r_true": R_OBS / (k * k)}

    r_true_vals = [v["r_true"] for v in closed.values()]
    closed_range = (min(r_true_vals), max(r_true_vals))

    # generative sweep
    order = topo_order(author, parent)
    sweep = {}
    for label in ("panel-majority(2of3)", "unanimous(3of3)"):
        Se, Sp, p = closed[label]["Se"], closed[label]["Sp"], closed[label]["p_true"]
        rows = []
        for rr in RR_GRID:
            ta, oa, mc = [], [], []
            for s in range(SIMS):
                true, obs = simulate(order, author, parent, p, rr, Se, Sp, 100 * rr + s)
                ta.append(assortativity(edges, true))
                oa.append(assortativity(edges, obs))
                mc.append(largest_toxic_component(author, obs, parent))
            rows.append({"rr": rr, "true_assort": mean(ta), "obs_assort": mean(oa), "obs_maxcomp": mean(mc)})
        match = min(rows, key=lambda r: abs(r["obs_assort"] - R_OBS))
        sweep[label] = {"rows": rows, "match_true_assort": match["true_assort"], "match_rr": match["rr"]}

    gen_vals = [s["match_true_assort"] for s in sweep.values()]
    return {
        "r_obs": R_OBS,
        "closed_form": closed,
        "closed_form_range": closed_range,
        "generative": sweep,
        "generative_range": (min(gen_vals), max(gen_vals)),
    }


def main() -> None:
    res = run()
    print(f"observed assortativity r_obs = {res['r_obs']}\n")
    print("(1) CLASSICAL DISATTENUATION  (r_true = r_obs / kappa^2)")
    print(f"  {'reference':22}{'Se':>6}{'Sp':>7}{'p_true':>8}{'kappa':>7}{'r_true':>8}  reading")
    for name, v in res["closed_form"].items():
        rt = v["r_true"]
        tag = "negligible" if rt < 0.10 else ("small" if rt < 0.20 else "MODERATE")
        print(
            f"  {name:22}{v['Se']:>6.2f}{v['Sp']:>7.3f}{v['p_true']:>8.3f}{v['kappa']:>7.3f}{rt:>8.3f}  {tag}"
        )
    lo, hi = res["closed_form_range"]
    print(f"\n  -> disattenuated true assortativity range: r_true in [{lo:.3f}, {hi:.3f}]")
    print("\n(2) GENERATIVE SWEEP  (true contagion that reproduces r_obs through the noisy classifier)")
    for label, s in res["generative"].items():
        print(f"\n  scenario {label}:")
        print(f"    {'true RR':>8}{'true assort':>13}{'obs assort':>12}{'obs maxcomp':>12}")
        for r in s["rows"]:
            print(
                f"    {r['rr']:>8}{r['true_assort']:>13.3f}{r['obs_assort']:>12.3f}{r['obs_maxcomp']:>12.1f}"
            )
        print(
            f"    -> closest to r_obs={res['r_obs']}: true RR~{s['match_rr']}, "
            f"TRUE assortativity~{s['match_true_assort']:.3f} (observed real component = 7)"
        )


if __name__ == "__main__":
    main()
