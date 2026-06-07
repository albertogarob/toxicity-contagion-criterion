"""The method-selection criterion as a single callable + CLI.

Given a reply forest and the classifier's ``(precision, recall)``, run Steps 1-3 and the decision
rule and output whether a producer-count baseline suffices or whether network methods may pay off.
This is the deployable form of the criterion: it has no dependency on the paper's corpus, so a
practitioner can run the pre-check on any new corpus.

  python -m toxicity_criterion.criterion.decision --demo --precision 0.92 --recall 0.32
  python -m toxicity_criterion.criterion.decision --demo --plant-rr 16 --precision 0.92 --recall 0.32
  python -m toxicity_criterion.criterion.decision --input forest.jsonl --precision 0.92 --recall 0.32 0.77
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Sequence

from ..forest import (
    build_reply_forest,
    chain_histogram,
    largest_toxic_component,
    longest_toxic_chain,
    reply_edges,
    topo_order,
)
from ..io import load_jsonl
from ..stats import null_summary, pearson

NPERM = 300
SEED = 0

# Decision-rule thresholds (documented, overridable). The paper reads the criterion
# qualitatively; these operationalize that reading. The case study (r_true ~0.1-0.17,
# component 7 ~= null 7.5, deepest chain 3, 94% isolated) lands on "producer-count".
R_TRUE_MEANINGFUL = 0.20  # corrected assortativity at/above this counts as "more than small"
CHAIN_ABS_CASCADE = 4  # a deepest consecutive-toxic chain >= this counts as a cascade
COMP_RATIO_CASCADE = 2.0  # largest toxic component >= this x the null mean counts as aggregation
SIG_P = 0.05  # permutation-null significance threshold


def _cond_and_r(edges, toxd) -> tuple[float, float]:
    pt = [c for p, c in edges if toxd[p]]
    cond = (sum(1 for c in pt if toxd[c]) / len(pt)) if pt else 0.0
    r = pearson([1 if toxd[p] else 0 for p, _ in edges], [1 if toxd[c] else 0 for _, c in edges])
    return cond, r


def disattenuate(r_obs: float, precision: float, recall: float, q: float) -> dict[str, float]:
    """Correct r_obs for classifier error from (precision, recall, observed positive rate q).

    Maps to (Se, Sp, true prevalence p) then kappa = (Se+Sp-1)*sqrt(p(1-p)/(q(1-q))) and
    r_true = r_obs / kappa^2.
    """
    Se = recall
    p = max(min(precision * q / recall, 0.999999), 1e-9) if recall else float("nan")
    Sp = 1.0 - q * (1.0 - precision) / (1.0 - p) if (1.0 - p) else float("nan")
    if not (0 < p < 1 and 0 < q < 1) or any(math.isnan(v) for v in (Se, Sp)):
        return {"Se": Se, "Sp": Sp, "p_true": p, "q_obs": q, "kappa": float("nan"), "r_true": float("nan")}
    kappa = (Se + Sp - 1) * math.sqrt(p * (1 - p) / (q * (1 - q)))
    return {
        "Se": Se,
        "Sp": Sp,
        "p_true": p,
        "q_obs": q,
        "kappa": kappa,
        "r_true": r_obs / (kappa**2) if kappa else float("nan"),
    }


def run_criterion(
    posts: list[dict],
    precision: float | None = None,
    recall: float | Sequence[float] | None = None,
    n_perm: int = NPERM,
    seed: int = SEED,
    r_true_meaningful: float = R_TRUE_MEANINGFUL,
    chain_abs_cascade: int = CHAIN_ABS_CASCADE,
    comp_ratio_cascade: float = COMP_RATIO_CASCADE,
    sig_p: float = SIG_P,
) -> dict:
    """Run Steps 1-3 + the decision rule. ``recall`` may be a float or a (lo, hi) band."""
    rng = random.Random(seed)
    author, tox, parent = build_reply_forest(posts)
    comments = list(author)
    n = len(comments)
    n_tox = sum(1 for c in comments if tox[c])
    base = n_tox / n if n else 0.0
    edges = reply_edges(author, parent)
    order = topo_order(author, parent)

    cond_obs, r_obs = _cond_and_r(edges, tox)
    comp_obs = largest_toxic_component(author, tox, parent)
    chain_obs = longest_toxic_chain(order, parent, tox)
    chain_hist = chain_histogram(order, parent, tox)
    frac_isolated = (chain_hist.get(1, 0) / n_tox) if n_tox else 0.0

    null_r: list[float] = []
    null_comp: list[float] = []
    null_chain: list[float] = []
    for _ in range(n_perm):
        perm = set(rng.sample(comments, n_tox))
        toxp = {c: (c in perm) for c in comments}
        _, r_ = _cond_and_r(edges, toxp)
        null_r.append(r_)
        null_comp.append(largest_toxic_component(author, toxp, parent))
        null_chain.append(longest_toxic_chain(order, parent, toxp))

    step1 = {
        "assortativity_obs": r_obs,
        "assortativity_null": null_summary(r_obs, null_r),
        "P_child_given_parent": cond_obs,
        "base_rate": base,
        "relative_risk": (cond_obs / base) if base else float("inf"),
    }

    step2 = None
    if precision is not None and recall is not None:
        recalls = recall if isinstance(recall, (list, tuple)) else [recall]
        band = [disattenuate(r_obs, precision, rc, base) for rc in recalls]
        r_true_vals = [b["r_true"] for b in band if not math.isnan(b["r_true"])]
        step2 = {
            "precision": precision,
            "recall": list(recalls),
            "scenarios": band,
            "r_true_min": min(r_true_vals) if r_true_vals else float("nan"),
            "r_true_max": max(r_true_vals) if r_true_vals else float("nan"),
        }

    step3 = {
        "largest_component_obs": comp_obs,
        "largest_component_null": null_summary(comp_obs, null_comp),
        "deepest_chain_obs": chain_obs,
        "deepest_chain_null": null_summary(chain_obs, null_chain),
        "frac_toxic_isolated": frac_isolated,
        "chain_histogram": dict(chain_hist),
    }

    local_effect = step1["assortativity_null"]["p"] < sig_p and step1["relative_risk"] > 1.0
    if step2 is not None and not math.isnan(step2["r_true_max"]):
        effect_meaningful = step2["r_true_max"] >= r_true_meaningful
        effect_basis = f"corrected r_true up to {step2['r_true_max']:.3f}"
    else:
        effect_meaningful = abs(r_obs) >= r_true_meaningful
        effect_basis = f"observed r_obs {r_obs:.3f} (uncorrected; provide precision/recall)"
    comp_aggregates = step3["largest_component_null"]["p"] < sig_p and comp_obs >= comp_ratio_cascade * max(
        step3["largest_component_null"]["mean"], 1e-9
    )
    chain_cascades = chain_obs >= chain_abs_cascade
    aggregates = comp_aggregates or chain_cascades
    use_network = bool(effect_meaningful and aggregates)

    decision = {
        "recommendation": "network methods may pay off" if use_network else "producer-count suffices",
        "use_network_methods": use_network,
        "local_effect_present": bool(local_effect),
        "effect_meaningful": bool(effect_meaningful),
        "effect_basis": effect_basis,
        "aggregates_into_cascades": bool(aggregates),
        "reasons": _reasons(
            local_effect,
            effect_meaningful,
            effect_basis,
            comp_aggregates,
            chain_cascades,
            comp_obs,
            step3,
            chain_obs,
            frac_isolated,
        ),
    }

    return {
        "corpus": {"comments": n, "toxic": n_tox, "toxic_rate": base, "reply_edges": len(edges)},
        "step1_local_effect": step1,
        "step2_corrected_effect": step2,
        "step3_cascade_structure": step3,
        "decision": decision,
        "params": {
            "n_perm": n_perm,
            "seed": seed,
            "r_true_meaningful": r_true_meaningful,
            "chain_abs_cascade": chain_abs_cascade,
            "comp_ratio_cascade": comp_ratio_cascade,
        },
    }


def _reasons(local, meaningful, basis, comp_agg, chain_casc, comp_obs, step3, chain_obs, frac_iso):
    cm = step3["largest_component_null"]["mean"]
    return [
        ("a statistically detectable" if local else "no detectable") + " parent->child local effect",
        (basis + " >= threshold" if meaningful else basis + " stays small (below threshold)"),
        f"largest toxic component {comp_obs} vs null ~{cm:.1f} "
        + ("(aggregates)" if comp_agg else "(at chance, no aggregation)"),
        f"deepest toxic chain {chain_obs} " + ("(cascades)" if chain_casc else "(too shallow for a cascade)"),
        f"{100 * frac_iso:.0f}% of toxic comments are isolated singletons",
    ]


def make_demo_forest(
    n_threads: int = 400, thread_size: int = 12, base_rate: float = 0.05, plant_rr: float = 1.0, seed: int = 0
) -> list[dict]:
    """Synthetic reply forest. ``plant_rr`` > 1 plants parent->child contagion (positive control)."""
    rng = random.Random(seed)
    posts: list[dict] = []
    cid = 0
    for _ in range(n_threads):
        root = cid
        cid += 1
        root_tox = rng.random() < base_rate
        posts.append(
            {"id": str(root), "author": f"u{rng.randint(0, 800)}", "parent_id": "t3_post", "toxic": root_tox}
        )
        prev_tox = {str(root): root_tox}
        for _ in range(thread_size - 1):
            this = cid
            cid += 1
            par = rng.randint(root, this - 1)
            ptox = prev_tox.get(str(par), False)
            prob = min(base_rate * plant_rr, 0.9) if ptox else base_rate
            t = rng.random() < prob
            posts.append(
                {"id": str(this), "author": f"u{rng.randint(0, 800)}", "parent_id": f"t1_{par}", "toxic": t}
            )
            prev_tox[str(this)] = t
    return posts


def print_report(v: dict) -> None:
    c = v["corpus"]
    print(
        f"Corpus: {c['comments']} comments, {c['toxic']} toxic ({100 * c['toxic_rate']:.2f}%), "
        f"{c['reply_edges']} reply edges\n"
    )
    s1 = v["step1_local_effect"]
    a = s1["assortativity_null"]
    print("Step 1 , local effect")
    print(
        f"  assortativity r_obs = {s1['assortativity_obs']:+.3f}  "
        f"(null {a['mean']:+.3f}+/-{a['sd']:.3f}, z={a['z']:.1f}, p={a['p']:.4f})"
    )
    print(
        f"  P(child|parent) = {100 * s1['P_child_given_parent']:.2f}%  vs base "
        f"{100 * s1['base_rate']:.2f}%  (RR {s1['relative_risk']:.2f}x)\n"
    )
    s2 = v["step2_corrected_effect"]
    print("Step 2 , corrected effect")
    if s2 is None:
        print("  (skipped; pass --precision and --recall)\n")
    else:
        for sc in s2["scenarios"]:
            print(
                f"  recall={sc['Se']:.2f} precision={s2['precision']:.2f}: Se={sc['Se']:.2f} "
                f"Sp={sc['Sp']:.3f} p_true={sc['p_true']:.3f} kappa={sc['kappa']:.3f}  ->  "
                f"r_true={sc['r_true']:.3f}"
            )
        print(
            f"  corrected assortativity range: r_true in "
            f"[{s2['r_true_min']:.3f}, {s2['r_true_max']:.3f}]\n"
        )
    s3 = v["step3_cascade_structure"]
    cc, dc = s3["largest_component_null"], s3["deepest_chain_null"]
    print("Step 3 , cascade structure")
    print(
        f"  largest toxic component = {s3['largest_component_obs']}  "
        f"(null {cc['mean']:.1f}+/-{cc['sd']:.1f}, z={cc['z']:.1f}, p={cc['p']:.4f})"
    )
    print(
        f"  deepest toxic chain = {s3['deepest_chain_obs']}  "
        f"(null {dc['mean']:.2f}+/-{dc['sd']:.2f}, z={dc['z']:.1f}, p={dc['p']:.4f})"
    )
    print(f"  toxic comments isolated: {100 * s3['frac_toxic_isolated']:.0f}%\n")
    d = v["decision"]
    print("=" * 64)
    print(f"DECISION: {d['recommendation'].upper()}  (use network methods: {d['use_network_methods']})")
    for r in d["reasons"]:
        print(f"   - {r}")
    print("=" * 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Contagion-based method-selection criterion.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSONL reply forest (id, parent_id, prediction|toxic per line)")
    src.add_argument("--demo", action="store_true", help="run on synthetic data")
    ap.add_argument("--precision", type=float, default=None, help="classifier precision (PPV)")
    ap.add_argument(
        "--recall",
        type=float,
        nargs="+",
        default=None,
        help="classifier recall; pass two values for a corrected-effect band",
    )
    ap.add_argument(
        "--plant-rr",
        type=float,
        default=1.0,
        help="[demo] plant parent->child contagion at this relative risk",
    )
    ap.add_argument("--n-perm", type=int, default=NPERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--json", action="store_true", help="print the full verdict as JSON")
    args = ap.parse_args()

    posts = make_demo_forest(plant_rr=args.plant_rr, seed=args.seed) if args.demo else load_jsonl(args.input)
    recall = args.recall[0] if (args.recall and len(args.recall) == 1) else args.recall
    v = run_criterion(posts, precision=args.precision, recall=recall, n_perm=args.n_perm, seed=args.seed)
    print(json.dumps(v, indent=2) if args.json else "", end="")
    if not args.json:
        print_report(v)


if __name__ == "__main__":
    main()
