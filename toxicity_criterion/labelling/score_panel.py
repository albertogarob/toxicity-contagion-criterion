"""
Score the 3-LLM silver-standard panel on the 200 validation comments.

Raters:
  VOTERS (panel)   : opus (claude-opus-4-8), gpt (gpt-5.5), gemini (gemini-3.1-pro-preview)
  SUBJECTS         : sonnet (claude-sonnet-4-6) and gpt-oss  -- evaluated, never voters
  human            : the (acknowledged-weak, non-UK-native) annotator, reported alongside

Reference label = 2-of-3 LLM majority (odd panel -> always decisive, no ties). The
classifier under evaluation (Sonnet) is NOT a voter, so this is not self-validation.

Outputs
  (1) Per-rater toxic rate on the 200.
  (2) Pairwise % agreement + Cohen's kappa across all five raters.
  (3) Fleiss' kappa for the LLM trio, and for trio+human (human non-unsure items).
  (4) Lone-dissenter ("outlier") counts: who disagrees with everyone else, how often.
      Specifically: is the HUMAN the outlier vs a tight LLM trio? Is GEMINI?
  (5) HT-reweighted, corpus-level precision/recall/F1 of Sonnet and GPT-OSS against
      (a) the LLM consensus and (b) the human -- reported as a RANGE, the honest headline.

Silver standard, not gold: three LLMs can share a UK-context blind spot, so consensus
!= truth. We report agreement so the reader sees how strong the reference actually is.
"""

import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from . import lab_paths as P

HERE = P.VAL
VOTERS = ["opus", "gpt", "gemini"]
ALL_RATERS = VOTERS + ["sonnet", "gptoss", "human"]
LABELS = ("non-toxic", "toxic")


def norm(x):
    s = (x or "").strip().lower()
    if s in {"toxic", "t", "1", "yes", "y"}:
        return "toxic"
    if s in {"non-toxic", "nontoxic", "n", "0", "no", "clean"}:
        return "non-toxic"
    return ""  # blank / unsure


def load_col(path, col):
    out = {}
    if not Path(path).exists():
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        v = norm(r.get(col))
        if v:
            out[r["sample_id"]] = v
    return out


def cohen_kappa(pairs):
    """pairs: list of (a,b) labels. Returns (n, %agree, kappa)."""
    n = len(pairs)
    if not n:
        return 0, float("nan"), float("nan")
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in LABELS)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")
    return n, po, kappa


def fleiss_kappa(rows):
    """rows: list of per-item Counters over LABELS, each summing to the same n raters."""
    rows = [r for r in rows if sum(r.values()) >= 2]
    if not rows:
        return float("nan"), 0
    n = sum(rows[0].values())
    N = len(rows)
    Pi = [(sum(c**2 for c in r.values()) - n) / (n * (n - 1)) for r in rows]
    Pbar = sum(Pi) / N
    pj = {k: sum(r[k] for r in rows) / (N * n) for k in LABELS}
    Pe = sum(v**2 for v in pj.values())
    return ((Pbar - Pe) / (1 - Pe) if Pe != 1 else float("nan")), N


def main():
    key = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "validation_key.csv", encoding="utf-8"))}
    pops = json.loads((HERE / "validation_populations.json").read_text())["cell_populations"]

    R = {
        "opus": load_col(HERE / "panel_opus.csv", "opus_label"),
        "gpt": load_col(HERE / "panel_gpt.csv", "gpt_label"),
        "gemini": load_col(HERE / "panel_gemini.csv", "gemini_label"),
        "sonnet": {s: norm(r["sonnet_label"]) for s, r in key.items() if norm(r["sonnet_label"])},
        "gptoss": {s: norm(r["gpt_label"]) for s, r in key.items() if norm(r["gpt_label"])},
        "human": load_col(HERE / "human_labels.csv", "human_label"),
    }
    ids = list(key)

    print("Coverage / toxic rate on the 200:")
    for name in ALL_RATERS:
        d = R[name]
        ntox = sum(1 for v in d.values() if v == "toxic")
        print(f"  {name:7} labeled {len(d):3}/200  toxic {ntox:3} ({100*ntox/max(len(d),1):.0f}%)")

    # ---- consensus = majority among AVAILABLE voters (>=2); ties (even split) unresolved ----
    consensus = {}
    unanimous = tievoid = 0
    for s in ids:
        vs = [R[v].get(s) for v in VOTERS if R[v].get(s)]
        if len(vs) < 2:
            continue
        c = Counter(vs)
        top, n = c.most_common(1)[0]
        if n * 2 == len(vs):  # exact tie (e.g. opus/gpt disagree, gemini absent)
            tievoid += 1
            continue
        consensus[s] = top
        unanimous += n == len(vs)
    cov = {v: sum(1 for s in ids if R[v].get(s)) for v in VOTERS}
    print("\nVoter coverage: " + ", ".join(f"{v}={cov[v]}" for v in VOTERS))
    print(
        f"LLM consensus resolved on {len(consensus)}/200 "
        f"(unanimous-among-present {unanimous}; {tievoid} unresolved ties where voters split evenly)."
    )

    # ---- pairwise agreement + Cohen kappa ----
    print("\nPairwise agreement / Cohen's kappa (overlap items):")
    print(f"  {'pair':<18}{'n':>5}{'%agree':>9}{'kappa':>8}")
    for a, b in combinations(ALL_RATERS, 2):
        common = [s for s in ids if R[a].get(s) and R[b].get(s)]
        pairs = [(R[a][s], R[b][s]) for s in common]
        n, po, k = cohen_kappa(pairs)
        print(f"  {a+'~'+b:<18}{n:>5}{100*po:>8.0f}%{k:>8.2f}")

    # ---- Fleiss kappa ----
    trio_rows = [Counter(R[v][s] for v in VOTERS) for s in ids if all(R[v].get(s) for v in VOTERS)]
    k3, N3 = fleiss_kappa(trio_rows)
    quad_rows = [
        Counter([R[v][s] for v in VOTERS] + [R["human"][s]])
        for s in ids
        if all(R[v].get(s) for v in VOTERS) and R["human"].get(s)
    ]
    k4, N4 = fleiss_kappa(quad_rows)
    print(f"\nFleiss' kappa  LLM trio (n=3): {k3:.3f}  over {N3} items")
    print(f"Fleiss' kappa  trio+human (n=4): {k4:.3f}  over {N4} items (human non-unsure)")

    # ---- lone-dissenter / outlier counts ----
    def lone(raters):
        cnt = Counter()
        tot = 0
        for s in ids:
            vals = {r: R[r].get(s) for r in raters if R[r].get(s)}
            if len(vals) < len(raters):
                continue
            tot += 1
            c = Counter(vals.values())
            if len(c) == 2 and min(c.values()) == 1:  # exactly one dissenter
                odd = [r for r, v in vals.items() if c[v] == 1][0]
                cnt[odd] += 1
        return cnt, tot

    lc3, t3 = lone(VOTERS)
    print(f"\nLone-dissenter among the LLM trio (over {t3} fully-voted items):")
    for r in VOTERS:
        print(f"  {r:7} is the odd one out {lc3[r]:3} times ({100*lc3[r]/max(t3,1):.0f}%)")
    lc4, t4 = lone(VOTERS + ["human"])
    print(f"\nLone-dissenter among trio+human (over {t4} items, human non-unsure):")
    for r in VOTERS + ["human"]:
        print(f"  {r:7} is the odd one out {lc4[r]:3} times ({100*lc4[r]/max(t4,1):.0f}%)")
    print("  ^ if 'human' dominates here, the weak link is the human labels, not Sonnet.")

    # ---- HT-reweighted corpus metrics of subjects vs a reference ----
    CELLS = ["agree_tox", "gpt_tox_son_non", "gpt_non_son_tox", "agree_non"]
    cell_pred = {
        "agree_tox": ("toxic", "toxic"),
        "gpt_tox_son_non": ("toxic", "non-toxic"),
        "gpt_non_son_tox": ("non-toxic", "toxic"),
        "agree_non": ("non-toxic", "non-toxic"),
    }

    def ht_metrics(ref, clf_idx):
        """ref: {sid->label} reference; clf_idx 0=gpt-oss,1=sonnet. HT over the 4 overlap cells."""
        n_lab, n_tox = Counter(), Counter()
        for sid, r in key.items():
            if r["stratum"] in CELLS and ref.get(sid):
                n_lab[r["stratum"]] += 1
                n_tox[r["stratum"]] += ref[sid] == "toxic"
        T, Npop = {}, {}
        for c in CELLS:
            Npop[c] = pops[c]
            T[c] = pops[c] * (n_tox[c] / n_lab[c]) if n_lab[c] else float("nan")
        total_tox = sum(T.values())
        ppos = [c for c in CELLS if cell_pred[c][clf_idx] == "toxic"]
        ppop = sum(Npop[c] for c in ppos)
        tp = sum(T[c] for c in ppos)
        prec = tp / ppop if ppop else float("nan")
        rec = tp / total_tox if total_tox else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else float("nan")
        return prec, rec, f1

    print("\nHT-reweighted corpus-level metrics (toxic class), by reference:")
    print(f"  {'reference':<12}{'subject':<9}{'precision':>11}{'recall':>9}{'F1':>7}")
    for refname, ref in (("consensus", consensus), ("human", R["human"])):
        for subj, idx in (("GPT-OSS", 0), ("Sonnet", 1)):
            p, rc, f = ht_metrics(ref, idx)
            print(f"  {refname:<12}{subj:<9}{p:>11.3f}{rc:>9.3f}{f:>7.3f}")
    print("  ^ Sonnet's recall/F1 RANGE across the two references is the honest headline.")

    # ---- where does the human sit vs the consensus ----
    common = [s for s in ids if R["human"].get(s) and consensus.get(s)]
    agree = sum(1 for s in common if R["human"][s] == consensus[s])
    h_tox_c_non = sum(1 for s in common if R["human"][s] == "toxic" and consensus[s] == "non-toxic")
    h_non_c_tox = sum(1 for s in common if R["human"][s] == "non-toxic" and consensus[s] == "toxic")
    print(
        f"\nHuman vs LLM consensus on {len(common)} items: agree {agree} ({100*agree/len(common):.0f}%); "
        f"human-toxic/consensus-non {h_tox_c_non}; human-non/consensus-toxic {h_non_c_tox}."
    )


if __name__ == "__main__":
    main()
