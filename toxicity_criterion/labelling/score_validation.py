"""
Score the agreement-stratified validation once human_labels.csv exists.

Computes, by Horvitz-Thompson reweighting over the 2x2 agreement cells (which partition
the GPT-OSS/Sonnet overlap), each classifier's in-domain precision / recall / F1 against
the human gold labels; the disagreement adjudication; and the random-stratum prevalence.

Run after labelling in label_app.html and downloading human_labels.csv into this folder.
"""

import csv
import json
from collections import Counter

from . import lab_paths as P

HERE = P.VAL

PREC_GATE = (
    0.80  # pre-registered: Sonnet is a usable reference if precision >= this and recall is not catastrophic
)


def norm(x):
    s = (x or "").strip().lower()
    if s in {"toxic", "t", "1", "yes", "y"}:
        return "toxic"
    if s in {"non-toxic", "nontoxic", "n", "0", "no", "clean"}:
        return "non-toxic"
    return ""


def main():
    key = {r["sample_id"]: r for r in csv.DictReader(open(HERE / "validation_key.csv", encoding="utf-8"))}
    pops = json.loads((HERE / "validation_populations.json").read_text())["cell_populations"]
    hl = HERE / "human_labels.csv"
    if not hl.exists():
        print("human_labels.csv not found. Label in label_app.html, click Download, move it here.")
        return
    human = {}
    for r in csv.DictReader(open(hl, encoding="utf-8")):
        h = norm(r.get("human_label"))
        if h:
            human[r["sample_id"]] = h

    # per-cell tallies (overlap cells only)
    CELLS = ["agree_tox", "gpt_tox_son_non", "gpt_non_son_tox", "agree_non"]
    cell_pred = {
        "agree_tox": ("toxic", "toxic"),
        "gpt_tox_son_non": ("toxic", "non-toxic"),
        "gpt_non_son_tox": ("non-toxic", "toxic"),
        "agree_non": ("non-toxic", "non-toxic"),
    }
    n_lab = Counter()
    n_tox = Counter()
    for sid, r in key.items():
        if r["stratum"] in CELLS and sid in human:
            n_lab[r["stratum"]] += 1
            if human[sid] == "toxic":
                n_tox[r["stratum"]] += 1

    # estimated toxic population per cell
    T = {}  # estimated # truly-toxic comments in cell
    Npop = {}
    for c in CELLS:
        Npop[c] = pops[c]
        T[c] = pops[c] * (n_tox[c] / n_lab[c]) if n_lab[c] else float("nan")

    total_toxic = sum(T.values())

    def metrics(clf_idx):  # 0 = gpt, 1 = sonnet
        predpos_cells = [c for c in CELLS if cell_pred[c][clf_idx] == "toxic"]
        predpos_pop = sum(Npop[c] for c in predpos_cells)
        tp = sum(T[c] for c in predpos_cells)
        prec = tp / predpos_pop if predpos_pop else float("nan")
        rec = tp / total_toxic if total_toxic else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec and rec) else float("nan")
        return prec, rec, f1, predpos_pop, tp

    print(f"Human labels: {len(human)} of 200 ({sum(1 for s in key if s not in human)} unlabeled/unsure)\n")
    print("Per-cell human-toxic share (n_toxic / n_labeled):")
    for c in CELLS:
        share = f"{n_tox[c]}/{n_lab[c]}" if n_lab[c] else "-/-"
        print(f"  {c:<16} pop={Npop[c]:>6}  human-toxic {share}  -> est. toxic in cell = {T[c]:.0f}")
    print(
        f"  estimated total toxic in overlap population: {total_toxic:.0f} "
        f"({100*total_toxic/sum(Npop.values()):.1f}% of {sum(Npop.values())})\n"
    )

    print(f"{'classifier':<10}{'precision':>11}{'recall':>9}{'F1':>7}")
    for name, idx in (("GPT-OSS", 0), ("Sonnet", 1)):
        p, r, f, _, _ = metrics(idx)
        print(f"{name:<10}{p:>11.3f}{r:>9.3f}{f:>7.3f}")

    # disagreement adjudication
    print("\nDisagreement adjudication (who do humans side with?):")
    for c, sider_if_toxic, sider_if_non in [
        ("gpt_tox_son_non", "GPT-OSS (toxic)", "Sonnet (non-toxic)"),
        ("gpt_non_son_tox", "Sonnet (toxic)", "GPT-OSS (non-toxic)"),
    ]:
        if n_lab[c]:
            ptox = n_tox[c] / n_lab[c]
            print(
                f"  {c}: {n_tox[c]}/{n_lab[c]} human-toxic -> "
                f"{100*ptox:.0f}% side with {sider_if_toxic}, {100*(1-ptox):.0f}% with {sider_if_non}"
            )

    # random stratum
    rand = [sid for sid, r in key.items() if r["stratum"] == "random" and sid in human]
    rtox = sum(1 for sid in rand if human[sid] == "toxic")
    if rand:
        print(
            f"\nRandom stratum (model-independent prevalence anchor): "
            f"{rtox}/{len(rand)} toxic ({100*rtox/len(rand):.1f}%)"
        )

    # decision gate
    ps, rs, fs, _, _ = metrics(1)
    print("\n--- DECISION GATE ---")
    ok = ps >= PREC_GATE
    print(
        f"Sonnet precision {ps:.3f} {'>=' if ok else '<'} {PREC_GATE} -> "
        f"{'PASS: usable as in-domain reference' if ok else 'FAIL: reconsider'}; recall {rs:.3f} "
        f"(check not catastrophically low)."
    )


if __name__ == "__main__":
    main()
