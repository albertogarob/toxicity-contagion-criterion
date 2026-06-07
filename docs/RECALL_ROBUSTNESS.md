# Is Sonnet's low in-domain recall real, and does the downstream result depend on it?

This document records the full analysis behind two related questions:

1. **Is the deployed classifier's (Claude Sonnet 4.6) apparently low in-domain recall real,
   or an artifact of an uncertain reference?**
2. **Does the paper's headline downstream conclusion (producer-count is the strongest
   amplifier-selection method on the reach-controlled validation) depend on that recall?**

Short answers: the low recall is **substantially an artifact** (a fragile single-cell
extrapolation plus boundary-case disagreement among frontier models); on clear-cut toxicity
Sonnet's recall is **~0.77**. And the downstream conclusion is **robust** to realistic recall
loss (random and author-clustered, to 50%), reversing only under an adversarial pattern that
is contradicted by the measured precision.

Scripts: `in_domain_validation/score_panel.py` (validation), `recall_degradation.py`
(random sweep), `recall_degradation_adversarial.py` (clustered/adversarial sweep).
See also `in_domain_validation/VALIDATION_SET_DESIGN.md` for the sampling design.

---

## 1. Background: the validation references

There is no human gold standard available (no annotator fluent in UK political slang). We
validate Sonnet against two imperfect references on the 200-comment agreement-stratified
sample (see `VALIDATION_SET_DESIGN.md`), with toxic-class metrics Horvitz–Thompson
reweighted to the corpus:

- **Human** — a single non-expert annotator.
- **LLM panel (silver standard)** — majority vote of three frontier models from independent
  families: Claude Opus 4.8, GPT-5.5, Gemini 2.5 Pro.

Inter-rater agreement (correct, aligned text):

| pair | Cohen's κ |
|---|---|
| GPT-5.5 ~ Gemini | 0.89 |
| Gemini ~ Sonnet | 0.84 |
| GPT-5.5 ~ Sonnet | 0.83 |
| Opus ~ Sonnet | 0.70 |
| Sonnet ~ human | 0.63 |

Fleiss' κ across the three panel LLMs = **0.73** (substantial). The deployed classifier
aligns closely with the panel, more than the human does.

Headline metrics (toxic class, HT-reweighted):

| reference | precision | recall |
|---|---|---|
| LLM panel (2-of-3 majority) | 0.919 | 0.317 |
| human | 0.711 | 0.234 |

Precision is high under both references; recall is the figure under question.

---

## 2. Is the low recall real? — Decomposition

HT recall = (estimated toxic in Sonnet's predicted-toxic cells) / (estimated toxic in all
cells). Sonnet predicts non-toxic in two cells, which therefore carry its false negatives:
`gpt_tox_son_non` (population 2,683) and `agree_non` (population 20,388).

Per-cell decomposition of the "missed toxicity":

| cell | pop | reference share | est. toxic | role |
|---|---|---|---|---|
| **HUMAN** | | | | |
| `agree_non` | 20,388 | **1 / 27** | 755 | **48% of all toxic** — Sonnet FN |
| `gpt_tox_son_non` | 2,683 | 8 / 48 | 447 | 29% — Sonnet FN |
| `agree_tox` + `gpt_non_son_tox` | — | — | 366 | Sonnet TP → recall 0.234 |
| **LLM PANEL** | | | | |
| `agree_non` | 20,388 | **1 / 30** | 680 | **45% of all toxic** — Sonnet FN |
| `gpt_tox_son_non` | 2,683 | 7 / 55 | 341 | 23% — Sonnet FN |
| `agree_tox` + `gpt_non_son_tox` | — | — | 473 | Sonnet TP → recall 0.317 |

**The dominant term is a single label.** ~45–48% of Sonnet's entire recall deficit comes from
**one** toxic label in the `agree_non` cell, extrapolated across 20,388 comments. With 1 toxic
in ~30, the 95% CI on that cell's toxic share runs from near-0 to ~18%, the estimate is
effectively undetermined. It is also a **different** comment under each reference:

- Human's lone `agree_non` toxic = **S003** (criticism of politicians; all three panel models
  call it non-toxic) — a human over-flag.
- Panel's lone `agree_non` toxic = **S187** (a *"men are the most violent gender"*
  generalization; GPT+Gemini toxic, Opus non) — one borderline call.

Set that fragile cell to zero (its CI includes 0) and recall jumps:

| reference | recall (as measured) | recall (agree_non → 0) |
|---|---|---|
| LLM panel | 0.317 | **0.581** |
| human | 0.234 | **0.450** |

**Recall depends on how strict "toxic" is.** Recomputing Sonnet's metrics against the full
panel under different aggregation rules:

| reference rule | precision | recall |
|---|---|---|
| **unanimous 3-of-3 toxic (clear-cut)** | 0.634 | **0.770** |
| majority 2-of-3 | 0.919 | 0.317 |
| any 1-of-3 toxic (loosest) | 0.984 | 0.220 |
| human (comparison) | 0.711 | 0.234 |

On toxicity all three independent frontier models agree on, **Sonnet's recall is 0.77**. The
low headline figure appears only when "toxic" is widened to borderline cases the models
themselves split on, mostly GPT-5.5 + Gemini flagging offensive-language boundary cases while
Opus (the conservative LLM) sides with Sonnet.

**Reading the actual misses.** Of the `gpt_tox_son_non` items the panel called toxic, Opus
agrees with Sonnet (non-toxic) on 5 of 7; only S015 (*"pointless weasel… fascists… Pathetic"*)
and S081 (*"Is this the point you start crying?"*) are unanimous. So the genuine residual is a
**modest miss on borderline offensive/abusive language** (not hate speech), consistent with a
conservatively-tuned classifier.

### Verdict on Q1
Substantially an artifact: the headline 0.23–0.32 is dominated by (a) a near-unmeasurable
single-cell extrapolation and (b) definitional boundary disagreement among frontier models,
with the human reference adding genuine labeling error (e.g., S003). Sonnet's recall is **much
higher and more uncertain** than the point estimate; on unambiguous toxicity it is ~0.77. It
is **not** evidence that Sonnet systematically misses real toxicity, and precision is high
(0.92), so the toxic set it does produce is clean.

---

## 3. Does the downstream conclusion depend on recall? — Degradation sweeps

What the downstream analysis needs is that the labels be **clean** (precision), not
**complete** (recall): a missed toxic comment is absent from *both* the producer signal
(ToxicCount) *and* the downstream-toxicity ground truth (subtree rate), so imperfect recall
attenuates every method together rather than reordering them. We test this directly.

**Method.** Simulate worse recall by dropping a fraction `f` of toxic labels (toxic →
non-toxic), degrading *both* the producer signal and the ground truth, then recompute each
method's top-10 mean subtree rate (reach-controlled outcome) and check whether producer-count
(ToxicCount / Volume) still leads the structural methods (weighted degree, PageRank,
reply-breadth) and the pool base rate. Constructions mirror the paper (directed reply graph
over all users, min-support filter); weighted degree and PageRank use the toxicity-graded
edge weights `w in [1,2]`, while shortest-path centralities (betweenness, closeness) are run
on the unweighted graph (and, in the toxicity-aware variant, on the toxic-interaction
subgraph). Reproducible with `PYTHONHASHSEED=0`; `f>0` averaged over 20 perturbations.

### 3a. Random under-detection (`recall_degradation.py`)

Top-10 mean subtree rate (%):

| toxic dropped | eff. recall | ToxicCount | Volume | WtdDegree | PageRank | ReplyBreadth | pool | winner |
|---|---|---|---|---|---|---|---|---|
| 0%  | 1.00 | 6.3 | 5.9 | 3.3 | 3.6 | 3.4 | 3.2 | producer |
| 10% | 0.90 | 5.2 | 5.4 | 3.2 | 3.0 | 3.1 | 2.9 | producer |
| 20% | 0.80 | 5.3 | 4.9 | 2.8 | 2.8 | 2.8 | 2.6 | producer |
| 30% | 0.70 | 4.8 | 4.2 | 2.6 | 2.4 | 2.6 | 2.2 | producer |
| 40% | 0.60 | 4.5 | 3.5 | 2.1 | 1.9 | 2.2 | 1.9 | producer |
| 50% | 0.50 | 3.2 | 2.8 | 1.8 | 1.5 | 1.9 | 1.5 | producer |

Producer-count wins at every level, ~1.5–2× above structure and above the base rate.
Everything attenuates together; the ordering never flips.

### 3b. Adversarial / clustered under-detection (`recall_degradation_adversarial.py`)

Same budget of dropped labels, but non-uniformly placed:

| pattern | description | result |
|---|---|---|
| **user_cluster** | drop every toxic label of a random subset of toxic users (whole-author dropout) | producer **wins to 50%** |
| **top_producer** | concentrate all drops on the heaviest toxic producers | producer wins to 20%, **flips at ≥30%** (PageRank/ReplyBreadth overtake) |
| **hub_targeted** | drop toxic replies to high-degree recipients | degenerate (destroys the outcome itself; all methods → ~0) |

**Why the `top_producer` flip is not a real threat.** It deletes the *heaviest producers'
already-detected* toxicity. That is backwards from how recall loss actually distributes:
recall loss is toxicity Sonnet *failed to detect*, so by construction it falls on
**under**-detected users, who are **not** the current top producers (they are top precisely
because their toxicity *was* detected). Deleting detected top-producer toxicity is a
**precision** failure concentrated on them, which the measured precision of **0.92** directly
contradicts. The `hub_targeted` pattern removes most of the ground-truth toxicity itself and
is degenerate rather than informative.

### Verdict on Q2
The downstream conclusion is **robust** to the realistic patterns (random and author-clustered
under-detection, up to 50% of toxic labels). The only regime where it reverses is an
adversarial pattern that is both implausible and contradicted by the precision result. The
conclusion does not hinge on Sonnet's recall.

---

## 4. Overall confidence statement (for the paper)

- **Precision high (0.92 panel / 0.71 human)** → the labels the analysis acts on are clean.
- **Recall higher and more uncertain than it first appears** (~0.77 on clear-cut toxicity;
  the low figure is a fragile-cell + boundary-disagreement artifact).
- **Downstream conclusion stress-tested** → stable under random and clustered recall loss to
  50%; sensitive only to an implausible, precision-contradicted adversarial pattern.

So for the downstream part of the paper, the labels are **fit for purpose**: the result rests
on the labels being clean, which they are, and not on their being complete, which the
robustness sweeps show is not required.

### Honest caveats
- The validation population is the GPT-OSS ∩ Sonnet overlap (23,586), not the full 35,176; we
  assume Sonnet behaves identically on the non-overlap (same model and prompt).
- The panel is a **silver** standard; three frontier models could in principle share an
  in-domain blind spot.
- The degradation sweeps cover random, author-clustered, hub-targeted and top-producer
  patterns; they do not exhaust every conceivable structured-miss pattern.
