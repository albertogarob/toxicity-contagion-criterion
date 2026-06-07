# Toxicity contagion: is it negligible, or attenuated? A disattenuation analysis

This records the analysis of whether the observed parent→child toxicity correlation on the
reply forest is *practically negligible*, or merely *attenuated to look negligible* by the
classifier's imperfect recall. Conclusion: the contagion is **statistically significant and
practically small (not negligible, not moderate)** once corrected for label noise, and the
toxic structure it produces is **practically tiny by two independent measures** (largest
user-cluster 7, deepest reply-chain 3; 94% of toxic comments isolated), toxicity is local,
not a cascade.

Scripts: `contagion_analysis.py` (observed effect + permutation null),
`contagion_disattenuation.py` (this analysis). Classifier error rates come from the panel
validation (`in_domain_validation/score_panel.py`, see `RECALL_ROBUSTNESS.md`).

---

## 0. What these two methods are (plain language), and which data

Both methods solve a single problem: **noisy labels hide correlations.** When you measure two
things with an error-prone instrument, the correlation you *observe* is always weaker than the
*true* correlation, the errors blur the relationship toward zero.

> Analogy: measure everyone's height with a warped tape and their weight with a broken scale.
> Height and weight are genuinely correlated, but because both measurements are noisy, the
> correlation in your spreadsheet looks weaker than reality.

Here the error-prone instrument is the **Sonnet classifier** (toxic / non-toxic). It misses
some real toxicity (recall < 1), so the weak observed parent→child correlation (assortativity
= 0.046) might be a genuinely weak effect, *or* a stronger effect blurred down by the
classifier's mistakes. The two methods below distinguish those.

**Method 1 — Classical disattenuation (a correction formula).** A 120-year-old fix
(Spearman, 1904): `true correlation = observed correlation / reliability`. "Reliability"
(here `kappa`, 0–1) is how faithfully the labels track the truth; reliability 1 = perfect
labels, no correction; reliability 0.5 = the true correlation is double what you saw. Because
*both* ends of a parent→child pair are labelled by the same noisy classifier, the blurring
happens twice, so we divide by `kappa²`. `kappa` is computed from two numbers measured on the
validation set, the classifier's **sensitivity** (recall) and **specificity**. (Details and
formula in §2.)

**Method 2 — Generative sensitivity sweep (a simulation).** Instead of trusting a formula, we
build a world where we *control* the truth and see what it looks like through the noisy
classifier: (1) take the **real** reply trees; (2) plant toxicity with a *dial* for true
contagion strength; (3) **corrupt** those true labels at the classifier's measured error rate
(randomly flip labels as often as Sonnet gets it wrong); (4) measure the correlation we would
*observe*; (5) turn the dial until the simulated observed correlation matches the real one
(0.046). The dial setting that matches tells us the true contagion. This is more honest than
the formula because it respects the actual tree structure and also lets us check clustering.
(Details in §3.)

Both methods landing in the same place (~0.1–0.2) is why we trust the answer.

**Which data (no new data; two existing sets):**
- **The 35,176-comment UK Reddit corpus** (reply forest, Sonnet-labelled), used to measure the
  *observed* contagion (0.046) and as the tree structure for the simulation.
- **The 200-comment validation set** (human labels + the Opus / GPT-5.5 / Gemini panel), used
  only to measure the classifier's **error rates** (sensitivity, specificity), the inputs that
  drive both the formula and the simulation's corruption step.

In one line: the corpus tells us what we *see*; the validation set tells us how *wrong the
labels are*; the two methods combine them to estimate what is *actually true*.

---

## 1. The observed effect (Sonnet labels)

On the reply forest (35,176 comments, 15,264 reply edges), with `contagion_analysis.py`'s
label-permutation null (NPERM = 300):

| quantity | observed | null | significance |
|---|---|---|---|
| P(child toxic \| parent toxic) | 5.90% (vs base 1.76%) → RR 3.35× | 1.76% ± 0.80 | z = 5.2, p ≈ 0.003 |
| parent~child assortativity (Pearson) | **+0.046** | 0.000 ± 0.008 | z = 5.7, p ≈ 0.003 |
| largest toxic-reply component | **7** | 7.5 ± 2.0 | z = −0.3, p = 0.65 (NOT above chance) |

So: a statistically robust local parent→child correlation, but a tiny coefficient and no
larger-than-chance clustering. This is what tempted the "significant but practically
negligible" framing. The problem: label noise attenuates correlations toward zero, so
r_obs = 0.046 is a *lower bound* on the true effect.

**A second, independent cascade metric** (`contagion_chain_test.py`): the deepest run of
*consecutive* toxic comments along any root→leaf reply path (cascade depth):

| quantity | observed | null | significance |
|---|---|---|---|
| longest consecutive-toxic chain | **3** | 2.02 ± 0.18 | z = 5.4, p = 0.030 |

Chain-length histogram across the whole corpus: **582 isolated toxic comments, 17 toxic pairs,
1 toxic triple.** I.e. the deepest toxic cascade in 35k comments is **3 messages (one
instance)**, and **94%** of toxic comments have no toxic parent or child at all. This is
*statistically* deeper than chance (random labelling almost never stacks three toxic in a row,
hence the single depth-3 chain registers as significant) but *practically* trivial, the same
"significant yet tiny" pattern as the assortativity. The two metrics (cluster size, chain
depth) independently say the toxic structure is practically nil.

## 2. Classical disattenuation

For a binary label measured with sensitivity Se, specificity Sp at true prevalence p, the
true/observed-label correlation is

```
kappa = (Se + Sp - 1) * sqrt( p(1-p) / (q(1-q)) ),   q = p*Se + (1-p)(1-Sp).
```

Both endpoints of every reply edge are measured by the *same* classifier, so the observed
assortativity is attenuated by kappa² and `r_true = r_obs / kappa²`. We read (Se, Sp, p) from
the validation 2×2 (Sonnet vs reference, HT-reweighted) under several recall scenarios:

| reference (sets recall) | Se | Sp | p_true | kappa | **r_true** | reading |
|---|---|---|---|---|---|---|
| unanimous 3-of-3 (clear-cut) | 0.77 | 0.992 | 0.018 | 0.69 | **0.096** | negligible |
| agree_non → 0 | 0.58 | 0.998 | 0.035 | 0.72 | **0.088** | negligible |
| panel-majority (2-of-3) | 0.32 | 0.998 | 0.063 | 0.53 | **0.167** | small |
| human (least reliable) | 0.23 | 0.993 | 0.067 | 0.39 | **0.308** | moderate |

**Disattenuated range: r_true ∈ [0.09, 0.31].** The negligible end holds only under the
higher-recall references; the central/low-recall estimate is *small* (~0.10–0.17), and the
worst case (human, recall 0.23, the least trustworthy reference) is *moderate*.

## 3. Generative sweep (structure-aware)

On the real forest we assign TRUE toxic labels with a tunable parent→child contagion (true
RR), corrupt them with the measured (Se, Sp), and find which true contagion reproduces the
observed assortativity r_obs = 0.046:

| scenario | true RR matching r_obs | **TRUE assortativity** | obs max component (real = 7) |
|---|---|---|---|
| panel-majority (Se=0.32) | ~4 | **0.185** | ~10 |
| unanimous (Se=0.77) | ~6 | **0.073** | ~8 |

Agrees with the closed-form: true assortativity ~0.07–0.19 centrally. Note the joint
constraint: at the assortativity-matching point the model predicts a max component of 8–10,
yet we observe **7**, i.e. the real data shows *no more* clustering than weak contagion would,
pushing the estimate toward the lower end and confirming the absence of cascades.

## 4. Conclusion and reframe

- **"Practically negligible" does not survive disattenuation.** Corrected for classifier
  reliability, the true parent→child toxic assortativity is **small** (~0.1–0.2 centrally;
  bounded ≲ 0.31 in the worst case), not negligible and not moderate.
- **Robust claim: toxicity is local, not epidemic.** Two independent metrics say the toxic
  structure is practically tiny, largest user-cluster 7 (≈ chance), deepest reply-chain 3
  (94% of toxic comments isolated). The component-size claim is robust to label noise: even
  contagion strong enough to reproduce the observed correlation, passed through the measured
  error rate, would leave large components (~30–60 nodes at high true RR), which we never see.
  Toxic structure does not scale into cascades. (Cascade *depth* is marginally above a random
  null at z = 5.4, but the practical depth is 3, so this is "significant yet tiny," not
  evidence of real cascades.)
- **Mechanism for the paper's thesis.** Because toxicity does not aggregate into network
  structure, structural/network position has little to exploit, which is why content-blind
  centrality and reach-based methods do not beat ranking users by their own toxic production.
  The contagion result and the producer-count result are the same fact seen twice: toxicity is
  a property of individuals, not of the network.

**Publishable framing:** *statistically significant and practically small* (with the
disattenuation shown, which preempts the obvious referee attack), plus *no cascade structure*.
Do **not** publish "negligible."

## 5. Caveats

- The disattenuation range is wide because recall is only pinned to a range; the human
  reference (r_true = 0.31) is the least reliable (single non-expert annotator) and the
  high-recall references (r_true ≈ 0.09) are the most credible per `RECALL_ROBUSTNESS.md`.
- The generative model assumes a simple first-order parent→child mechanism on the real forest
  structure; higher-order or thread-level mechanisms are not modelled.
- Specificity is very high (≈0.99) throughout, so attenuation is driven almost entirely by
  recall; this is why the result hinges on the recall estimate.
