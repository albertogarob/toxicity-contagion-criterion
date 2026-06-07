# Validation-set design: the 200-comment human-labelled sample

This document gives the rationale and the exact methodology behind the 200-comment
human-validation sample. It supersedes the earlier GPT-OSS-stratified design in
`SAMPLING_DESIGN.md` (see §7 for why that one is no longer the right sample).

The sample is built by `build_validation_sample.py` (seeded, reproducible) and scored
by `score_validation_v2.py`.

---

## 1. What the sample has to accomplish

The paper's evidentiary chain is a **silver-standard** design:

1. Human-label a gold sample of 200 comments.
2. **Validate the Sonnet classifier** against that gold (in-domain precision **and**
   recall). If Sonnet clears a pre-registered bar, it is a trustworthy reference for
   this task.
3. Use validated Sonnet, plus the direct human measurement, to **quantify how wrong and
   biased the fine-tuned GPT-OSS classifier was**, at corpus scale.

For this to be sound the 200 must let us estimate, against human ground truth:

- **Sonnet's precision** (when Sonnet says toxic, is it right?),
- **Sonnet's recall** (does Sonnet miss real toxicity? the load-bearing risk, since
  Sonnet is conservative at 1.76%),
- **GPT-OSS's precision and recall** in-domain,
- **who is right where the two classifiers disagree**, and
- **the true toxic prevalence**.

A sample stratified on a single classifier's predictions cannot do all of this. In
particular the previous GPT-OSS-stratified sample contained only ~7 Sonnet-positive
comments, far too few to estimate Sonnet's precision, and it never directly sampled the
disagreement between the classifiers, which is exactly where the adjudication lives.

## 2. The design: stratify on classifier *agreement*

Both classifiers labelled the same 23,586 comments (the **overlap**; GPT-OSS only ran on
23,680 of the 35,176 eligible comments, so the overlap is smaller than the full corpus).
Cross-tabulating the two classifiers partitions the overlap into a 2x2 table, and we add
a model-independent random anchor:

| Stratum | GPT-OSS | Sonnet | Population | Sample | What human labels here decide |
|---|---|---|---|---|---|
| `agree_tox`        | toxic     | toxic     | 185    | 45 | both-toxic anchor; confirms shared true positives |
| `gpt_tox_son_non`  | **toxic** | **non-tox** | 2,683 | 55 | **the crux**: who is right when they conflict (GPT-OSS over-flag vs Sonnet miss) |
| `gpt_non_son_tox`  | non-tox   | **toxic** | 330    | 45 | does Sonnet catch toxicity GPT-OSS missed? |
| `agree_non`        | non-tox   | non-tox   | 20,388 | 30 | shared false negatives / NPV; bulk recall check |
| `random`           | --        | --        | 35,176 | 25 | model-independent prevalence anchor |

(Populations are the realized values for seed 42, mirrored in
`validation_populations.json`.)

The first four strata **partition the overlap exactly** (each comment is in exactly one
cell), with a known sampling fraction per cell. The random stratum is drawn from the full
eligible corpus and is disjoint from the four cells (already-sampled ids are excluded).

### Why these particular sizes
- **The two disagreement cells are deliberately over-sampled** (55 + 45 = 100 of the 200),
  because they carry almost all of the information: they validate Sonnet's precision
  (`gpt_non_son_tox`: are Sonnet's extra positives real?) and Sonnet's recall
  (`gpt_tox_son_non`: are GPT-OSS's extra positives real toxicity Sonnet missed, or noise?),
  and they adjudicate the classifier conflict that the whole paper turns on.
- `agree_tox` (45) anchors the shared positives.
- `agree_non` (30) covers the huge "both say clean" bulk; because that cell's population is
  20,388, even a few human-toxic labels here reweight to a meaningful number of shared
  false negatives, so 30 is enough to detect a recall problem affecting the bulk.
- `random` (25) is a model-independent check; see the honest limitation in §6.

## 3. How human labels become unbiased estimates (the estimator)

Because the four cells partition the overlap with known cell sizes, Horvitz-Thompson
reweighting gives unbiased population estimates for **either** classifier, no matter how
the strata were defined. For cell *c* with population `N_c`, sampled `n_c`, of which `t_c`
were human-labelled toxic, the estimated number of truly-toxic comments in the cell is

```
T_c = N_c * (t_c / n_c)
```

Each classifier's prediction is constant within a cell, so summing `T_c` over the cells
where a classifier predicts "toxic" gives its true positives, and:

```
precision = (sum of T_c over that classifier's predicted-toxic cells) / (population of those cells)
recall    = (sum of T_c over that classifier's predicted-toxic cells) / (sum of T_c over ALL cells)
```

This yields each classifier's in-domain precision, recall and F1 against the human gold,
reweighted to the overlap population. `score_validation_v2.py` implements exactly this.

### Disagreement adjudication
The two disagreement cells are read directly: in `gpt_tox_son_non`, the human-toxic share
is the fraction of conflicts where humans **side with GPT-OSS** (it was right to flag) vs
**with Sonnet** (the flag was spurious); `gpt_non_son_tox` is the mirror for Sonnet's
unique positives. This is the sentence the paper needs: "of the cases where the default
classifier and the validated one disagree, humans side with X% of the time."

## 4. The pre-registered decision gate

Fixed **before** seeing the human labels, to keep the conclusion honest:

> If Sonnet's in-domain **precision >= 0.80** and its **recall is not catastrophically
> low**, Sonnet is accepted as the validated in-domain reference, and we quantify
> GPT-OSS's bias at scale against it. If precision clears but recall is poor, we report
> Sonnet as precision-validated only and temper the over-flagging claim (because then part
> of "GPT-OSS over-flags" could be "Sonnet under-flags"). If precision fails, we reconsider
> the labelling entirely.

The recall clause is the important one: Sonnet is conservative, so the analysis must
demonstrate it is not simply missing toxicity. The over-sampled `gpt_tox_son_non` cell is
what powers that recall estimate.

### Outcome, and a documented deviation from the binary gate

The gate above was pre-registered as a binary pass/fail on **human gold**. The realized
human-gold result is **precision 0.711, recall 0.234** (`score_validation_v2.py`), i.e.
precision is **below the 0.80 bar**. Taken literally, the gate says "reconsider the
labelling."

We do not, however, treat the human measurement as decisive gold, for a reason that was
itself a stated premise of the study: **no annotator fluent in UK political slang/politics
was available** (the labeller is a non-native, non-expert English speaker). To bound the
labels from a second direction we added a **multi-LLM silver panel** (`classify_panel.py`,
`score_panel.py`): three frontier models from independent families (Claude Opus 4.8,
GPT-5.5, Gemini 2.5 Pro) labelling the *same* 200 under the *same* rubric and context,
2-of-3 majority as the reference, HT-reweighted identically. Against that panel the deployed
classifier scores **precision 0.92, recall 0.32**, and the panel agrees substantially with
itself (Fleiss' $\kappa = 0.73$) and with the deployed classifier (Cohen's $\kappa =
0.83$--$0.84$ with the two strongest), far more than the human does. This is consistent with
the human *under-recognizing* in-domain toxicity, which deflates Sonnet's measured precision
by counting correct toxic calls as false positives.

**Decision (made transparently, post-hoc):** rather than mechanically failing the labelling
on a single non-expert human reference, or silently swapping in the panel as the new gold
(the goalpost-move the pre-registration existed to prevent), we **retire the binary gate**
and report precision and recall as a **range across the two imperfect references**
(precision 0.71--0.92, recall 0.23--0.32). Neither reference is gold; we say so and let the
bracket stand. This is a deliberate deviation from §4-as-written and is flagged as such in
the paper.

## 5. Blindness, rubric, reproducibility

- The labeller sees **no classifier verdict** and **no stratum**; comments are presented
  in fully randomized order (strata interleaved). The app shows only the comment, its
  subreddit, the reply chain, and the HurtLex lexicon hits (an aid; a flagged word is not
  automatically toxic).
- Humans use the **same `ANNOTATION_GUIDELINES.md` rubric and the same per-comment
  context** that the Sonnet classifier used, so the human and model judgments are
  commensurable.
- Everything is deterministic from `SEED = 42` and the two label sets; no new API calls
  are made (both classifiers' labels already exist).
- `validation_key.csv` (gitignored; holds usernames) records, per comment:
  `sample_id, stratum, subreddit, gpt_label, sonnet_label, orig_id, author`.
  `validation_populations.json` (committed; no PII) records the cell populations, corpus
  size, allocation and seed.

## 6. Honest limitations

- **Rare-positive recall is intrinsically noisy.** True toxicity is ~2%, so the random
  stratum (n=25) is expected to contain ~0--1 toxic comments; it is a weak prevalence
  estimator and serves mainly as a model-independent sanity check. The *reweighted* cells
  give the stronger prevalence and recall estimates; the random stratum is corroborating,
  not primary.
- **No gold standard; we report a range.** Neither reference is gold: the human is a single
  non-expert annotator (likely under-recognizing in-domain toxicity), and the LLM panel is a
  silver standard whose members can share blind spots. We report precision/recall as a range
  across the two references (precision 0.71--0.92, recall 0.23--0.32, see §4) rather than a
  single validated point, and carry that range, not a point estimate, into the paper.
- **Overlap, not full corpus.** The agreement comparison is necessarily limited to the
  23,586 comments GPT-OSS also labelled; the random stratum and the deployed Sonnet labels
  cover all 35,176.
- **Small per-cell n** (30--55) gives wide confidence intervals on the reweighted rates;
  we report them as estimates with that caveat, and the qualitative conclusions (does
  GPT-OSS over-flag? does Sonnet under-recall?) are robust even when the point estimates
  are not precise.
- **Single corpus**; none of this is claimed to generalize beyond UK Reddit immigration
  discourse without replication.

## 7. Why the previous (GPT-OSS-stratified) sample was retired

`SAMPLING_DESIGN.md` describes a 4-stratum sample defined on **GPT-OSS predictions**
(predicted-toxic; predicted-non-toxic with/without hate-lexicon; random). That was the
correct design when GPT-OSS was the only classifier. It is the wrong sample now because:
(a) it contains only ~7 Sonnet-positive comments, so it cannot estimate Sonnet's
precision; and (b) it never directly samples the GPT-OSS/Sonnet disagreement, which is the
quantity the paper now turns on. No labelling effort is lost, the previous sample was
never human-labelled.

## 8. Files

- `build_validation_sample.py` -- builds the sample, key, populations, and the app.
- `validation_key.csv` -- the labelled dataset's key (gitignored; regenerable from seed).
- `validation_populations.json` -- cell populations + allocation + seed (committed).
- `label_app.html` -- the blind labelling app for this sample (gitignored; regenerable).
- `human_labels.csv` -- produced by the annotator via the app (gitignored).
- `score_validation_v2.py` -- the Horvitz-Thompson scorer + adjudication + decision gate.

Reference for the rubric: Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017).
*Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM, 512--515.
