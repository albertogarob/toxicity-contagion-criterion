# Data availability statement

This document is the repository's answer to **ML Reproducibility Checklist v2.0, item 9**
(downloadable dataset / simulation environment). It states what is shared, what is withheld, and
why.

## Status: data withheld pending institutional approval

At this time the repository shares **no corpus data , not even the dehydrated id+label file.**

Three coauthors are based at a university in the Netherlands, so the project is subject to the
EU **GDPR** / Dutch **UAVG** and the university's research-data governance. Sharing any data
derived from the Reddit corpus (including a dehydrated, text-free id+label file, which remains
*re-identifiable* on rehydration and therefore personal data under the GDPR) must first be
cleared by the university **privacy officer (FG/DPO)** and **research-ethics committee**, with a
DPIA and Data Management Plan as required. Until that approval is in place, only code, aggregate
results, and anonymized figures are published.

The discourse studied (immigration) can touch **special-category data** (Art. 9), which is the
main reason the release is gated rather than automatic.

## What IS shared now

| Shared | Where |
|---|---|
| All analysis / criterion / figure code | `toxicity_criterion/` |
| Scraping, labelling, dehydration + rehydration code | `toxicity_criterion/scraping/`, `toxicity_criterion/labelling/` |
| Aggregate result tables (no personal data) | `results/*.json` |
| Generated figures (anonymized reply structures; no text/usernames; same as in the paper) | `figures/*.pdf` |
| Generative simulation (positive control; code-only, no data) | `toxicity_criterion/criterion/disattenuation.py`, `toxicity_criterion/figures/positive_control.py` |
| Documentation | `docs/`, `data/README.md` |

No usernames, no comment text, and no comment IDs are published.

## What is WITHHELD now (and the planned release on approval)

Nothing under `data/` is committed except `data/README.md` (see `.gitignore`). The machinery to
release data is already in place; on ethics/DPO approval it can be published in order of
increasing exposure by un-ignoring the relevant paths:

| Withheld | Sensitivity | Release mechanism (on approval) |
|---|---|---|
| `data/public/validation/panel_*.csv`, `validation_populations.json` | PII-free (sample_id + labels) | commit directly |
| `data/public/corpus_dehydrated.csv` (id, parent_id, subreddit, label) | dehydrated; re-identifiable on rehydration | commit + ship `rehydrate.py` |
| `data/derived/*`, `data/raw/*` (text + usernames) | full PII | gated/dedicated access only; never committed |

## The prepared sharing mechanism (dehydration / rehydration)

When approved, the intended public form is a **dehydrated** dataset (comment IDs + reply
structure + toxic/non-toxic labels, no text, no usernames), reconstructed locally via the public
Reddit API:

```bash
# AUTHORS, once approved: build the dehydrated file (PII-free) and un-ignore it
python -m toxicity_criterion.labelling.build_dehydrated   # -> data/public/corpus_dehydrated.csv

# ANYONE, once the dehydrated file is published:
uv pip install -e ".[labelling]"
make rehydrate     # re-fetch text by ID -> data/derived/ + data/raw/
make reproduce
```

This approach minimises data, respects deletions (removed comments do not rehydrate), and ships
no text or usernames. The dehydrate -> rehydrate round-trip is validated to reconstruct the
corpus exactly when no comments have been deleted (619 toxic / 34,557 non-toxic recovered
identically). Comments deleted since collection cannot be re-fetched; their structure + label are
preserved but text/author are unavailable, which can cause slight drift in graph-based results
(Tables II–III); the criterion (Table I) is robust.

## Reviewers

To verify the headline results before any public data release, the authors can provide access to
the necessary files under the program committee's confidentiality terms and within the
institutional approval. The committed `results/*.json` give the reference values the analyses
produce.

## Correction to earlier project notes

The raw/derived files retained on the authors' machines contain **real usernames** (they are
*not* anonymized at collection). This is precisely why they are withheld.

## Ethics statement

Should the dehydrated dataset be released after approval, it would attach a `toxic`/`non-toxic`
label to re-fetchable comment IDs. We note:

- **Public, comment-level data.** Labels describe individual public comments, not persons, and
  carry no demographic or identity attributes. No usernames would be published.
- **Automated labels are imperfect.** The Sonnet labels are model predictions (measured
  precision ~ 0.92, recall reported as a range), not ground truth, and should not be treated as
  authoritative judgements about any individual. See `docs/ANNOTATION_GUIDELINES.md`.
- **Right to erasure.** Rehydration respects deletions; a removed comment cannot be recovered.
- **Takedown.** Authors will remove specific IDs from any released file on a reasonable request.
- **Intended use.** Research on moderation methodology only; not for profiling, enforcement
  against individuals, or any use that re-identifies or targets users.

## Pre-registered determinism

All reported numbers use `SEED = 42` and `PYTHONHASHSEED = 0`; permutation null `B = 300`;
recall degradation 20 perturbations × 6 fractions; influence maximization mean over 5 runs
(`R = 200`). With the corpus reconstructed, the same inputs reproduce the same outputs.
