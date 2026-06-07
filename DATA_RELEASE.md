# Data availability statement

This document is the repository's answer to **ML Reproducibility Checklist v2.0, item 9**
(downloadable dataset / simulation environment).

## Status: available

The repository ships the anonymized and dehydrated data and all derivative artifacts needed to
**reproduce every result offline (no Reddit calls)**, while publishing **no comment text and no
real usernames**. We do not redistribute the raw comment **text** or **real usernames** (Reddit's
terms + user privacy); only anonymized, text-free derivatives and dehydrated IDs are shared.

The corpus was collected by querying Reddit's public per-subreddit `.json` web endpoints (not the
official Data API); the released artifacts contain neither text nor usernames.

## What IS shared (PII-free)

| Released | Path | Contents |
|---|---|---|
| **Anonymized corpus** | `data/public/corpus/{toxic,non_toxic}_comments.jsonl` | `id, author (pseudonym), type, parent_id, prediction` , **no text** |
| Anonymized per-subreddit files | `data/public/corpus/raw/<sub>_dataset.jsonl` | `id, author (pseudonym), type, parent_id` , **no text** |
| **Dehydrated corpus** | `data/public/corpus_dehydrated.csv` | `id, parent_id, subreddit, sonnet_label` |
| Silver-panel labels | `data/public/validation/panel_{opus,gpt,gemini}.csv` | `sample_id` + label |
| Validation populations | `data/public/validation/validation_populations.json` | strata sizes + seed |
| Human labels | `data/public/validation/human_labels.csv` | `sample_id, human_label` |
| **Anonymized** validation key | `data/public/validation/validation_key.csv` | `sample_id, stratum, subreddit, gpt_label, sonnet_label` (drops `orig_id` + `author`) |
| Code, aggregate results, figures | `toxicity_criterion/`, `results/*.json`, `figures/*.pdf` | , |

Reproduce with no Reddit access:

```bash
make setup
make reproduce     # Tables I-III + criterion, from the anonymized corpus
make figures
make test
```

### Anonymization (why results are identical)
A deterministic bijection maps each real username to a stable pseudonym (`u00001`...), applied
consistently across the corpus and per-subreddit files; sentinel authors (`[deleted]`, etc.) are
left unchanged. The reply graph is therefore **isomorphic** to the real one, so every reported
number reproduces **identically**. The username↔pseudonym map is **not** published. Comment text
is dropped entirely (no analysis uses it). Regenerate the public release from private sources with
`python -m toxicity_criterion.labelling.build_public_data`.

## What is WITHHELD (private; gitignored)

| Withheld | Why | How to obtain |
|---|---|---|
| Raw comment **text** (`data/raw/`) | not redistributed (Reddit terms + privacy) | `make rehydrate` re-fetches by ID from Reddit's public `.json` endpoints |
| Real **usernames** (`data/derived/*_sonnet.jsonl`) | personal data | recovered into your local copy by rehydration; never shipped by us |
| Real re-identification key (`orig_id`/`author` in the validation key) | links `sample_id` -> a person | not released |

## Optional: recover the real text (rehydration)
Not needed for reproduction (the anonymized corpus already reproduces everything). To work with
the actual comment text:

```bash
uv pip install -e ".[labelling]"
make rehydrate     # re-fetch text by ID -> data/derived/ + data/raw/ (real text + usernames, local only)
```

Rehydration **respects deletions** (removed comments are not recoverable), so it cannot resurrect
content a user has since deleted.

## Responsible use

The released data attaches a `toxic`/`non-toxic` label to (pseudonymous) public comments.

- **Public, comment-level data.** Labels describe individual public comments, not persons, and
  carry no demographic or identity attributes. No usernames are published.
- **Automated labels are imperfect.** The Sonnet labels are model predictions (measured
  precision ~ 0.92, recall reported as a range), not ground truth, and must not be treated as
  authoritative judgements about any individual. See `docs/ANNOTATION_GUIDELINES.md`.
- **Right to erasure.** Rehydration respects deletions; a removed comment cannot be recovered.
- **Takedown.** The authors will remove specific IDs from the released files on request: [contact].
- **Intended use.** Research on moderation methodology only; not for profiling, enforcement
  against individuals, or any use that re-identifies or targets users.

## Pre-registered determinism

All reported numbers use `SEED = 42` and `PYTHONHASHSEED = 0`; permutation null `B = 300`;
recall degradation 20 perturbations × 6 fractions; influence maximization mean over 5 runs
(`R = 200`). The same inputs reproduce the same outputs.
