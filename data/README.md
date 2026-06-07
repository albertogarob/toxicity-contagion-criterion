# Data

The repository ships **anonymized, text-free** data sufficient to reproduce every result
**offline**. No comment text and no real usernames are published. See `../DATA_RELEASE.md` for
the full policy and ethics statement.

## Committed (public, PII-free)

```
data/public/
├── corpus/
│   ├── toxic_comments.jsonl        id, author (pseudonym), type, parent_id, prediction   (no text)
│   ├── non_toxic_comments.jsonl    (same)
│   └── raw/<subreddit>_dataset.jsonl   id, author (pseudonym), type, parent_id           (no text)
├── corpus_dehydrated.csv           id, parent_id, subreddit, sonnet_label
└── validation/
    ├── panel_{opus,gpt,gemini}.csv  sample_id + label
    ├── validation_populations.json  strata sizes + seed
    ├── human_labels.csv             sample_id, human_label
    └── validation_key.csv           sample_id, stratum, subreddit, gpt_label, sonnet_label  (anonymized)
```

Usernames are replaced by stable pseudonyms via a deterministic bijection, so the reply graph is
isomorphic to the real one and every reported number reproduces **identically**. The
username↔pseudonym map and the real `orig_id`/`author` are **not** published. Regenerate the
public release from private sources with `python -m toxicity_criterion.labelling.build_public_data`.

## Withheld (private; gitignored; never committed)

```
data/raw/reddit/*_dataset.jsonl          raw comment text + real usernames
data/derived/*_sonnet.jsonl              Sonnet labels with real text + usernames
data/derived/validation/validation_key.csv   real re-identification key (orig_id, author)
```

## Recovering the real text (optional)

Reproduction does **not** need it, but to work with the actual comments:

```bash
uv pip install -e ".[labelling]"   # rehydration needs `requests`
make rehydrate                     # re-fetch text by ID from Reddit's public .json endpoints -> data/derived/ + data/raw/
```

Rehydration respects deletions (removed comments cannot be recovered).
