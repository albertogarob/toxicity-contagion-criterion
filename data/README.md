# Data

## Status: withheld pending institutional approval

**Nothing under `data/` is shared in this repository except this README.** No comment text, no
usernames, no comment IDs, and not even a dehydrated id+label file.

Three coauthors are based at a university in the Netherlands, so any release of data derived from
the Reddit corpus is subject to the GDPR / Dutch UAVG and must first be approved by the
university privacy officer (FG/DPO) and research-ethics committee (with a DPIA / Data Management
Plan as required). A dehydrated id+label file is re-identifiable on rehydration and therefore
still personal data, so it is gated too. See `../DATA_RELEASE.md`.

## Local layout (gitignored; authors only)

When the corpus is present on a machine, it lives here:

```
data/
├── raw/reddit/        raw scraped corpus (text + usernames)
├── derived/           Sonnet label files + validation intermediates
│   ├── toxic_comments_sonnet.jsonl
│   ├── non_toxic_comments_sonnet.jsonl
│   └── validation/    validation_key.csv, human_labels.csv
└── public/            the *would-be* public artifacts (still gitignored for now)
    ├── corpus_dehydrated.csv          id, parent_id, subreddit, sonnet_label
    └── validation/    panel_{opus,gpt,gemini}.csv, validation_populations.json
```

All of these are excluded by `.gitignore`.

## Planned release on approval (order of increasing exposure)

1. `public/validation/panel_*.csv` + `validation_populations.json` , PII-free (sample_id + labels).
2. `public/corpus_dehydrated.csv` , dehydrated corpus; reconstruct text locally with
   `make rehydrate` (re-fetches by ID from Reddit's public .json endpoints; respects deletions).
3. Full hydrated set (text + usernames) , by gated/dedicated access only; never committed.

The dehydration / rehydration code (`../toxicity_criterion/labelling/build_dehydrated.py`,
`rehydrate.py`) is already in place; enabling a release is a matter of un-ignoring the relevant
paths once approval is granted.
