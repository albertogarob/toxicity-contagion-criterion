"""Build the public, PII-free release from the local private data (authors only).

Produces everything the repository shares so the analyses reproduce **offline** (no Reddit
calls) with **no comment text and no real usernames**. Outputs go under ``data/public/``
(committed).

Outputs:
  data/public/corpus/toxic_comments.jsonl        id, author(pseudonym), type, parent_id, prediction
  data/public/corpus/non_toxic_comments.jsonl    (same; no text)
  data/public/corpus/raw/<subreddit>_dataset.jsonl   id, author(pseudonym), type, parent_id (no text)
  data/public/corpus_dehydrated.csv              id, parent_id, subreddit, sonnet_label
  data/public/validation/validation_key.csv      sample_id, stratum, subreddit, gpt_label, sonnet_label
                                                 (drops orig_id + author , the re-identification fields)
  data/public/validation/human_labels.csv        sample_id, human_label (copied; already PII-free)

Anonymisation: a deterministic bijection maps each real username to a stable pseudonym
``u00001`` ... applied consistently across the corpus and the per-subreddit files, so the reply
graph is isomorphic to the real one and every reported number reproduces identically. Sentinel
authors ([deleted], [removed], unknown, empty) are left unchanged (they are already non-identifying
and are treated as single nodes by the analysis, as in the original runs).

Inputs (private; local only): data/raw/reddit/*_dataset.jsonl, data/derived/*_sonnet.jsonl,
data/derived/validation/{validation_key.csv,human_labels.csv}.

Run: python -m toxicity_criterion.labelling.build_public_data
"""

from __future__ import annotations

import csv
import glob
import json
import os

from .. import config

RAW_PRIVATE = config.RAW_PRIVATE_DIR  # data/raw/reddit (real)
DERIVED = config.DERIVED_DIR  # data/derived
VAL_PRIVATE = config.VAL_PRIVATE_SRC  # data/derived/validation (real key)
SONNET_TOXIC_SRC = config.SONNET_TOXIC_PRIVATE
SONNET_NONTOXIC_SRC = config.SONNET_NONTOXIC_PRIVATE

PUB = config.PUBLIC_DIR  # data/public
PUB_CORPUS = os.path.join(PUB, "corpus")
PUB_RAW = os.path.join(PUB_CORPUS, "raw")
PUB_VAL = os.path.join(PUB, "validation")

SENTINELS = {"[deleted]", "[removed]", "[deleted by user]", "unknown", ""}


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_author_map():
    """Deterministic real-username -> pseudonym bijection over all raw records (sorted by id)."""
    authors = []
    seen = set()
    records = []
    for path in sorted(glob.glob(os.path.join(RAW_PRIVATE, "*_dataset.jsonl"))):
        records.extend(_read_jsonl(path))
    for rec in sorted(records, key=lambda r: r.get("id", "")):
        a = rec.get("author")
        if a is None or a in SENTINELS:
            continue
        if a not in seen:
            seen.add(a)
            authors.append(a)
    return {a: f"u{i + 1:05d}" for i, a in enumerate(authors)}


def _anon(author, amap):
    if author is None or author in SENTINELS:
        return author if author is not None else "unknown"
    return amap.get(author, "unknown")


def run() -> dict:
    amap = _build_author_map()
    os.makedirs(PUB_CORPUS, exist_ok=True)
    os.makedirs(PUB_RAW, exist_ok=True)
    os.makedirs(PUB_VAL, exist_ok=True)
    counts = {"authors_anonymized": len(amap)}

    # --- anonymized Sonnet label files (no text) ---
    for src, dst in (
        (SONNET_TOXIC_SRC, os.path.join(PUB_CORPUS, "toxic_comments.jsonl")),
        (SONNET_NONTOXIC_SRC, os.path.join(PUB_CORPUS, "non_toxic_comments.jsonl")),
    ):
        n = 0
        with open(dst, "w", encoding="utf-8") as out:
            for rec in _read_jsonl(src):
                out.write(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "author": _anon(rec.get("author"), amap),
                            "type": rec.get("type", ""),
                            "parent_id": rec.get("parent_id", ""),
                            "prediction": rec.get("prediction", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
        counts[os.path.basename(dst)] = n

    # --- anonymized raw per-subreddit (no text; for the per-subreddit breakdown) ---
    for path in sorted(glob.glob(os.path.join(RAW_PRIVATE, "*_dataset.jsonl"))):
        sub = os.path.basename(path)
        with open(os.path.join(PUB_RAW, sub), "w", encoding="utf-8") as out:
            for rec in _read_jsonl(path):
                out.write(
                    json.dumps(
                        {
                            "id": rec["id"],
                            "author": _anon(rec.get("author"), amap),
                            "type": rec.get("type", ""),
                            "parent_id": rec.get("parent_id", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    # --- dehydrated CSV (id, parent_id, subreddit, label) ---
    sub_of = {}
    for path in sorted(glob.glob(os.path.join(RAW_PRIVATE, "*_dataset.jsonl"))):
        sub = os.path.basename(path).replace("_dataset.jsonl", "")
        for rec in _read_jsonl(path):
            if rec.get("id"):
                sub_of[rec["id"]] = sub
    rows = []
    for src, label in ((SONNET_TOXIC_SRC, "toxic"), (SONNET_NONTOXIC_SRC, "non-toxic")):
        for rec in _read_jsonl(src):
            rows.append(
                {
                    "id": rec["id"],
                    "parent_id": rec.get("parent_id", ""),
                    "subreddit": sub_of.get(rec["id"], ""),
                    "sonnet_label": rec.get("prediction", label),
                }
            )
    rows.sort(key=lambda r: r["id"])
    with open(os.path.join(PUB, "corpus_dehydrated.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "parent_id", "subreddit", "sonnet_label"])
        w.writeheader()
        w.writerows(rows)
    counts["dehydrated_rows"] = len(rows)

    # --- anonymized validation key (drop orig_id + author) + human labels ---
    key_src = os.path.join(VAL_PRIVATE, "validation_key.csv")
    keep_cols = ["sample_id", "stratum", "subreddit", "gpt_label", "sonnet_label"]
    with (
        open(key_src, encoding="utf-8") as f,
        open(os.path.join(PUB_VAL, "validation_key.csv"), "w", newline="", encoding="utf-8") as out,
    ):
        r = csv.DictReader(f)
        w = csv.DictWriter(out, fieldnames=keep_cols)
        w.writeheader()
        for row in r:
            w.writerow({c: row.get(c, "") for c in keep_cols})
    hl = os.path.join(VAL_PRIVATE, "human_labels.csv")
    if os.path.exists(hl):
        with open(hl, encoding="utf-8") as f:
            data = f.read()
        with open(os.path.join(PUB_VAL, "human_labels.csv"), "w", encoding="utf-8") as out:
            out.write(data)

    return counts


def main() -> None:
    c = run()
    print("Built public PII-free release under data/public/:")
    for k, v in c.items():
        print(f"  {k}: {v}")
    print("  (no comment text, no real usernames; reproduces offline)")


if __name__ == "__main__":
    main()
