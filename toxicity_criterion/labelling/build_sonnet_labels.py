"""
Produce Sonnet-relabeled versions of the prediction jsonl files used by the rest
of the pipeline. Joins the FULL raw corpus (5 per-subreddit files = 35,279
comments) with the Sonnet labels (in_domain_validation/corpus_labels.csv) and
writes:

  toxic_comments_sonnet.jsonl       (records where sonnet=='toxic')
  non_toxic_comments_sonnet.jsonl   (records where sonnet=='non-toxic')

Same record schema as the original GPT-OSS files (id, text, author, type,
parent_id, prediction). Comments missing a Sonnet label (the ~66 persistent-
retry-failure rows) are skipped. The originals are NOT modified.
"""

import csv
import html
import json
from pathlib import Path

from . import lab_paths as P

RAW_DIR = P.RAW
LABELS = P.VAL / "corpus_labels.csv"
OUT_TOX = P.SONNET_TOXIC
OUT_NON = P.SONNET_NONTOXIC


def load_jsonl(path):
    rows, dec = [], json.JSONDecoder()
    data = Path(path).read_text(encoding="utf-8")
    i, n = 0, len(data)
    while i < n:
        while i < n and data[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        try:
            r, i = dec.raw_decode(data, i)
        except json.JSONDecodeError:
            break
        rows.append(r)
    return rows


def main():
    labels = {}
    with open(LABELS, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if r and r[1] in ("toxic", "non-toxic"):
                labels[r[0]] = r[1]
    print(f"Sonnet labels: {len(labels)} (toxic={sum(1 for v in labels.values() if v=='toxic')})")

    tox_records, non_records = [], []
    seen = set()
    raw_total = 0
    for f in sorted(RAW_DIR.glob("*_dataset.jsonl")):
        for r in load_jsonl(f):
            raw_total += 1
            cid = r.get("id")
            if cid in seen or cid not in labels:
                continue
            seen.add(cid)
            rec = {
                "id": cid,
                "text": html.unescape(r.get("text") or ""),
                "author": r.get("author"),
                "type": r.get("type"),
                "parent_id": r.get("parent_id"),
                "prediction": labels[cid],
            }
            (tox_records if labels[cid] == "toxic" else non_records).append(rec)
    print(f"raw corpus: {raw_total}; sonnet-labeled records assembled: {len(tox_records)+len(non_records)}")
    print(f"  toxic: {len(tox_records)}")
    print(f"  non-toxic: {len(non_records)}")

    for path, recs in [(OUT_TOX, tox_records), (OUT_NON, non_records)]:
        with open(path, "w", encoding="utf-8") as fo:
            for r in recs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(recs)} records")


if __name__ == "__main__":
    main()
