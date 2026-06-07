"""Rehydrate the corpus from the public dehydrated file (re-fetch comment text by id).

Reads ``data/public/corpus_dehydrated.csv`` (id, parent_id, subreddit, sonnet_label) and
re-fetches each comment's text and author from Reddit's public JSON API
(``/api/info.json``, 100 ids per call), then writes the PII-bearing files the reproduction
consumes:

  data/derived/toxic_comments_sonnet.jsonl
  data/derived/non_toxic_comments_sonnet.jsonl
  data/raw/reddit/<subreddit>_dataset.jsonl

After running this, ``make reproduce`` works on your own machine without the authors shipping
any text or usernames.

Limitation (by design): comments deleted since collection cannot be re-fetched. Their structure
and label are preserved (so the reply forest stays intact), but their text is "[unavailable]"
and author "[deleted]". Because the reply graph keys on author, results that depend on the graph
(Tables II-III) can drift slightly from the published numbers in proportion to how much of the
corpus has since been deleted; the criterion (Table I) depends mainly on labels + structure and
is more robust. This deletion-respecting behaviour is the privacy feature of dehydration.

Requires ``requests`` (``uv pip install -e ".[labelling]"``). Be polite to the API: a delay
between calls is enforced; the full corpus is ~350 calls.

Run: ``python -m toxicity_criterion.labelling.rehydrate``  (optional: ``--delay 2``)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import requests

from .. import config

DEHYDRATED = os.path.join(config.PUBLIC_DIR, "corpus_dehydrated.csv")
INFO_URL = "https://www.reddit.com/api/info.json"
USER_AGENT = "toxicity-criterion-rehydrator/1.0 (academic reproduction; contact: see paper)"
BATCH = 100


def _ctype(parent_id: str) -> str:
    if parent_id.startswith("t3"):
        return "comment"
    if parent_id.startswith("t1"):
        return "reply"
    return "unknown"


def _load_dehydrated(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fetch_batch(ids: list[str], session: requests.Session) -> dict[str, dict]:
    """Fetch up to 100 comments by bare id; return {id: {body, author}} for those still present."""
    fullnames = ",".join(f"t1_{i}" for i in ids)
    resp = session.get(INFO_URL, params={"id": fullnames}, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    out: dict[str, dict] = {}
    for child in resp.json().get("data", {}).get("children", []):
        d = child.get("data", {})
        if d.get("id"):
            out[d["id"]] = {"body": d.get("body", ""), "author": d.get("author", "[deleted]")}
    return out


def run(dehydrated_path: str = DEHYDRATED, delay: float = 2.0) -> dict:
    """Rehydrate and write the derived + raw files. Returns counts of fetched vs missing."""
    rows = _load_dehydrated(dehydrated_path)
    fetched: dict[str, dict] = {}
    session = requests.Session()
    ids = [r["id"] for r in rows]
    for start in range(0, len(ids), BATCH):
        batch = ids[start : start + BATCH]
        try:
            fetched.update(_fetch_batch(batch, session))
        except requests.RequestException as e:
            print(f"  [warn] batch {start // BATCH} failed ({e}); its comments will be marked unavailable")
        print(f"  rehydrated {min(start + BATCH, len(ids))}/{len(ids)}", end="\r")
        time.sleep(delay)
    print()

    os.makedirs(config.DERIVED_DIR, exist_ok=True)
    os.makedirs(config.RAW_PRIVATE_DIR, exist_ok=True)
    tox_f = open(config.SONNET_TOXIC_PRIVATE, "w", encoding="utf-8")
    non_f = open(config.SONNET_NONTOXIC_PRIVATE, "w", encoding="utf-8")
    raw_files: dict[str, object] = {}
    n_missing = 0
    try:
        for r in rows:
            cid, pid, sub, label = r["id"], r["parent_id"], r["subreddit"], r["sonnet_label"]
            got = fetched.get(cid)
            if got is None:
                n_missing += 1
            text = got["body"] if got else "[unavailable]"
            author = got["author"] if got else "[deleted]"
            rec = {"id": cid, "text": text, "author": author, "type": _ctype(pid), "parent_id": pid}
            # raw per-subreddit (no label)
            if sub:
                if sub not in raw_files:
                    raw_files[sub] = open(
                        os.path.join(config.RAW_PRIVATE_DIR, f"{sub}_dataset.jsonl"), "w", encoding="utf-8"
                    )
                raw_files[sub].write(json.dumps(rec, ensure_ascii=False) + "\n")
            # derived sonnet (with label)
            rec_lab = {**rec, "prediction": label}
            (tox_f if label == "toxic" else non_f).write(json.dumps(rec_lab, ensure_ascii=False) + "\n")
    finally:
        tox_f.close()
        non_f.close()
        for fh in raw_files.values():
            fh.close()

    return {"total": len(rows), "fetched": len(rows) - n_missing, "missing_deleted": n_missing}


def main() -> None:
    ap = argparse.ArgumentParser(description="Rehydrate the corpus from the dehydrated id+label file.")
    ap.add_argument("--delay", type=float, default=2.0, help="seconds between API batches (be polite)")
    ap.add_argument("--input", default=DEHYDRATED, help="path to corpus_dehydrated.csv")
    args = ap.parse_args()
    stats = run(args.input, delay=args.delay)
    print(
        f"done: {stats['fetched']}/{stats['total']} rehydrated, "
        f"{stats['missing_deleted']} deleted/unavailable since collection."
    )
    print("wrote data/derived/*_sonnet.jsonl and data/raw/reddit/*_dataset.jsonl")
    if stats["missing_deleted"]:
        print(
            "note: deleted comments keep their structure + label but have no text/author; "
            "graph-based results (Tables II-III) may drift slightly. See module docstring."
        )


if __name__ == "__main__":
    main()
