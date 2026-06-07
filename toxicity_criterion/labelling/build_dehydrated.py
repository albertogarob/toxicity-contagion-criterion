"""Build the public *dehydrated* corpus: comment IDs + structure + labels, no text/usernames.

Reddit's API terms prohibit redistributing comment text, and the usernames are personal data.
We therefore publish only what is needed to *rehydrate* the corpus, the comment id, its parent
id, its subreddit, and our Sonnet toxic/non-toxic label, and a rehydration script
(:mod:`toxicity_criterion.labelling.rehydrate`) that re-fetches the text by id from the public
Reddit API. This honours deletions (removed comments do not come back) and ships no PII.

Inputs (PII; local only): the raw per-subreddit files (for the subreddit of each id) and the
Sonnet label files. Output (PII-free; committed): ``data/public/corpus_dehydrated.csv``.

Run: ``python -m toxicity_criterion.labelling.build_dehydrated``
"""

from __future__ import annotations

import csv
import glob
import json
import os

from .. import config

OUT = os.path.join(config.PUBLIC_DIR, "corpus_dehydrated.csv")


def _subreddit_by_id() -> dict[str, str]:
    """Map each comment id -> its subreddit, read from the raw per-subreddit files."""
    sub_of: dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(config.RAW_DIR, "*_dataset.jsonl"))):
        sub = os.path.basename(path).replace("_dataset.jsonl", "")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("id"):
                        sub_of[rec["id"]] = sub
    return sub_of


def run(out_path: str = OUT) -> int:
    """Write the dehydrated CSV; return the number of rows."""
    sub_of = _subreddit_by_id()
    rows = []
    for path, label in ((config.SONNET_TOXIC, "toxic"), (config.SONNET_NONTOXIC, "non-toxic")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cid = rec.get("id")
                if not cid:
                    continue
                rows.append(
                    {
                        "id": cid,
                        "parent_id": rec.get("parent_id", ""),
                        "subreddit": sub_of.get(cid, ""),
                        "sonnet_label": rec.get("prediction", label),
                    }
                )
    rows.sort(key=lambda r: r["id"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=["id", "parent_id", "subreddit", "sonnet_label"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main() -> None:
    n = run()
    print(f"wrote {OUT} ({n} rows; id, parent_id, subreddit, sonnet_label , no text, no author)")


if __name__ == "__main__":
    main()
