"""
Build the full-corpus aggregated dataset for classification.

Produces corpus_dataset_full.csv: one row per ELIGIBLE comment in the entire Reddit
corpus (all 5 subreddits), with the SAME columns and built by the SAME functions as
the 200-sample annotation_dataset_full.csv -- so the context the classifier sees on the
35k is byte-for-byte consistent with what it saw on the validated 200
(validation == deployment).

Reuses build_validation_app.py's helpers (no duplicated logic): load, load_lexicon,
corpus_doc_freq, lexicon_hits, ancestor_chain, render_chain. The only differences vs
the 200 builder are (a) no sampling -- every eligible comment is included, and (b) the
id column is the real Reddit comment id (not an S### sample id).

Run:  python3 build_corpus_dataset.py
Out:  corpus_dataset_full.csv  (gitignored; contains Reddit text; regenerable)
"""

import csv
import html

from . import build_validation_app as B

HERE = B.HERE
OUT = HERE / "corpus_dataset_full.csv"


def eligible(r):
    t = (r.get("text") or "").strip()
    return t.lower() not in B.SKIP and len(t) >= 3


def main():
    # Full raw corpus: every comment id -> record (for ancestor resolution) and ->
    # subreddit (from the per-subreddit filename). Text HTML-unescaped, exactly as the
    # 200 builder does.
    byid, id2sub, records = {}, {}, []
    for f in sorted(B.RAW_DIR.glob("*_dataset.jsonl")):
        sub = f.stem.replace("_dataset", "")
        for r in B.load(f):
            r["text"] = html.unescape(r.get("text") or "")
            byid[r["id"]] = r
            id2sub[r["id"]] = sub
            records.append((r, sub))

    # Display lexicon: built IDENTICALLY to the 200 (doc-frequency over the classified
    # set, drop corpus-common words), so the lexicon hits shown are the same construction.
    tox, non = B.load(B.TOXIC), B.load(B.NONTOXIC)
    for r in tox + non:
        r["text"] = html.unescape(r.get("text") or "")
    id2text = {r.get("id"): r.get("text", "") for r in tox + non}
    single, multi = B.load_lexicon()
    df, ndoc = B.corpus_doc_freq(id2text)
    single = {w: c for w, c in single.items() if df.get(w, 0) / ndoc <= B.COMMON}
    print(f"display lexicon: {len(single)} single-word + {len(multi)} multi-word")

    n_total = len(records)
    n_written = 0
    with open(OUT, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, quoting=csv.QUOTE_ALL)
        w.writerow(["id", "subreddit", "reply_chain", "lexicon_terms", "comment_text", "root", "omitted"])
        for r, sub in records:
            if not eligible(r):
                continue
            text = (r.get("text") or "").strip()
            parents, omitted, root = B.ancestor_chain(r, byid)
            w.writerow(
                [
                    r["id"],
                    sub,
                    B.render_chain(parents, omitted, root),
                    ", ".join(B.lexicon_hits(text, single, multi)),
                    text,
                    root,
                    omitted,
                ]
            )
            n_written += 1

    print(f"wrote {OUT}: {n_written} eligible comments (of {n_total} raw).")


if __name__ == "__main__":
    main()
