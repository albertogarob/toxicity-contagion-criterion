"""
Claude toxicity classifier (the paper's classifier, replacing GPT-OSS).

Ingests the aggregated dataset annotation_dataset_full.csv (columns: sample_id,
subreddit, reply_chain, lexicon_terms, comment_text, root, omitted) and labels each
comment toxic / non-toxic using the SAME context a human annotator sees in
label_app.html: the subreddit, the reply chain, and the HurtLex terms found (an aid,
not a verdict). build_user() renders the reply_chain text verbatim, so the prompt
matches the CSV the human inspects (inspection == ingestion).

Design (per the Anthropic SDK skill):
  - model: claude-sonnet-4-6
  - structured output (json_schema) -> guaranteed {"label": "toxic"|"non-toxic"}
  - prompt caching on the shared system prompt (cache_control: ephemeral)
  - two execution paths sharing ONE config (CFG):
      interactive  -> one request per comment (use for the 200-sample validation)
      batch        -> Message Batches API, 50% cost (use to scale to ~35k later)

IMPORTANT: validate and deploy with the SAME CFG, or the 200-sample precision/
recall won't describe the classifier you actually run on the full corpus.

Usage:
  python3 classify_claude.py --preview            # render real prompts, NO API call
  python3 classify_claude.py                      # interactive (needs ANTHROPIC_API_KEY)
  python3 classify_claude.py --batch              # submit/poll a batch instead
  python3 classify_claude.py --input all.csv      # a different dataset (e.g. the 35k)
"""

import argparse
import csv
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

from . import lab_paths as P

HERE = P.VAL  # claude_labels.csv / prompt_previews.txt under data/derived/validation

DEFAULT_INPUT = P.VAL / "annotation_dataset_full.csv"
OUTPUT = P.VAL / "claude_labels.csv"
MODEL = "claude-sonnet-4-6"

# Shared, frozen instructions (cached). The classifier uses the SAME annotation
# guidelines file as the human annotator, so it judges against the identical target.
GUIDELINES = (P.DOCS / "ANNOTATION_GUIDELINES.md").read_text(encoding="utf-8")
SYSTEM = (
    GUIDELINES
    + "\n\n---\n"
    + "You are the automated annotator applying the guidelines above. Each item has four "
    "sections: SUBREDDIT (the community the comment was posted in), THREAD CONTEXT (the "
    "chain of parent comments, oldest first, with the immediate parent last; context "
    "only), HATE-LEXICON TERMS FOUND (an aid only, a flagged term is NOT automatically "
    "toxic), and COMMENT TO LABEL. Judge ONLY the COMMENT TO LABEL itself; use the "
    "subreddit and thread purely as context, never as the thing being labeled.\n\n"
    "Assign exactly one label, which must be either `toxic` or `non-toxic` (no other "
    "value is allowed). Output ONLY that single label, nothing else: no explanation, no "
    "reasoning, no punctuation, no extra text."
)

SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string", "enum": ["toxic", "non-toxic"]}},
    "required": ["label"],
    "additionalProperties": False,
}

# Single source of truth for the request config, shared by both paths.
CFG = {
    "model": MODEL,
    "max_tokens": 200,
    "thinking": {"type": "disabled"},  # structured output -> no leaked reasoning; cheap at scale
    "system": [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
    "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
}


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_user(row):
    """Render the user message from one dataset row. The reply_chain column is the
    exact text shown in the inspectable CSV, so the prompt == what the human sees."""
    sub = (row.get("subreddit") or "unknown").strip()
    chain = (row.get("reply_chain") or "").strip()
    lex = (row.get("lexicon_terms") or "").strip()
    comment = row.get("comment_text") or ""
    parts = [
        f"SUBREDDIT: r/{sub}",
        chain,  # already a self-headed "THREAD CONTEXT..." block (from render_chain)
        "HATE-LEXICON TERMS FOUND: " + (lex if lex else "none"),
        "COMMENT TO LABEL:\n" + comment,
    ]
    return "\n\n".join(parts)


def parse_label(message):
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        return json.loads(text).get("label", "")
    except json.JSONDecodeError:
        return ""


def classify_one(client, row, id_col, max_retries=7):
    """One comment -> (id, label or None). Retries 429/5xx/connection with backoff;
    returns label=None on exhaustion or a non-retryable error (the row stays unlabeled
    so a later idempotent re-run retries it)."""
    cid = row[id_col]
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(messages=[{"role": "user", "content": build_user(row)}], **CFG)
            return cid, parse_label(msg)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) and 500 <= e.status_code < 600:
                time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
                continue
            print(f"  [{cid}] non-retryable: {e}", flush=True)
            return cid, None
    print(f"  [{cid}] gave up after {max_retries} retries", flush=True)
    return cid, None


def run_streaming(rows, client, out_path, id_col, workers, limit=None):
    """Idempotent, streaming, concurrent. Skips ids already in out_path, classifies the
    rest with a thread pool, and appends each (id,label) the instant it returns (flush +
    fsync), so a crash loses at most in-flight items and a re-run resumes cleanly."""
    done = set()
    if out_path.exists():
        with open(out_path, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd, None)
            for r in rd:
                if r and r[0]:
                    done.add(r[0])
    todo = [r for r in rows if r[id_col] not in done]
    if limit:
        todo = todo[:limit]
    print(f"{len(done)} already labeled, {len(todo)} to classify, workers={workers}", flush=True)
    if not todo:
        print("nothing to do (already complete).", flush=True)
        return

    new_file = not out_path.exists()
    fout = open(out_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new_file:
        writer.writerow(["id", "claude_label"])
        fout.flush()
    lock = threading.Lock()
    stats = {"ok": 0, "toxic": 0, "err": 0}

    def work(row):
        cid, label = classify_one(client, row, id_col)
        with lock:
            if label:
                writer.writerow([cid, label])
                fout.flush()
                os.fsync(fout.fileno())
                stats["ok"] += 1
                if label == "toxic":
                    stats["toxic"] += 1
            else:
                stats["err"] += 1
            seen = stats["ok"] + stats["err"]
            if seen % 200 == 0 or seen == len(todo):
                print(f"  {seen}/{len(todo)}  (toxic {stats['toxic']}, errors {stats['err']})", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    print(
        f"\nwrote {stats['ok']} labels ({stats['toxic']} toxic) to {out_path}; "
        f"{stats['err']} errors (re-run to retry only those).",
        flush=True,
    )


def run_batch(items, client):
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [
        Request(
            custom_id=it["sample_id"],
            params=MessageCreateParamsNonStreaming(
                messages=[{"role": "user", "content": build_user(it)}], **CFG
            ),
        )
        for it in items
    ]
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch {batch.id} submitted ({len(requests)} requests). Polling...", flush=True)
    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"  status={batch.processing_status} processing={batch.request_counts.processing}", flush=True)
        time.sleep(30)
    rows = []
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            rows.append((result.custom_id, parse_label(result.result.message)))
        else:
            print(f"  [{result.custom_id}] {result.result.type}", flush=True)
            rows.append((result.custom_id, ""))
    return rows


def preview(rows):
    """Render the EXACT system + user messages for representative cases, no API call."""

    def first(pred):
        return next((r for r in rows if pred(r)), None)

    def nlex(r):
        s = (r.get("lexicon_terms") or "").strip()
        return len(s.split(", ")) if s else 0

    def starts(r, s):
        return (r.get("reply_chain") or "").startswith(s)

    def nparents(r):
        return sum(1 for ln in (r.get("reply_chain") or "").splitlines() if ln[:1] == "[")

    cases = [
        (
            "true top-level comment (directly under the post)",
            first(lambda r: starts(r, "THREAD CONTEXT: top-level")),
        ),
        (
            "reply that fully traces up to the post",
            first(lambda r: r["root"] == "submission" and starts(r, "THREAD CONTEXT (oldest")),
        ),
        ("deepest reply chain (most ancestors shown in full)", max(rows, key=nparents)),
        (
            "broken: immediate parent not collected",
            first(lambda r: starts(r, "THREAD CONTEXT: this is a reply")),
        ),
        (
            "partial chain, then an uncollected ancestor",
            first(lambda r: r["root"] == "broken" and starts(r, "THREAD CONTEXT (oldest")),
        ),
        ("lexicon-heavy comment", max(rows, key=nlex)),
        ("clean comment, no lexicon hits", first(lambda r: nlex(r) == 0)),
    ]
    out = [
        "=" * 100,
        f"SYSTEM PROMPT (identical & cached for all {len(rows)} comments; {len(SYSTEM)} chars)",
        "=" * 100,
        SYSTEM,
    ]
    for label, row in cases:
        out += ["", "#" * 100, f"# CASE: {label}"]
        if not row:
            out.append("#   (no matching case in the dataset)")
            continue
        out.append(
            f"#   id={row.get('sample_id') or row.get('id')}  subreddit=r/{row['subreddit']}  "
            f"root={row['root']}  omitted={row['omitted']}  lexicon_terms=[{row.get('lexicon_terms','')}]"
        )
        out += ["#" * 100, "----- USER MESSAGE (verbatim) -----", build_user(row)]
    text = "\n".join(out)
    (HERE / "prompt_previews.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n[written to {HERE / 'prompt_previews.txt'}]  No API call was made.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(OUTPUT), help="labels CSV (idempotent: resumes/append)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests")
    ap.add_argument("--limit", type=int, default=0, help="classify only the first N unlabeled (smoke test)")
    ap.add_argument("--preview", action="store_true", help="render real prompts and exit (no API call)")
    args = ap.parse_args()

    rows = load_rows(args.input)
    id_col = "sample_id" if (rows and "sample_id" in rows[0]) else "id"
    print(f"{len(rows)} comments from {args.input}; model={MODEL}; id_col={id_col}\n")

    if args.preview:
        preview(rows)
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY in the environment first.")
    client = anthropic.Anthropic()
    run_streaming(rows, client, Path(args.output), id_col, args.workers, limit=args.limit or None)


if __name__ == "__main__":
    main()
