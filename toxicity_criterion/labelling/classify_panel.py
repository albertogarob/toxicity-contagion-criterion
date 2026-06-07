"""
Multi-LLM silver-standard panel for the 200-comment validation set.

We have no annotators fluent in UK slang/politics, so a true human gold standard is
not achievable. Instead we build an LLM-consensus *silver* standard from three SOTA
models with three INDEPENDENT training lineages, and report the (acknowledged-weak)
human labels alongside it:

  PANEL (voters, 2-of-3 majority -> reference label):
    opus    = claude-opus-4-8     (Anthropic)
    gpt     = gpt-5.5             (OpenAI)
    gemini  = gemini-3.1-pro      (Google)

  SUBJECTS (evaluated AGAINST the panel, never voters):
    Sonnet  (claude-sonnet-4-6, the deployed classifier)  and  GPT-OSS.
  The classifier under evaluation must not vote on its own evaluation, and GPT-OSS is
  the known-unreliable one; both stay out of the panel.

Every model sees the IDENTICAL prompt the Sonnet classifier and the human annotator
see: same ANNOTATION_GUIDELINES.md rubric (SYSTEM) and same per-comment context
(SUBREDDIT + THREAD CONTEXT + HATE-LEXICON TERMS + COMMENT TO LABEL, via build_user).
Reusing classify_claude.SYSTEM/build_user guarantees inspection == ingestion.

Blind (no model sees another's label), models pinned, structured single-label output.
Idempotent streaming per model: one CSV per model, append+flush+fsync, resume by id.
Keys come from the environment (gitignored .env): ANTHROPIC_API_KEY, OPENAI_API_KEY,
GEMINI_API_KEY. Never printed.

Usage:
  python3 classify_panel.py --preview                 # render prompts + show wiring, NO API call
  python3 classify_panel.py --smoke                   # 1 comment per model (cheap live wiring test)
  python3 classify_panel.py --models opus,gpt,gemini  # full run over the 200 (default: all three)
  python3 classify_panel.py --limit 10                # first 10 unlabeled per model
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

# Reuse the EXACT rubric + context rendering the deployed classifier/human use.
from classify_claude import SCHEMA, SYSTEM, build_user

from . import lab_paths as P

HERE = P.VAL
DATASET = HERE / "annotation_dataset_full.csv"  # the 200 validation comments + context
VALID_KEY = HERE / "validation_key.csv"  # canonical 200 sample_ids
ID_COL = "sample_id"

# Pinned model ids (latest of each lineage, per the panel spec).
MODELS = {
    "opus": {"provider": "anthropic", "id": "claude-opus-4-8"},
    "gpt": {"provider": "openai", "id": "gpt-5.5"},
    "gemini": {"provider": "google", "id": os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")},
}

# The label task is a closed binary; we ask every provider for the same JSON object.
JSON_INSTRUCTION = '\n\nReturn ONLY a JSON object of the form {"label": "toxic"} or {"label": "non-toxic"}.'


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def valid_ids():
    return [r[ID_COL] for r in load_rows(VALID_KEY)]


def assert_aligned(rows):
    """Guard against the stale-dataset bug: every sample's comment_text in DATASET must
    match the authoritative corpus text for its orig_id (validation_key -> corpus_dataset).
    A misaligned file means the panel would label the wrong comments; abort before spend."""
    corpus = HERE / "corpus_dataset_full.csv"
    if not corpus.exists():
        print("[guard] corpus_dataset_full.csv absent; skipping alignment check.")
        return
    key = {r[ID_COL]: r for r in load_rows(VALID_KEY)}
    corp = {r["id"]: r for r in load_rows(corpus)}
    bad = []
    for r in rows:
        sid = r[ID_COL]
        oid = key.get(sid, {}).get("orig_id")
        truth = (corp.get(oid, {}).get("comment_text") or "").strip()
        if truth and (r.get("comment_text") or "").strip() != truth:
            bad.append(sid)
    if bad:
        sys.exit(
            f"[guard] ABORT: {len(bad)} sample(s) in {DATASET.name} are misaligned with "
            f"validation_key/corpus (e.g. {bad[:5]}). Rebuild with build_validation_sample.py."
        )
    print(f"[guard] alignment OK: all {len(rows)} samples match corpus text for their orig_id.")


def parse_label(text):
    """Pull a toxic/non-toxic label out of a model reply (JSON preferred, then bare word)."""
    if not text:
        return None
    t = text.strip()
    try:
        v = json.loads(t).get("label", "")
        if v in ("toxic", "non-toxic"):
            return v
    except (json.JSONDecodeError, AttributeError):
        pass
    low = t.lower()
    # bare-word / fenced fallbacks; check non-toxic first ("toxic" is a substring of it)
    if "non-toxic" in low or "non toxic" in low:
        return "non-toxic"
    if "toxic" in low:
        return "toxic"
    return None


# ---- provider backends: (system, user) -> "toxic" | "non-toxic" | None --------------


def _anthropic_client():
    import anthropic

    return anthropic.Anthropic()


def classify_anthropic(client, model_id, system, user):
    import anthropic

    for attempt in range(7):
        try:
            msg = client.messages.create(
                model=model_id,
                max_tokens=200,
                thinking={"type": "disabled"},
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": user}],
            )
            text = next((b.text for b in msg.content if b.type == "text"), "")
            return parse_label(text)
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError):
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) and 500 <= e.status_code < 600:
                time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
                continue
            raise
    return None


def _openai_client():
    from openai import OpenAI

    return OpenAI()  # reads OPENAI_API_KEY


def classify_openai(client, model_id, system, user):
    import openai

    # Structured output via json_schema response format; reasoning models ignore temperature.
    rf = {"type": "json_schema", "json_schema": {"name": "label", "strict": True, "schema": SCHEMA}}
    for attempt in range(7):
        try:
            r = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format=rf,
            )
            return parse_label(r.choices[0].message.content)
        except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
            time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
        except openai.APIStatusError as e:
            if getattr(e, "status_code", 0) and 500 <= e.status_code < 600:
                time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
                continue
            raise
    return None


def _gemini_client():
    from google import genai

    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def classify_gemini(client, model_id, system, user):
    from google.genai import errors as gerrors
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {"label": {"type": "string", "enum": ["toxic", "non-toxic"]}},
            "required": ["label"],
        },
    )
    for attempt in range(7):
        try:
            r = client.models.generate_content(model=model_id, contents=user, config=cfg)
            return parse_label(r.text)
        except gerrors.APIError as e:
            code = getattr(e, "code", 0) or 0
            if code in (429, 500, 502, 503, 504):
                time.sleep(min(2**attempt, 30) + random.uniform(0, 1))
                continue
            raise
    return None


BACKENDS = {
    "anthropic": (_anthropic_client, classify_anthropic),
    "openai": (_openai_client, classify_openai),
    "google": (_gemini_client, classify_gemini),
}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GEMINI_API_KEY"}


# Anthropic's SYSTEM already ends with a single-label instruction; the other providers
# get an explicit JSON-object instruction appended (they have no system-cache contract).
def system_for(provider):
    return SYSTEM if provider == "anthropic" else SYSTEM + JSON_INSTRUCTION


# ---- streaming driver (one model) ---------------------------------------------------


def out_path(name):
    return HERE / f"panel_{name}.csv"


def run_model(name, rows, workers, limit=None):
    spec = MODELS[name]
    provider, model_id = spec["provider"], spec["id"]
    if not os.environ.get(KEY_ENV[provider]):
        print(f"[{name}] missing {KEY_ENV[provider]} in env; skipping.", flush=True)
        return
    make_client, classify = BACKENDS[provider]
    client = make_client()
    system = system_for(provider)

    out = out_path(name)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd, None)
            for r in rd:
                if r and r[0]:
                    done.add(r[0])
    todo = [r for r in rows if r[ID_COL] not in done]
    if limit:
        todo = todo[:limit]
    print(f"[{name}] {model_id}: {len(done)} done, {len(todo)} to label, workers={workers}", flush=True)
    if not todo:
        print(f"[{name}] complete.", flush=True)
        return

    new_file = not out.exists()
    fout = open(out, "a", newline="", encoding="utf-8")
    w = csv.writer(fout)
    if new_file:
        w.writerow([ID_COL, f"{name}_label"])
        fout.flush()
    lock = threading.Lock()
    stats = {"ok": 0, "toxic": 0, "err": 0}

    def work(row):
        try:
            label = classify(client, model_id, system, build_user(row))
        except Exception as e:  # non-retryable -> leave unlabeled
            print(f"  [{name}/{row[ID_COL]}] error: {type(e).__name__}: {str(e)[:140]}", flush=True)
            label = None
        with lock:
            if label:
                w.writerow([row[ID_COL], label])
                fout.flush()
                os.fsync(fout.fileno())
                stats["ok"] += 1
                stats["toxic"] += label == "toxic"
            else:
                stats["err"] += 1
            seen = stats["ok"] + stats["err"]
            if seen % 25 == 0 or seen == len(todo):
                print(
                    f"  [{name}] {seen}/{len(todo)} (toxic {stats['toxic']}, err {stats['err']})", flush=True
                )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, todo))
    fout.close()
    print(
        f"[{name}] wrote {stats['ok']} ({stats['toxic']} toxic); {stats['err']} errors -> {out.name}",
        flush=True,
    )


# ---- preview / smoke (no spend, or 1 call each) ------------------------------------


def preview(rows):
    r = rows[0]
    print("=" * 90)
    print("PANEL WIRING (no API call)")
    print("=" * 90)
    for name, spec in MODELS.items():
        env = KEY_ENV[spec["provider"]]
        present = "set" if os.environ.get(env) else "MISSING"
        print(f"  {name:7} -> {spec['provider']:10} id={spec['id']:20} key[{env}]={present}")
    print("\nAnthropic SYSTEM ends with the single-label instruction (cached).")
    print("OpenAI/Gemini SYSTEM has this appended:", repr(JSON_INSTRUCTION.strip()))
    print(f"\nSYSTEM ({len(SYSTEM)} chars), shared & identical across the panel:")
    print("-" * 90)
    print(SYSTEM)
    print("-" * 90)
    print(f"\nSAMPLE USER MESSAGE (sample_id={r[ID_COL]}, identical bytes sent to all 3 models):")
    print("-" * 90)
    print(build_user(r))
    print("-" * 90)
    print("\n[no API call was made]")


def smoke(rows):
    r = rows[0]
    print(f"SMOKE: labeling sample_id={r[ID_COL]} once per available model.\n")
    for name, spec in MODELS.items():
        prov = spec["provider"]
        if not os.environ.get(KEY_ENV[prov]):
            print(f"  {name:7} skip (no {KEY_ENV[prov]})")
            continue
        make_client, classify = BACKENDS[prov]
        try:
            t0 = time.time()
            label = classify(make_client(), spec["id"], system_for(prov), build_user(r))
            print(f"  {name:7} ({spec['id']}) -> {label}   [{time.time()-t0:.1f}s]")
        except Exception as e:
            print(f"  {name:7} ({spec['id']}) -> ERROR {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="opus,gpt,gemini", help="comma list of: opus,gpt,gemini")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="first N unlabeled per model (smoke/partial)")
    ap.add_argument("--preview", action="store_true", help="render prompts + wiring, no API call")
    ap.add_argument("--smoke", action="store_true", help="one live call per model (cheap wiring test)")
    args = ap.parse_args()

    ids = set(valid_ids())
    rows = [r for r in load_rows(DATASET) if r[ID_COL] in ids]
    rows.sort(key=lambda r: r[ID_COL])
    print(f"{len(rows)} validation comments loaded (of {len(ids)} ids in validation_key).\n")

    assert_aligned(rows)  # never spend on a stale/misaligned dataset

    if args.preview:
        preview(rows)
        return
    if args.smoke:
        smoke(rows)
        return

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    bad = [m for m in names if m not in MODELS]
    if bad:
        sys.exit(f"unknown model(s): {bad}; choose from {list(MODELS)}")
    for name in names:
        run_model(name, rows, args.workers, limit=args.limit or None)


if __name__ == "__main__":
    main()
