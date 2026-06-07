"""
Build the agreement-stratified 200-comment validation sample.

Rationale: VALIDATION_SET_DESIGN.md. In short, we stratify on the 2x2 agreement table
between the two classifiers (GPT-OSS, Sonnet) plus a model-independent random anchor, so
that human labels can (a) validate Sonnet's in-domain precision AND recall, (b) adjudicate
the cases where the two classifiers disagree, and (c) anchor true prevalence. Labels for
both classifiers already exist (no new API calls); we only resample and rebuild the app.

Strata (sample sizes): agree_tox 45 | gpt_tox_son_non 55 | gpt_non_son_tox 45 |
                        agree_non 30 | random 25  (= 200)

Reuses build_validation_app's context + app machinery. Outputs:
  label_app.html           blind labeling app (agreement-stratified sample)
  validation_key.csv       sample_id, stratum, subreddit, gpt_label, sonnet_label, orig_id, author
  validation_populations.json  cell populations + corpus size + allocation + seed
"""

import csv
import html
import json
import random

from . import build_validation_app as B

HERE = B.HERE
SONNET_LABELS = HERE / "corpus_labels.csv"
SEED = 42
ALLOC = {"agree_tox": 45, "gpt_tox_son_non": 55, "gpt_non_son_tox": 45, "agree_non": 30, "random": 25}


def main():
    rng = random.Random(SEED)

    # GPT-OSS labels (original fine-tuned predictions): file membership = label
    gtox, gnon = B.load(B.TOXIC), B.load(B.NONTOXIC)
    gpt = {r["id"]: "toxic" for r in gtox}
    gpt.update({r["id"]: "non-toxic" for r in gnon})

    # Sonnet labels (validated full-corpus run)
    son = {}
    with open(SONNET_LABELS, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if r and r[1] in ("toxic", "non-toxic"):
                son[r[0]] = r[1]

    # raw corpus -> byid (for text + reply chain) and subreddit; HTML-unescaped text
    byid, id2sub = {}, {}
    for f in sorted(B.RAW_DIR.glob("*_dataset.jsonl")):
        sub = f.stem.replace("_dataset", "")
        for r in B.load(f):
            r["text"] = html.unescape(r.get("text") or "")
            byid[r["id"]] = r
            id2sub[r["id"]] = sub

    def eligible(cid):
        r = byid.get(cid)
        if not r:
            return False
        t = (r.get("text") or "").strip()
        return t.lower() not in B.SKIP and len(t) >= 3

    # display lexicon: built IDENTICALLY to the corpus dataset Sonnet saw
    for r in gtox + gnon:
        r["text"] = html.unescape(r.get("text") or "")
    id2text = {r["id"]: r.get("text", "") for r in gtox + gnon}
    single, multi = B.load_lexicon()
    df, ndoc = B.corpus_doc_freq(id2text)
    single = {w: c for w, c in single.items() if df.get(w, 0) / ndoc <= B.COMMON}

    # ---- 2x2 agreement cells over the overlap (comments both classifiers labeled) ----
    cells = {"agree_tox": [], "gpt_tox_son_non": [], "gpt_non_son_tox": [], "agree_non": []}
    for cid in son:
        if cid not in gpt or not eligible(cid):
            continue
        g, s = gpt[cid], son[cid]
        if g == "toxic" and s == "toxic":
            cells["agree_tox"].append(cid)
        elif g == "toxic" and s == "non-toxic":
            cells["gpt_tox_son_non"].append(cid)
        elif g == "non-toxic" and s == "toxic":
            cells["gpt_non_son_tox"].append(cid)
        else:
            cells["agree_non"].append(cid)
    pops = {k: len(v) for k, v in cells.items()}
    n_overlap = sum(pops.values())
    n_corpus = sum(1 for cid in son if eligible(cid))

    # ---- sample ----
    sample = []
    realized = {}
    for k in ("agree_tox", "gpt_tox_son_non", "gpt_non_son_tox", "agree_non"):
        n = min(ALLOC[k], len(cells[k]))
        realized[k] = n
        sample += [(cid, k) for cid in rng.sample(cells[k], n)]
    used = {cid for cid, _ in sample}
    rand_pool = [cid for cid in son if eligible(cid) and cid not in used]
    nr = min(ALLOC["random"], len(rand_pool))
    realized["random"] = nr
    sample += [(cid, "random") for cid in rng.sample(rand_pool, nr)]
    rng.shuffle(sample)

    # ---- build comments (app) + key + classifier-ingestion CSV, all from ONE loop ----
    # The app (label_app.html), the key (validation_key.csv), and the CSV the classifier
    # ingests (annotation_dataset_full.csv) MUST come from the same sample/loop, or the
    # human inspects one comment while the classifier labels another. (This is exactly the
    # bug that previously left annotation_dataset_full.csv stale from build_validation_app.py.)
    comments, key_rows, ds_rows = [], [], []
    for i, (cid, strat) in enumerate(sample, 1):
        sid = f"S{i:03d}"
        r = byid[cid]
        text = (r.get("text") or "").strip()
        sub = id2sub.get(cid, "unknown")
        parents, omitted, root = B.ancestor_chain(r, byid)
        lex = B.lexicon_hits(text, single, multi)
        comments.append(
            {
                "id": sid,
                "text": text,
                "subreddit": sub,
                "parents": parents,
                "omitted": omitted,
                "root": root,
                "lex": lex,
            }
        )
        key_rows.append([sid, strat, sub, gpt.get(cid, ""), son.get(cid, ""), cid, r.get("author", "")])
        ds_rows.append(
            [sid, sub, B.render_chain(parents, omitted, root), ", ".join(lex), text, root, omitted]
        )

    with open(HERE / "annotation_dataset_full.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo, quoting=csv.QUOTE_ALL)
        w.writerow(
            ["sample_id", "subreddit", "reply_chain", "lexicon_terms", "comment_text", "root", "omitted"]
        )
        w.writerows(ds_rows)

    (HERE / "validation_key.csv").write_text(
        "sample_id,stratum,subreddit,gpt_label,sonnet_label,orig_id,author\n"
        + "\n".join(",".join(f'"{c}"' for c in row) for row in key_rows)
        + "\n",
        encoding="utf-8",
    )

    pops_out = {
        "seed": SEED,
        "cell_populations": pops,
        "n_overlap": n_overlap,
        "n_corpus_eligible": n_corpus,
        "sample_allocation": realized,
        "note": "cells partition the GPT-OSS/Sonnet overlap; random drawn from full corpus",
    }
    (HERE / "validation_populations.json").write_text(json.dumps(pops_out, indent=2), encoding="utf-8")

    guidelines = (HERE / "ANNOTATION_GUIDELINES.md").read_text(encoding="utf-8")
    guidelines_html = guidelines.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    page = (
        B.HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(comments, ensure_ascii=False))
        .replace("/*__GUIDELINES__*/", guidelines_html)
        .replace('const KEY="toxlabels_v3";', 'const KEY="valid_agree_v1";')
    )
    (HERE / "label_app.html").write_text(page, encoding="utf-8")

    print(json.dumps(pops_out, indent=2))
    print(
        f"\nwrote label_app.html ({len(comments)} items), validation_key.csv, "
        f"validation_populations.json, annotation_dataset_full.csv (aligned to the key)"
    )


if __name__ == "__main__":
    main()
