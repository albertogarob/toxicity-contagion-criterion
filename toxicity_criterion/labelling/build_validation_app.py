"""
Build a single-file HTML labeling app for the in-domain classifier validation.

Each item shows, as NEUTRAL context to help a human decide (it never shows any
model's verdict):
  - the parent comment being replied to (when available),
  - the comment to label,
  - hate-lexicon terms found in the comment, from HurtLex EN (an external,
    published lexicon), with their category, as an aid for spotting slang.

A flagged term does NOT mean toxic; the annotator judges in context.

Draws a seeded, stratified, BLIND sample (100 predicted-toxic + 100 predicted-
non-toxic). Writes the held-out key (sample_id -> model prediction, orig_id) for
scoring. Reproducible from SEED.

Run:  python3 build_validation_app.py
Then: open label_app.html, label, click "Download labels".
"""

import csv
import html
import json
import random
import re
from pathlib import Path

from . import lab_paths as P

SEED = 42
HERE = P.VAL  # writes the label app + validation artifacts under data/in_domain_validation
TOXIC = P.GPTOSS_TOXIC  # GPT-OSS stratifier predictions (regenerable; not shipped)
NONTOXIC = P.GPTOSS_NONTOXIC
RAW_DIR = P.RAW  # per-subreddit raw corpus (for subreddit + full reply chain)
HURTLEX = P.VAL / "lexicon" / "hurtlex_EN.tsv"
SKIP = {"[deleted]", "[removed]", "[deleted by user]", ""}
MINLEN = 3  # ignore very short lexicon lemmas to reduce noise
COMMON = 0.001  # drop lexicon words appearing in >0.1% of comments (real slurs are rare)
PER = 50  # comments per stratum (A/B/C/D), 200 total
CHAIN_CHARS = 12000  # reply-chain char budget (nearest-first). The 200 sample's deepest
# chain is ~6k chars, so nothing here is trimmed; this only guards pathologically deep
# threads in the full 35k. Kept identical for the 200 and the 35k (validation==deployment).


def load(path):
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


# HurtLex categories: keep the genuinely hate/offence-relevant ones, with readable
# names; drop noisy categories (professions 'pa', plants 'or', mild 'qas', vices
# 'svp') that flag common words.
CAT_NAMES = {
    "ps": "ethnic/racial slur",
    "rci": "origin/demonym",
    "om": "homophobic",
    "asf": "sexual (female)",
    "asm": "sexual (male)",
    "cds": "derogatory",
    "pr": "prostitution",
    "is": "social/economic slur",
    "ddf": "disability",
    "ddp": "cognitive disability",
    "dmc": "moral/behavioral",
    "re": "crime/immoral",
    "an": "animal/dehumanizing",
}

# Genuinely hate-LEANING categories (group-based hate, not generic profanity/sex),
# used to DEFINE stratum C so it probes missed HATE rather than mere swearing.
HATE_CATS = {"ps", "rci", "om", "is", "an", "ddf", "ddp"}


def load_lexicon(cats=CAT_NAMES):
    single, multi = {}, []
    for ln in HURTLEX.read_text(encoding="utf-8").splitlines()[1:]:
        parts = ln.split("\t")
        if len(parts) < 6:
            continue
        _, _, cat, _, lemma, level = parts[:6]
        if level.strip() != "conservative" or cat not in cats:
            continue
        lemma = lemma.strip().lower()
        if len(lemma) < MINLEN:
            continue
        if " " in lemma:
            multi.append(lemma)
        else:
            single[lemma] = CAT_NAMES[cat]
    return single, multi


def corpus_doc_freq(id2text):
    """document frequency of each word across the corpus (to drop common words)."""
    df = {}
    n = 0
    for t in id2text.values():
        n += 1
        for w in set(re.findall(r"[a-z']+", t.lower())):
            df[w] = df.get(w, 0) + 1
    return df, n


def lexicon_hits(text, single, multi):
    low = text.lower()
    hits = set()
    for w in re.findall(r"[a-z']+", low):
        if w in single:
            hits.add(w)
    for lemma in multi:
        if lemma in low:
            hits.add(lemma)
    return sorted(hits)


def ancestor_chain(r, byid, char_budget=CHAIN_CHARS):
    """Walk parent_id up the thread. Returns (parents_oldest_first, omitted, root).

    Includes the FULL chain of ancestor comments, nearest first, until adding the next
    (older) one would exceed char_budget; the immediate parent is always kept. Returns
    them oldest first (so the LAST element is the immediate parent). omitted: how many
    older ancestors were dropped by the budget (0 for normal threads). root: 'submission'
    (chain reached the post), 'broken' (an ancestor was not collected), or 'unknown'.
    The submission text itself is never available (comments-only scrape)."""
    anc, cur, seen, root = [], r, set(), "unknown"
    while (cur.get("parent_id") or "").startswith("t1_"):
        pid = cur["parent_id"][3:]
        if pid in seen or pid not in byid:
            root = "broken"
            break
        seen.add(pid)
        cur = byid[pid]
        anc.append((cur.get("text") or "").strip())
    else:
        root = "submission" if (cur.get("parent_id") or "").startswith("t3_") else "unknown"
    kept, used = [], 0  # anc = [immediate parent, grandparent, ...]
    for t in anc:
        if kept and used + len(t) > char_budget:
            break
        kept.append(t)
        used += len(t)
    return list(reversed(kept)), len(anc) - len(kept), root


def render_chain(parents, omitted, root):
    """Render the reply chain as the exact text used both in the dataset CSV and the
    Claude prompt (inspection == ingestion)."""
    if not parents:
        if root == "submission":
            return (
                "THREAD CONTEXT: top-level comment, replying to the original post (the post "
                "text was not collected, so only the subreddit signals the topic)."
            )
        return "THREAD CONTEXT: this is a reply, but the parent comment was not collected (unavailable)."
    lines = ["THREAD CONTEXT (oldest first; the last item is the immediate parent being replied to):"]
    if omitted:
        lines.append(
            f"(... {omitted} earlier comment{'s' if omitted > 1 else ''} in this thread omitted ...)"
        )
    for k, t in enumerate(parents, 1):
        tag = " (immediate parent)" if k == len(parents) else ""
        lines.append(f"[{k}]{tag} {t}")
    return "\n".join(lines)


def main():
    rng = random.Random(SEED)
    toxic, nontoxic = load(TOXIC), load(NONTOXIC)
    for r in toxic + nontoxic:  # decode HTML entities (&gt; &amp; &#39; ...)
        r["text"] = html.unescape(r.get("text") or "")
    id2text = {r.get("id"): r.get("text", "") for r in toxic + nontoxic}
    tox_ids = {r.get("id") for r in toxic}

    # Full raw corpus: maps every comment id -> record (for ancestor resolution) and
    # -> subreddit (recovered from the per-subreddit filename). Text is HTML-unescaped.
    byid, id2sub = {}, {}
    for f in sorted(RAW_DIR.glob("*_dataset.jsonl")):
        sub = f.stem.replace("_dataset", "")
        for r in load(f):
            r["text"] = html.unescape(r.get("text") or "")
            byid[r["id"]] = r
            id2sub[r["id"]] = sub

    def eligible(rows):
        return [
            r
            for r in rows
            if (r.get("text") or "").strip().lower() not in SKIP and len((r.get("text") or "").strip()) >= 3
        ]

    elig_tox, elig_non = eligible(toxic), eligible(nontoxic)

    # Display lexicon (broad hate+offence categories): the annotator AID, shown for
    # every comment regardless of stratum (so it does not leak the stratum).
    df, ndoc = corpus_doc_freq(id2text)
    single, multi = load_lexicon()
    before = len(single)
    single = {w: c for w, c in single.items() if df.get(w, 0) / ndoc <= COMMON}
    print(
        f"display lexicon: {len(single)} single-word (dropped {before-len(single)} common) + {len(multi)} multi-word"
    )

    # Hate-only lexicon that DEFINES stratum C (the false-negative probe): restricted
    # to hate-leaning categories so C targets missed HATE, not generic profanity.
    h_single, h_multi = load_lexicon(HATE_CATS)
    h_single = {w: c for w, c in h_single.items() if df.get(w, 0) / ndoc <= COMMON}

    def has_hate_term(text):
        return len(lexicon_hits(text, h_single, h_multi)) > 0

    # Split the predicted-non-toxic side into B (no hate term) and C (has hate term)
    # so A, B, C are DISJOINT and partition the corpus (A | B | C = all eligible).
    non_with = [r for r in elig_non if has_hate_term(r.get("text") or "")]
    non_without = [r for r in elig_non if not has_hate_term(r.get("text") or "")]
    print(f"predicted-non-toxic: {len(non_without)} no-hate-term (B), {len(non_with)} hate-term (C)")

    #   A = predicted-toxic                      -> GPT-OSS precision
    #   B = predicted-non-toxic, no hate term    -> implicit / no-flag misses
    #   C = predicted-non-toxic, has hate term   -> lexical false-negative probe
    #   D = pure random over the whole corpus    -> model-independent prevalence/accuracy
    A = [(r, "A") for r in rng.sample(elig_tox, PER)]
    B = [(r, "B") for r in rng.sample(non_without, PER)]
    nC = min(PER, len(non_with))
    C = [(r, "C") for r in rng.sample(non_with, nC)]
    used = {r.get("id") for r, _ in A + B + C}
    pool_D = [r for r in (elig_tox + elig_non) if r.get("id") not in used]  # disjoint from A/B/C
    nD = 4 * PER - (len(A) + len(B) + len(C))  # 50, plus top-up if C ran short
    D = [(r, "D") for r in rng.sample(pool_D, nD)]

    sample = A + B + C + D
    rng.shuffle(sample)  # fully interleave the strata: the app shows them in random order

    comments, key_rows = [], []
    for i, (r, stratum) in enumerate(sample, 1):
        sid = f"S{i:03d}"
        text = (r.get("text") or "").strip()
        sub = id2sub.get(r.get("id"), "unknown")
        parents, omitted, root = ancestor_chain(r, byid)
        pred = "toxic" if r.get("id") in tox_ids else "non-toxic"
        comments.append(
            {
                "id": sid,
                "text": text,
                "subreddit": sub,
                "parents": parents,
                "omitted": omitted,
                "root": root,
                "lex": lexicon_hits(text, single, multi),
            }
        )
        key_rows.append(
            [
                sid,
                stratum,
                sub,
                pred,
                "yes" if has_hate_term(text) else "no",
                r.get("id", ""),
                r.get("author", ""),
            ]
        )

    # The labeled dataset with its stratum, for full reproducibility (orig_id lets
    # anyone re-fetch the comment; stratum drives the reweighted estimator).
    (HERE / "annotation_key.csv").write_text(
        "sample_id,stratum,subreddit,model_prediction,has_hate_lexicon,orig_id,author\n"
        + "\n".join(",".join(f'"{c}"' for c in row) for row in key_rows)
        + "\n",
        encoding="utf-8",
    )

    # Aggregated dataset: one row per comment with ALL the context (subreddit, the
    # rendered reply chain, lexicon terms, the comment). This is the canonical input
    # ingested by classify_claude.py AND a human-inspectable view. Contains Reddit
    # text -> gitignored. Order/seed identical to the app and the key.
    with open(HERE / "annotation_dataset_full.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(
            ["sample_id", "subreddit", "reply_chain", "lexicon_terms", "comment_text", "root", "omitted"]
        )
        for c in comments:
            w.writerow(
                [
                    c["id"],
                    c["subreddit"],
                    render_chain(c["parents"], c["omitted"], c["root"]),
                    ", ".join(c["lex"]),
                    c["text"],
                    c["root"],
                    c["omitted"],
                ]
            )

    # Stratum populations for the Horvitz-Thompson (prevalence-reweighted) estimator.
    pops = {
        "seed": SEED,
        "N_A_pred_toxic": len(elig_tox),
        "N_B_pred_nontoxic_no_hatelex": len(non_without),
        "N_C_pred_nontoxic_hatelex": len(non_with),
        "N_corpus_eligible": len(elig_tox) + len(elig_non),
        "sample_per_stratum": {"A": len(A), "B": len(B), "C": len(C), "D": len(D)},
    }
    (HERE / "strata_populations.json").write_text(json.dumps(pops, indent=2), encoding="utf-8")

    # context (parent + lexicon) for the Claude classifier, so it sees the SAME inputs
    (HERE / "annotation_context.json").write_text(
        json.dumps(comments, ensure_ascii=False, indent=0), encoding="utf-8"
    )
    guidelines = (HERE / "ANNOTATION_GUIDELINES.md").read_text(encoding="utf-8")
    guidelines_html = guidelines.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    page = HTML_TEMPLATE.replace("/*__DATA__*/", json.dumps(comments, ensure_ascii=False)).replace(
        "/*__GUIDELINES__*/", guidelines_html
    )
    (HERE / "label_app.html").write_text(page, encoding="utf-8")
    print(
        f"Wrote label_app.html, annotation_context.json, annotation_dataset_full.csv, "
        f"annotation_key.csv, strata_populations.json ({len(comments)} items). Seed={SEED}."
    )
    print(json.dumps(pops, indent=2))


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Toxicity labeling</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:780px;margin:24px auto;padding:0 16px;color:#1a1a1a}
 .bar{height:8px;background:#eee;border-radius:4px;overflow:hidden;margin:8px 0}
 .bar>div{height:100%;background:#2471a3;width:0}
 .meta{display:flex;justify-content:space-between;color:#666;font-size:14px}
 .parent{border-left:3px solid #bbb;background:#f7f7f7;padding:10px 14px;margin:8px 0;color:#444;font-size:15px;white-space:pre-wrap}
 .parent .lab{color:#999;font-size:12px;text-transform:uppercase;letter-spacing:.5px}
 .sub{display:inline-block;background:#34495e;color:#fff;font-size:13px;font-weight:600;padding:3px 11px;border-radius:12px;margin:8px 0}
 .ctxnote{color:#999;font-size:12px;font-style:italic;margin:8px 0 2px}
 .card{border:1px solid #ddd;border-radius:10px;padding:20px;margin:6px 0;font-size:19px;line-height:1.5;white-space:pre-wrap;min-height:80px}
 .lxh{background:#fdd9d4;color:#c0392b;font-weight:700;border-radius:3px;padding:0 2px}
 .lex{font-size:16px;margin:10px 0;padding:10px 12px;border-radius:8px;border:1px solid}
 .lex b{font-weight:700}
 .lex.has{background:#fdecea;border-color:#e6a39a;color:#922b21}
 .lex.none{background:#eaf2fb;border-color:#a9c7e6;color:#1f4e79}
 .btns{display:flex;gap:12px;margin-top:8px}
 button{flex:1;padding:14px;font-size:16px;border:0;border-radius:8px;cursor:pointer;color:#fff}
 .tox{background:#c0392b}.non{background:#27ae60}.skip{background:#7f8c8d}.back{background:#34495e;flex:0 0 90px}
 .fwd{background:#34495e;flex:0 0 90px}
 .dl{background:#2471a3;margin-top:18px;width:100%}
 .reset{background:#7f8c8d;margin-top:8px;width:100%;font-size:13px;padding:8px}
 .hint{color:#666;font-size:13px;margin-top:10px}
 kbd{background:#eee;border-radius:4px;padding:1px 6px;font-family:monospace}
 .done{font-size:20px;color:#27ae60;text-align:center;padding:30px}
</style></head><body>
<h2>Is the comment toxic?</h2>
<details style="margin:8px 0;border:1px solid #ddd;border-radius:8px;padding:6px 12px;background:#fafafa">
<summary style="cursor:pointer;font-weight:600">Annotation guidelines (click to expand)</summary>
<pre style="white-space:pre-wrap;font-size:13px;line-height:1.45;font-family:system-ui,Arial,sans-serif">/*__GUIDELINES__*/</pre></details>
<p class="hint">Decide for yourself using the context below. The parent comment is shown for context; the
lexicon terms are an <b>aid</b> (a flagged word is <b>not</b> automatically toxic, e.g. quoted, reclaimed, or
non-hateful use). "Toxic" = hate <i>or</i> offensive/abusive language; heated-but-civil disagreement is non-toxic.
Keys: <kbd>T</kbd> toxic · <kbd>N</kbd> non-toxic · <kbd>U</kbd> unsure · <kbd>&larr;</kbd> back. Auto-saves.</p>
<div class="meta"><span id="counter"></span><span id="counts"></span></div>
<div class="bar"><div id="prog"></div></div>
<div id="sub" class="sub"></div>
<div id="ctx"></div>
<div class="card" id="text"></div>
<div class="lex" id="lex"></div>
<div class="btns">
 <button class="back" onclick="back()">&larr; Back</button>
 <button class="tox" onclick="label('toxic')">Toxic (T)</button>
 <button class="non" onclick="label('non-toxic')">Non-toxic (N)</button>
 <button class="skip" onclick="label('unsure')">Unsure (U)</button>
 <button class="fwd" onclick="fwd()">Fwd &rarr;</button>
</div>
<button class="dl" onclick="download()">Download labels (CSV)</button>
<button class="reset" onclick="reset_all()">Clear all labels &amp; start over</button>
<p class="hint" id="status"></p>
<script>
const COMMENTS=/*__DATA__*/;
const KEY="toxlabels_v3";
let labels=JSON.parse(localStorage.getItem(KEY)||"{}");
let i=COMMENTS.findIndex(c=>!labels[c.id]); if(i<0)i=COMMENTS.length-1;
function save(){localStorage.setItem(KEY,JSON.stringify(labels))}
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;")}
function isAlpha(ch){return ch!==undefined&&/[a-z']/i.test(ch);}
function highlight(text,terms){
 if(!terms||!terms.length) return esc(text);
 const low=text.toLowerCase();
 let ranges=[];
 for(const t of terms){
  const tl=(t||"").toLowerCase(); if(!tl) continue;
  const phrase=tl.indexOf(" ")>=0;
  let idx=0;
  while(true){
   const j=low.indexOf(tl,idx); if(j<0) break;
   const e=j+tl.length;
   if(phrase||((!isAlpha(low[j-1]))&&(!isAlpha(low[e])))) ranges.push([j,e]);
   idx=j+1;
  }
 }
 if(!ranges.length) return esc(text);
 ranges.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 let merged=[];
 for(const r of ranges){
  const last=merged[merged.length-1];
  if(last&&r[0]<=last[1]) last[1]=Math.max(last[1],r[1]); else merged.push([r[0],r[1]]);
 }
 let out="",pos=0;
 for(const [s,e] of merged){
  out+=esc(text.slice(pos,s))+'<span class="lxh">'+esc(text.slice(s,e))+'</span>';
  pos=e;
 }
 return out+esc(text.slice(pos));
}
function ctxHtml(c){
 const ps=c.parents||[];
 if(ps.length===0){
   return c.root==="submission"
     ? '<div class="parent"><span class="lab">Top-level comment: replying to the post (post text not collected)</span></div>'
     : '<div class="parent"><span class="lab">Replying to a comment that was not collected</span></div>';
 }
 let h="";
 if(c.omitted>0) h+='<div class="ctxnote">… '+c.omitted+' earlier comment'+(c.omitted>1?'s':'')+' in this thread omitted …</div>';
 ps.forEach((t,k)=>{
   const last=k===ps.length-1;
   const lab=last?"Immediate parent (being replied to)":"Earlier in thread";
   h+='<div class="parent" style="margin-left:'+(Math.min(k,4)*14)+'px"><span class="lab">'+lab+'</span><br>'+esc(t)+'</div>';
 });
 return h;
}
function render(){
 const done=Object.keys(labels).length, total=COMMENTS.length;
 document.getElementById("counts").textContent=done+" / "+total+" labeled";
 document.getElementById("prog").style.width=(100*done/total)+"%";
 if(i>=total){document.getElementById("sub").textContent="";document.getElementById("ctx").innerHTML="";document.getElementById("lex").innerHTML="";
   document.getElementById("text").innerHTML='<div class="done">All '+total+' labeled. Click <b>Download labels</b>.</div>';
   document.getElementById("counter").textContent="Done";return;}
 const c=COMMENTS[i];
 document.getElementById("counter").textContent="Item "+(i+1)+" of "+total+(labels[c.id]?"  (labeled: "+labels[c.id]+")":"");
 document.getElementById("sub").textContent="r/"+(c.subreddit||"unknown");
 document.getElementById("ctx").innerHTML=ctxHtml(c);
 document.getElementById("text").innerHTML=highlight(c.text,c.lex);
 const lex=document.getElementById("lex");
 if(c.lex&&c.lex.length){lex.className="lex has";lex.innerHTML="Potentially offensive terms (HurtLex): "+c.lex.map(t=>'<b>'+esc(t)+'</b>').join(", ");}
 else{lex.className="lex none";lex.innerHTML="No lexicon terms found.";}
}
function label(v){if(i<COMMENTS.length){labels[COMMENTS[i].id]=v;save();i++;render();}}
function back(){if(i>0){i--;render();}}
function fwd(){if(i<COMMENTS.length){i++;render();}}
function reset_all(){if(confirm("Clear ALL "+Object.keys(labels).length+" labels and start over? This cannot be undone.")){labels={};localStorage.removeItem(KEY);i=0;render();}}
function download(){
 let rows=[["sample_id","human_label"]];
 COMMENTS.forEach(c=>rows.push([c.id,labels[c.id]||""]));
 const csv=rows.map(r=>r.join(",")).join("\n");
 const b=new Blob([csv],{type:"text/csv"}), a=document.createElement("a");
 a.href=URL.createObjectURL(b); a.download="human_labels.csv"; a.click();
 document.getElementById("status").textContent="Saved human_labels.csv ("+Object.keys(labels).length+" labeled).";
}
document.addEventListener("keydown",e=>{
 const k=e.key.toLowerCase();
 if(k==="t")label("toxic");else if(k==="n")label("non-toxic");
 else if(k==="u")label("unsure");else if(e.key==="ArrowLeft")back();else if(e.key==="ArrowRight")fwd();
});
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
