"""Engagement-baseline foil , why the criterion measures transmission, not engagement.

The closest prior measure (an engagement-based "toxicity contagiousness score", a sum of a
message's reply/quote/like counts) reads this corpus *oppositely*: toxic comments attract more
reply engagement, so an engagement score ranks the corpus contagious. Yet almost none of those
replies are themselves toxic and the vast majority of toxic comments are isolated, so engagement
is not transmission. This module quantifies that divergence on the same corpus.

Run: ``python -m toxicity_criterion.criterion.engagement_baseline``
(reproduces 0.52 vs 0.43 mean replies; 38% vs 34% with >=1 reply; 94% isolated; 5.9% reply-toxic).
"""

from __future__ import annotations

import json
import os
import statistics as st
from collections import defaultdict

from .. import config
from ..io import load_jsonl

_OUT = os.path.join(config.RESULTS_DIR, "engagement_baseline.json")


def _strip_prefix(pid: str | None) -> str | None:
    if pid is None:
        return None
    return pid.split("_", 1)[1] if "_" in pid else pid


def run(
    toxic_path: str = config.SONNET_TOXIC, nontoxic_path: str = config.SONNET_NONTOXIC, out_path: str = _OUT
) -> dict:
    """Compare reply-engagement vs structural transmission on the corpus; write + return a dict."""
    pred: dict[str, str] = {}
    parent_of: dict[str, str | None] = {}
    children = defaultdict(list)

    records = load_jsonl(toxic_path) + load_jsonl(nontoxic_path)
    for rec in records:
        cid = rec["id"]
        pred[cid] = rec["prediction"]
        praw = _strip_prefix(rec.get("parent_id"))
        is_comment_parent = (rec.get("parent_id", "") or "").startswith("t1_")
        parent_of[cid] = praw if is_comment_parent else None

    for cid, p in parent_of.items():
        if p is not None and p in pred:
            children[p].append(cid)

    toxic_ids = [c for c in pred if pred[c] == "toxic"]
    nontox_ids = [c for c in pred if pred[c] == "non-toxic"]
    base_rate = len(toxic_ids) / len(pred)

    def reply_count(c):
        return len(children.get(c, []))

    def summ(xs):
        xs_sorted = sorted(xs)
        return {
            "n": len(xs),
            "mean": round(st.mean(xs), 3),
            "median": st.median(xs),
            "p90": xs_sorted[int(0.9 * len(xs_sorted))] if xs else 0,
            "max": max(xs) if xs else 0,
            "frac_with_>=1_reply": round(sum(1 for x in xs if x >= 1) / len(xs), 3) if xs else 0,
        }

    eng_tox = summ([reply_count(c) for c in toxic_ids])
    eng_non = summ([reply_count(c) for c in nontox_ids])

    replies_under_toxic = [k for c in toxic_ids for k in children.get(c, [])]
    replies_under_nontox = [k for c in nontox_ids for k in children.get(c, [])]
    p_tox_given_tox = (
        sum(1 for k in replies_under_toxic if pred[k] == "toxic") / len(replies_under_toxic)
        if replies_under_toxic
        else float("nan")
    )
    p_tox_given_non = (
        sum(1 for k in replies_under_nontox if pred[k] == "toxic") / len(replies_under_nontox)
        if replies_under_nontox
        else float("nan")
    )
    rr = p_tox_given_tox / p_tox_given_non if p_tox_given_non else float("nan")

    def isolated(c):
        p = parent_of.get(c)
        parent_tox = p in pred and pred[p] == "toxic"
        child_tox = any(pred[k] == "toxic" for k in children.get(c, []))
        return not (parent_tox or child_tox)

    n_isolated = sum(1 for c in toxic_ids if isolated(c))
    frac_isolated = n_isolated / len(toxic_ids)

    out = {
        "corpus": {
            "records": len(records),
            "toxic": len(toxic_ids),
            "non_toxic": len(nontox_ids),
            "base_rate": round(base_rate, 4),
        },
        "engagement_proxy_reply_count": {
            "toxic_parents": eng_tox,
            "nontoxic_parents": eng_non,
            "note": "Yousefi-style ReplyCount component; higher = more 'contagious' by their metric",
        },
        "structural_transmission": {
            "P(reply_toxic|parent_toxic)": round(p_tox_given_tox, 4),
            "P(reply_toxic|parent_nontoxic)": round(p_tox_given_non, 4),
            "base_rate_toxic": round(base_rate, 4),
            "risk_ratio_reply_level": round(rr, 3),
            "toxic_comments_isolated": n_isolated,
            "frac_toxic_isolated": round(frac_isolated, 4),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    out = run()
    print(json.dumps(out, indent=2))
    et = out["engagement_proxy_reply_count"]["toxic_parents"]
    en = out["engagement_proxy_reply_count"]["nontoxic_parents"]
    stt = out["structural_transmission"]
    print("\n--- DIVERGENCE ---")
    print(
        f"Engagement view: toxic comments get {et['mean']} replies on avg vs {en['mean']} for "
        f"non-toxic ({100 * et['frac_with_>=1_reply']:.0f}% of toxic comments draw >=1 reply)."
    )
    print(
        f"Structural view: a reply under a toxic comment is toxic "
        f"{100 * stt['P(reply_toxic|parent_toxic)']:.1f}% of the time; "
        f"{100 * stt['frac_toxic_isolated']:.0f}% of toxic comments are isolated. "
        "Engagement is present; contagion is not."
    )


if __name__ == "__main__":
    main()
