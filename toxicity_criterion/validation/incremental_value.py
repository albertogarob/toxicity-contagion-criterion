"""
Incremental value of the graph: does network structure add useful evidence about
downstream toxic impact BEYOND the graph-free producer signals (ToxicCount, Volume)?

Four tests, all on the proper reply graph (B) and the fixed reply-forest ground truth:

  T1 Correlation: Spearman correlation of each score (producer + structural) with
     the downstream subtree toxicity rate / elicitation rate, over toxic users with
     enough support. Baseline strength of each signal alone.
  T2 Within-stratum: the same correlation for structural scores computed WITHIN
     toxic-count strata (tc=1, tc=2, tc>=3). If structure discriminates impact among
     users with the SAME toxic count, it adds evidence; if not, it is redundant.
  T3 Hybrids: top-10 by simple combinations (ToxicCount tie-broken by breadth;
     ToxicCount x log-breadth product; breadth among tc>=2). Do any beat plain
     ToxicCount's top-10 on the four impact metrics?
  T4 Complementarity: define ground-truth high-impact users (top-20 by downstream
     toxic count, and by subtree rate with support). Recall@10 of ToxicCount vs each
     structural method vs ToxicCount+structure union. Does structure recover impact
     users that ToxicCount misses?

Output: incremental_value.json
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from statistics import mean

if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import networkx as nx

from .. import config
from ..forest import build_reply_forest, downstream_metrics
from ..graph import build_reply_graph
from ..io import load_jsonl

RESULTS_DIR = config.RESULTS_DIR
TOXIC = config.SONNET_TOXIC
NONTOXIC = config.SONNET_NONTOXIC
OUT = os.path.join(RESULTS_DIR, "incremental_value.json")
TOP_K = 10
MIN_DIRECT = 5
MIN_DESC = 5


def rate_metrics(author, toxic, parent):
    direct_tot, direct_tox = Counter(), Counter()
    for c in author:
        p = parent.get(c)
        if p is not None and p in author and author[p] != author[c]:
            direct_tot[author[p]] += 1
            if toxic[c]:
                direct_tox[author[p]] += 1
    desc_tot, desc_tox = Counter(), Counter()
    for c in author:
        ca = author[c]
        seen = set()
        node = parent.get(c)
        hops = 0
        while node is not None and hops < 500:
            a = author.get(node)
            if a is not None and a != ca and a not in seen:
                seen.add(a)
                desc_tot[a] += 1
                if toxic[c]:
                    desc_tox[a] += 1
            node = parent.get(node)
            hops += 1
    return direct_tot, direct_tox, desc_tot, desc_tox


def spearman(xs, ys):
    """Spearman rho without scipy (average ranks for ties)."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def main():
    toxic = load_jsonl(TOXIC)
    non = load_jsonl(NONTOXIC)
    author, tx, parent = build_reply_forest(toxic + non)
    d_cnt, s_cnt = downstream_metrics(author, tx, parent)
    dt_tot, dt_tox, ds_tot, ds_tox = rate_metrics(author, tx, parent)

    g = build_reply_graph(toxic, non, weighting="proportion")
    mapper = g.anonymizer.anon_to_original
    toxic_users = [u for u in g.users if g.users[u].toxic_count >= 1]
    print(f"toxic users: {len(toxic_users)}")

    # per-user scores
    G = nx.DiGraph()
    for (u, v), w in g.edge_weights.items():
        G.add_edge(u, v, weight=w)
    pagerank = nx.pagerank(G, weight="weight")
    wd = defaultdict(float)
    for (u, v), w in g.edge_weights.items():
        wd[u] += w
        wd[v] += w
    scores = {}
    for u in toxic_users:
        o = mapper.get(u, u)
        scores[u] = {
            "ToxicCount": g.users[u].toxic_count,
            "Volume": g.users[u].total_posts,
            "ReplyBreadth": g.get_out_degree(u),
            "WtdDegree": wd.get(u, 0.0),
            "PageRank": pagerank.get(u, 0.0),
            "_orig": o,
            "_sub_rate": (ds_tox[o] / ds_tot[o]) if ds_tot[o] >= MIN_DESC else None,
            "_eli_rate": (dt_tox[o] / dt_tot[o]) if dt_tot[o] >= MIN_DIRECT else None,
            "_dwn_cnt": s_cnt.get(o, 0),
            "_dir_cnt": d_cnt.get(o, 0),
        }

    SIGNALS = ["ToxicCount", "Volume", "ReplyBreadth", "WtdDegree", "PageRank"]
    res = {}

    # ---- T1: marginal correlations ----
    t1 = {}
    sub_users = [u for u in toxic_users if scores[u]["_sub_rate"] is not None]
    eli_users = [u for u in toxic_users if scores[u]["_eli_rate"] is not None]
    for sig in SIGNALS:
        t1[sig] = {
            "rho_subtree_rate": round(
                spearman([scores[u][sig] for u in sub_users], [scores[u]["_sub_rate"] for u in sub_users])
                or 0,
                3,
            ),
            "rho_elicit_rate": round(
                spearman([scores[u][sig] for u in eli_users], [scores[u]["_eli_rate"] for u in eli_users])
                or 0,
                3,
            ),
            "rho_downstream_cnt": round(
                spearman([scores[u][sig] for u in toxic_users], [scores[u]["_dwn_cnt"] for u in toxic_users])
                or 0,
                3,
            ),
        }
    res["T1_marginal_spearman"] = {
        "n_subtree": len(sub_users),
        "n_elicit": len(eli_users),
        "n_all": len(toxic_users),
        "rho": t1,
    }

    # ---- T2: structural signal WITHIN toxic-count strata ----
    t2 = {}
    strata = {
        "tc=1": [u for u in sub_users if scores[u]["ToxicCount"] == 1],
        "tc=2": [u for u in sub_users if scores[u]["ToxicCount"] == 2],
        "tc>=3": [u for u in sub_users if scores[u]["ToxicCount"] >= 3],
    }
    for sname, us in strata.items():
        row = {"n": len(us)}
        for sig in ["ReplyBreadth", "WtdDegree", "PageRank", "Volume"]:
            rho = (
                spearman([scores[u][sig] for u in us], [scores[u]["_sub_rate"] for u in us])
                if len(us) >= 5
                else None
            )
            row[sig] = round(rho, 3) if rho is not None else None
        t2[sname] = row
    res["T2_within_stratum_subtree_rho"] = t2

    # ---- T3: hybrid top-10 vs plain ToxicCount top-10 ----
    def impact(top):
        os_ = [scores[u]["_orig"] for u in top]
        e = [scores[u]["_eli_rate"] for u in top if scores[u]["_eli_rate"] is not None]
        s = [scores[u]["_sub_rate"] for u in top if scores[u]["_sub_rate"] is not None]
        return {
            "direct_cnt": round(mean([d_cnt.get(o, 0) for o in os_]), 2),
            "downstream_cnt": round(mean([s_cnt.get(o, 0) for o in os_]), 2),
            "elicit_rate": round(mean(e), 3) if e else None,
            "subtree_rate": round(mean(s), 3) if s else None,
        }

    def topk(keyfn):
        return sorted(toxic_users, key=keyfn, reverse=True)[:TOP_K]

    hybrids = {
        "ToxicCount (plain)": topk(lambda u: (scores[u]["ToxicCount"], 0)),
        "TC, tie-break breadth": topk(lambda u: (scores[u]["ToxicCount"], scores[u]["ReplyBreadth"])),
        "TC x log(1+breadth)": topk(
            lambda u: scores[u]["ToxicCount"] * math.log(2 + scores[u]["ReplyBreadth"])
        ),
        "breadth | tc>=2": [
            u
            for u in sorted(toxic_users, key=lambda u: scores[u]["ReplyBreadth"], reverse=True)
            if scores[u]["ToxicCount"] >= 2
        ][:TOP_K],
        "Volume (plain)": topk(lambda u: (scores[u]["Volume"], 0)),
    }
    res["T3_hybrids"] = {name: impact(top) for name, top in hybrids.items()}

    # ---- T4: complementarity / recall of ground-truth high-impact users ----
    gt_cnt = [
        u
        for u, _ in sorted(
            ((u, scores[u]["_dwn_cnt"]) for u in toxic_users), key=lambda x: x[1], reverse=True
        )[:20]
    ]
    gt_rate = [
        u
        for u, _ in sorted(
            ((u, scores[u]["_sub_rate"]) for u in sub_users), key=lambda x: x[1], reverse=True
        )[:20]
    ]

    def recall(top, gt):
        return round(len(set(top) & set(gt)) / len(gt), 2)

    tc_top = hybrids["ToxicCount (plain)"]
    t4 = {}
    for sig in ["ReplyBreadth", "WtdDegree", "PageRank"]:
        s_top = topk(lambda u, s=sig: scores[u][s])
        t4[sig] = {
            "recall20_cnt": recall(s_top, gt_cnt),
            "recall20_rate": recall(s_top, gt_rate),
            "union_with_TC_recall20_cnt": recall(list(set(s_top) | set(tc_top)), gt_cnt),
            "union_with_TC_recall20_rate": recall(list(set(s_top) | set(tc_top)), gt_rate),
            "overlap_with_TC_top10": len(set(s_top) & set(tc_top)),
        }
    t4["ToxicCount"] = {"recall20_cnt": recall(tc_top, gt_cnt), "recall20_rate": recall(tc_top, gt_rate)}
    res["T4_complementarity"] = t4

    json.dump(res, open(OUT, "w"), indent=2)

    print("\nT1 marginal Spearman (signal -> impact):")
    for sig in SIGNALS:
        r = t1[sig]
        print(
            f"  {sig:<13} sub_rate={r['rho_subtree_rate']:>6}  eli_rate={r['rho_elicit_rate']:>6}  dwn_cnt={r['rho_downstream_cnt']:>6}"
        )
    print("\nT2 within-toxic-count-stratum Spearman vs subtree rate:")
    for sname, row in t2.items():
        print(f"  {sname:<6} n={row['n']:<4} " + "  ".join(f"{k}={row[k]}" for k in row if k != "n"))
    print("\nT3 hybrid top-10 impact:")
    for name, imp in res["T3_hybrids"].items():
        print(
            f"  {name:<22} dir={imp['direct_cnt']:>5} dwn={imp['downstream_cnt']:>5} "
            f"eli={imp['elicit_rate']} sub={imp['subtree_rate']}"
        )
    print("\nT4 recall@10 of ground-truth top-20 impact users:")
    for k, v in t4.items():
        print(f"  {k:<13} {v}")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
