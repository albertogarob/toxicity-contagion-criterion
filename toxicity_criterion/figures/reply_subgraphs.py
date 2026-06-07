"""Dataset-backed reply-subgraph examples for the case study.

Extracts ACTUAL subgraphs from the Sonnet-labelled reply forest and renders them, so
the figures illustrate exactly the numbers reported in the paper:
  Fig A (fig_examples.pdf): (a) locality thread, (b) largest toxic component (7 users),
                            (c) deepest toxic chain (3) beside a toxic pair.
  Fig B (fig_reach.pdf):    a high-reach producer's subtree (few toxic, many descendants).

Node identities are anonymized to generic ids; toxic = filled orange, non-toxic = open grey.
Reproducible: PYTHONHASHSEED pinned, fixed layout seed.
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execvpe(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .. import config
from ..forest import build_reply_forest
from ..io import load_jsonl

FIG_DIR = config.FIG_DIR
ROOT = FIG_DIR
SEED = 42
TOXC = "#d62728"  # toxic: filled red
TOXE = "#7f1212"  # toxic: dark-red outline
NONC = "#d9d9d9"  # non-toxic: light grey fill
NONE = "#7a7a7a"  # non-toxic edge ring


def draw(ax, G, tox, pos, rings=None, nsize=180):
    ax.set_axis_off()
    nx.draw_networkx_edges(
        ax=ax,
        G=G,
        pos=pos,
        edge_color="#555555",
        arrows=True,
        arrowsize=8,
        width=1.0,
        node_size=nsize,
        min_target_margin=2,
    )
    tx = [n for n in G if tox.get(n)]
    nt = [n for n in G if not tox.get(n)]
    nx.draw_networkx_nodes(
        ax=ax, G=G, pos=pos, nodelist=nt, node_color=NONC, edgecolors=NONE, linewidths=0.7, node_size=nsize
    )
    nx.draw_networkx_nodes(
        ax=ax, G=G, pos=pos, nodelist=tx, node_color=TOXC, edgecolors=TOXE, linewidths=0.7, node_size=nsize
    )
    if rings:
        rings = [rings] if not isinstance(rings, (list, set, tuple)) else list(rings)
        nx.draw_networkx_nodes(
            ax=ax,
            G=G,
            pos=pos,
            nodelist=rings,
            node_color="none",
            edgecolors="#1a1a1a",
            linewidths=1.4,
            node_size=nsize * 1.9,
        )


def tree_layout(G, xgap=1.0, tree_gap=1.6):
    """Tidy top-down layout for a forest (DiGraph parent->child); root at top."""
    sys.setrecursionlimit(10000)
    ch = {n: [] for n in G}
    indeg = {n: 0 for n in G}
    for u, v in G.edges():
        ch[u].append(v)
        indeg[v] += 1
    roots = [n for n in G if indeg[n] == 0]
    pos = {}
    cur = [0.0]

    def place(n, depth, seen):
        seen.add(n)
        kids = [c for c in ch[n] if c not in seen]
        if not kids:
            x = cur[0]
            cur[0] += xgap
        else:
            xs = [place(c, depth + 1, seen) for c in kids]
            x = (min(xs) + max(xs)) / 2.0
        pos[n] = (x, -depth)
        return x

    seen = set()
    for r in roots:
        place(r, 0, seen)
        cur[0] += tree_gap
    return pos, roots


def main():
    author, tox, parent = build_reply_forest(
        load_jsonl(config.SONNET_TOXIC) + load_jsonl(config.SONNET_NONTOXIC)
    )
    C = list(author)
    children = defaultdict(list)
    for c in C:
        p = parent.get(c)
        if p in author:
            children[p].append(c)

    # ---------- (a) locality thread: non-toxic parent, >=4 replies, some toxic ----------
    par = next(
        p
        for c in C
        if tox[c]
        for p in [parent.get(c)]
        if p in author and not tox[p] and len(children.get(p, [])) >= 4
    )
    kids = children[par][:5]
    Ga = nx.DiGraph()
    toxa = {}
    Ga.add_node("p")
    toxa["p"] = tox[par]
    for i, k in enumerate(kids):
        Ga.add_edge("p", f"r{i}")
        toxa[f"r{i}"] = tox[k]
    posa = {"p": (0, 1)}
    for i in range(len(kids)):
        posa[f"r{i}"] = ((i - (len(kids) - 1) / 2) * 0.6, 0)

    # ---------- (b) largest toxic component (author space, toxic-reply edges) ----------
    aedges = [
        (author[p], author[c], c) for c in C if (p := parent.get(c)) in author and author[p] != author[c]
    ]
    Gtox = nx.DiGraph()  # directed: edge = a toxic reply (replier -> recipient)
    for pa, ca, c in aedges:
        if tox[c]:
            Gtox.add_edge(pa, ca)
    comp = max(nx.connected_components(Gtox.to_undirected()), key=len)
    Gb = Gtox.subgraph(comp).copy()
    toxb = {n: True for n in Gb}  # every node here is a user in toxic exchanges
    posb = nx.spring_layout(Gb, seed=SEED, k=0.9)

    # ---------- (c) deepest toxic chain (3) + a toxic pair ----------
    order = sorted(C, key=lambda c: _depth(c, parent, author))
    run = {}
    for c in order:
        run[c] = (
            (run.get(parent.get(c), 0) + 1) if (tox[c] and tox.get(parent.get(c))) else (1 if tox[c] else 0)
        )

    def chain_from(endpoint):
        ch = [endpoint]
        while parent.get(ch[-1]) in author and tox.get(parent.get(ch[-1])):
            ch.append(parent[ch[-1]])
        return ch[::-1]

    triple = chain_from(next(c for c in order if run.get(c) == 3))
    pair = chain_from(next(c for c in order if run.get(c) == 2))
    Gc = nx.DiGraph()
    toxc = {}
    for j, _ in enumerate(triple):
        Gc.add_node(f"t{j}")
        toxc[f"t{j}"] = True
        if j:
            Gc.add_edge(f"t{j-1}", f"t{j}")
    for j, _ in enumerate(pair):
        Gc.add_node(f"d{j}")
        toxc[f"d{j}"] = True
        if j:
            Gc.add_edge(f"d{j-1}", f"d{j}")
    posc = {f"t{j}": (0, -j) for j in range(len(triple))}
    posc.update({f"d{j}": (1.1, -j - 0.5) for j in range(len(pair))})

    # ----- render Fig A (3 panels) -----
    figA, axs = plt.subplots(1, 3, figsize=(7.1, 2.25))
    draw(axs[0], Ga, toxa, posa, rings=["p"])
    axs[0].set_title("(a) Local: toxic replies\nunder a non-toxic parent", fontsize=8)
    draw(axs[1], Gb, toxb, posb, nsize=150)
    axs[1].set_title("(b) Largest toxic\ncomponent (7 users)", fontsize=8)
    draw(axs[2], Gc, toxc, posc)
    axs[2].set_title("(c) Deepest toxic chain (3)\nand a toxic pair", fontsize=8)
    figA.tight_layout()
    outA = os.path.join(ROOT, "fig_examples.pdf")
    figA.savefig(outA, bbox_inches="tight")
    print("wrote", outA)

    # ---------- Fig B: high-reach producer subtree (low toxicity, high reach) ----------
    prod = Counter(author[c] for c in C if tox[c])

    def subtree(u):
        roots = [c for c in C if author[c] == u]
        nodes = set(roots)
        stack = list(roots)
        while stack:
            x = stack.pop()
            for k in children.get(x, []):
                if k not in nodes:
                    nodes.add(k)
                    stack.append(k)
        return roots, nodes

    # pick the highest-reach producer; show each thread only to depth D so long thin chains
    # do not waste vertical space (the reach point is breadth, not depth)
    cand = [u for u, _ in prod.most_common(10)]
    u = max(cand, key=lambda uu: len(subtree(uu)[1]) - len(subtree(uu)[0]))
    roots, nodes = subtree(u)
    # full reply graph over the subtree, then cap by STRUCTURAL depth (the depth the tidy layout
    # uses) so long thin chains do not waste vertical space; reach is breadth, not depth
    Gfull = nx.DiGraph()
    for c in nodes:
        p = parent.get(c)
        Gfull.add_edge(p, c) if p in nodes else Gfull.add_node(c)
    Gfull.remove_nodes_from([n for n in list(Gfull) if Gfull.degree(n) == 0])
    D = 3
    sroots = [n for n in Gfull if Gfull.in_degree(n) == 0]
    sdepth = {r: 0 for r in sroots}
    keep = set(sroots)
    trunc = set()
    q = list(sroots)
    while q:
        x = q.pop(0)
        for k in Gfull.successors(x):
            if sdepth[x] < D:
                if k not in sdepth:
                    sdepth[k] = sdepth[x] + 1
                    keep.add(k)
                    q.append(k)
            else:
                trunc.add(x)
    Gr = Gfull.subgraph(keep).copy()
    Gr.remove_nodes_from([n for n in list(Gr) if Gr.degree(n) == 0])
    toxr = {n: tox[n] for n in Gr}
    posr, _ = tree_layout(Gr, xgap=1.0, tree_gap=1.6)
    rings = [n for n in Gr if author[n] == u]  # ring all of the producer's own comments
    figB, axb = plt.subplots(figsize=(7.1, 2.7))  # depth-capped: wide, short, dense (little whitespace)
    draw(axb, Gr, toxr, posr, rings=rings, nsize=34)

    def hidden_downstream(x):  # # of further (non-producer) replies below x
        cnt = 0
        st = list(children.get(x, []))
        seen = set()
        while st:
            a = st.pop()
            if a in nodes and a not in seen:
                seen.add(a)
                if author[a] != u:
                    cnt += 1
                st.extend(children.get(a, []))
        return cnt

    hsum = 0
    for n in trunc:  # label threads that continue below the shown depth
        if n in posr:
            hc = hidden_downstream(n)
            if hc <= 0:
                continue
            hsum += hc
            x, y = posr[n]
            axb.annotate(
                f"+{hc}",
                xy=(x, y),
                xytext=(x, y - 0.95),
                ha="center",
                va="top",
                fontsize=6.5,
                color="#555",
                arrowprops=dict(arrowstyle="-", ls=":", color="#aaa", lw=0.8),
            )
    axb.margins(x=0.02, y=0.16)
    ndesc = len(nodes) - len(roots)
    shown_down = sum(1 for n in Gr if author[n] != u)
    print(
        f"  fig_reach: producer={u[:7]} reach={ndesc} toxic={prod[u]} "
        f"shown_down={shown_down} hidden_sum={hsum} shown+hidden={shown_down + hsum} "
        f"shown_depth={max((sdepth.get(n,0) for n in Gr), default=0)} trunc_threads={len(trunc)}"
    )
    axb.set_title(
        f"High-reach producer: own comments (ringed) almost all non-toxic "
        f"(red = toxic), yet {ndesc} downstream replies",
        fontsize=9,
    )
    figB.tight_layout()
    outB = os.path.join(ROOT, "fig_reach.pdf")
    figB.savefig(outB, bbox_inches="tight")
    print("wrote", outB)
    print(
        f"stats: comp={len(comp)} chain={len(triple)} pair={len(pair)} "
        f"locality_kids={len(kids)} toxic_kids={sum(toxa[f'r{i}'] for i in range(len(kids)))} "
        f"reach_root_tox={prod[u]} reach_desc={len(nodes)-len(roots)}"
    )


def _depth(c, parent, author):
    d = 0
    n = parent.get(c)
    while n in author and d < 500:
        n = parent.get(n)
        d += 1
    return d


if __name__ == "__main__":
    main()
