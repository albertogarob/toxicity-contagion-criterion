"""Reply-forest construction and structural utilities.

A reply forest is represented by three dicts keyed on comment id:
``author[cid]`` (str), ``toxic[cid]`` (bool), and ``parent[cid]`` (parent comment id, or
``None`` for a thread root). These helpers are the single source of truth for the parent
pointers, topological ordering, downstream-impact accounting, and the toxic component /
chain statistics used by both the criterion and the validation modules.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx


def build_reply_forest(posts: list[dict]) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    """Build ``(author, toxic, parent)`` from post records.

    Toxicity is read from ``prediction`` == "toxic" when present, else from a boolean
    ``toxic`` field. ``parent_id`` uses Reddit prefixes: ``t1_<id>`` is a comment parent,
    anything else (e.g. ``t3_<post>``) marks a thread root (``parent`` = ``None``).
    """
    author: dict[str, str] = {}
    toxic: dict[str, bool] = {}
    parent: dict[str, str] = {}
    for p in posts:
        cid = p.get("id", "")
        if cid == "":
            continue
        author[cid] = p.get("author", "unknown")
        if "prediction" in p:
            toxic[cid] = str(p.get("prediction", "")).strip().lower() == "toxic"
        else:
            toxic[cid] = bool(p.get("toxic", False))
        pid = p.get("parent_id", "") or ""
        parent[cid] = pid[3:] if pid.startswith("t1_") else None
    return author, toxic, parent


def reply_edges(author: dict[str, str], parent: dict[str, str]) -> list[tuple[str, str]]:
    """Comment-level reply edges ``(parent_cid, child_cid)`` with both endpoints present."""
    return [(parent[c], c) for c in author if parent.get(c) in author]


def topo_order(author: dict[str, str], parent: dict[str, str]) -> list[str]:
    """Comment ids ordered by depth from their thread root (parents before children)."""
    depth: dict[str, int] = {}
    for c in author:
        d, node = 0, parent.get(c)
        while node in author and d < 1000:
            node = parent.get(node)
            d += 1
        depth[c] = d
    return sorted(author, key=lambda c: depth[c])


def downstream_metrics(
    author: dict[str, str], toxic: dict[str, bool], parent: dict[str, str]
) -> tuple[Counter, dict[str, int]]:
    """Per-user ground truth from the real reply forest.

    Returns ``(direct, subtree)`` where ``direct[u]`` counts toxic comments posted as a direct
    reply to one of ``u``'s comments, and ``subtree[u]`` counts distinct toxic comments anywhere
    downstream of ``u`` (each ancestor author credited at most once per toxic comment, excluding
    the toxic comment's own author).
    """
    direct: Counter = Counter()
    subtree_sets: dict[str, set] = defaultdict(set)
    toxic_ids = [c for c, t in toxic.items() if t]
    for t in toxic_ids:
        ta = author[t]
        par = parent.get(t)
        if par is not None and par in author and author[par] != ta:
            direct[author[par]] += 1
        seen_auth: set = set()
        node = parent.get(t)
        hops = 0
        while node is not None and hops < 200:
            a = author.get(node)
            if a is not None and a != ta and a not in seen_auth:
                seen_auth.add(a)
                subtree_sets[a].add(t)
            node = parent.get(node)
            hops += 1
    subtree = {u: len(s) for u, s in subtree_sets.items()}
    return direct, subtree


def largest_toxic_component(author: dict[str, str], toxic: dict[str, bool], parent: dict[str, str]) -> int:
    """Size (in users) of the largest connected component joined by toxic reply exchanges."""
    G = nx.Graph()
    for c in author:
        p = parent.get(c)
        if p is not None and p in author and author[p] != author[c] and toxic[c]:
            G.add_edge(author[p], author[c])
    return max((len(x) for x in nx.connected_components(G)), default=0)


def longest_toxic_chain(order: list[str], parent: dict[str, str], toxic: dict[str, bool]) -> int:
    """Longest run of consecutive toxic comments along any root-to-leaf reply path."""
    run: dict[str, int] = {}
    best = 0
    for c in order:
        if toxic[c]:
            p = parent.get(c)
            run[c] = (run.get(p, 0) + 1) if (p is not None and toxic.get(p)) else 1
            best = max(best, run[c])
        else:
            run[c] = 0
    return best


def chain_histogram(order: list[str], parent: dict[str, str], toxic: dict[str, bool]) -> Counter:
    """Counts of maximal toxic chains by length (length 1 = isolated toxic singletons)."""
    run: dict[str, int] = {}
    for c in order:
        run[c] = (
            (run.get(parent.get(c), 0) + 1)
            if (toxic[c] and toxic.get(parent.get(c)))
            else (1 if toxic[c] else 0)
        )
    has_tox_child: set = set()
    for c in order:
        p = parent.get(c)
        if toxic[c] and p is not None and toxic.get(p):
            has_tox_child.add(p)
    return Counter(run[c] for c in order if toxic[c] and c not in has_tox_child)
