"""Directed reply-graph construction over all users.

Builds a :class:`~toxicity_criterion.influence.ToxicInfluenceGraph` from the labelled corpus:
an edge runs from a comment's author to each user who replies, weighted by the proportion of
toxic interactions on the edge (``w in [1, 2]`` under the default "proportion" scheme). The
graph is used only to run the network methods under test; the criterion itself uses none of it.
"""

from __future__ import annotations

from collections import defaultdict

from .influence import ToxicInfluenceGraph

COUNT_CAP = 3  # saturation cap for the 'count' weighting scheme


def build_reply_graph(
    toxic_posts: list[dict], nontoxic_posts: list[dict], weighting: str = "proportion"
) -> ToxicInfluenceGraph:
    """Build a directed weighted reply graph over all users.

    ``weighting`` is one of ``"proportion"`` (default; ``1 + toxic/total``), ``"binary"``
    (``2`` if any toxic reply else ``1``), or ``"count"`` (``1 + min(toxic, CAP)/CAP``).
    """
    g = ToxicInfluenceGraph()
    all_posts = toxic_posts + nontoxic_posts
    all_posts.sort(key=lambda p: p.get("id", ""))

    # Pass 1: register every user, count toxic/non-toxic posts, map post -> author.
    for post in all_posts:
        author = g.add_user(post.get("author", "unknown"))
        g.post_authors[post.get("id", "")] = author
        if post.get("prediction", "") == "toxic":
            g.users[author].toxic_count += 1
        else:
            g.users[author].non_toxic_count += 1
        g.users[author].posts.append(post)

    # Pass 2: accumulate per-edge reply statistics (order-independent).
    edge_total = defaultdict(int)
    edge_toxic = defaultdict(int)
    for post in all_posts:
        parent_id = post.get("parent_id", "")
        if not parent_id.startswith("t1_"):
            continue  # only comment->comment replies carry an author-to-author edge
        parent_author = g.post_authors.get(parent_id[3:])
        if parent_author is None:
            continue
        replier = g.post_authors.get(post.get("id", ""))
        if replier is None or replier == parent_author:
            continue
        edge = (parent_author, replier)  # influence flows parent -> replier
        edge_total[edge] += 1
        if post.get("prediction", "") == "toxic":
            edge_toxic[edge] += 1

    # Build directed edges with the chosen weighting.
    for edge, total in edge_total.items():
        u, v = edge
        nt = edge_toxic[edge]
        if weighting == "binary":
            w = 2.0 if nt > 0 else 1.0
        elif weighting == "proportion":
            w = 1.0 + nt / total
        elif weighting == "count":
            w = 1.0 + min(nt, COUNT_CAP) / COUNT_CAP
        else:
            raise ValueError(weighting)
        g.out_edges[u].add(v)
        g.in_edges[v].add(u)
        g.edge_weights[edge] = w

    return g
