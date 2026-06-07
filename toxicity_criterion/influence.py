"""Influence-maximization layer: graph model, Independent Cascade, and CELF++.

Ported from the original ``mapping.py``; the toxicity-graded reply graph
(:class:`ToxicInfluenceGraph`), the :class:`IndependentCascadeModel`, and the
:class:`CELFPlusPlus` seed-selection algorithm used by the influence-maximization baseline.
"""

import heapq
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field

random.seed(42)


@dataclass
class User:

    username: str
    original_username: str
    toxic_count: int = 0
    non_toxic_count: int = 0
    posts: list[dict] = field(default_factory=list)

    @property
    def toxicity_ratio(self) -> float:
        total = self.toxic_count + self.non_toxic_count
        return self.toxic_count / total if total > 0 else 0

    @property
    def total_posts(self) -> int:
        return self.toxic_count + self.non_toxic_count


class UsernameAnonymizer:

    def __init__(self):
        self.original_to_anon: dict[str, str] = {}
        self.anon_to_original: dict[str, str] = {}
        self.counter = 0

    def anonymize(self, original: str) -> str:
        if original not in self.original_to_anon:
            self.counter += 1
            anon_name = f"User_{self.counter:03d}"
            self.original_to_anon[original] = anon_name
            self.anon_to_original[anon_name] = original
        return self.original_to_anon[original]

    def get_original(self, anon: str) -> str:
        return self.anon_to_original.get(anon, anon)

    def export_mapping(self) -> dict[str, str]:

        return self.anon_to_original.copy()


class ToxicInfluenceGraph:

    def __init__(self):
        self.out_edges: dict[str, set[str]] = defaultdict(set)
        self.in_edges: dict[str, set[str]] = defaultdict(set)
        self.edge_weights: dict[tuple[str, str], float] = {}
        self.users: dict[str, User] = {}
        self.post_authors: dict[str, str] = {}
        self.anonymizer = UsernameAnonymizer()

    def add_user(self, original_username: str) -> str:
        anon_username = self.anonymizer.anonymize(original_username)
        if anon_username not in self.users:
            self.users[anon_username] = User(username=anon_username, original_username=original_username)
        return anon_username

    def add_post(self, post: dict):
        original_author = post.get("author", "unknown")
        post_id = post.get("id", "")
        parent_id = post.get("parent_id", "")
        is_toxic = post.get("prediction", "") == "toxic"

        anon_author = self.add_user(original_author)
        self.post_authors[post_id] = anon_author

        if is_toxic:
            self.users[anon_author].toxic_count += 1
        else:
            self.users[anon_author].non_toxic_count += 1

        self.users[anon_author].posts.append(post)

        if parent_id.startswith("t1_"):
            parent_post_id = parent_id[3:]
            if parent_post_id in self.post_authors:
                parent_anon_author = self.post_authors[parent_post_id]
                if parent_anon_author != anon_author:
                    self.out_edges[parent_anon_author].add(anon_author)
                    self.in_edges[anon_author].add(parent_anon_author)
                    weight = 2.0 if is_toxic else 1.0
                    edge_key = (parent_anon_author, anon_author)
                    self.edge_weights[edge_key] = max(self.edge_weights.get(edge_key, 0), weight)

    def build_thread_network(self, posts: list[dict]):
        threads: dict[str, list[str]] = defaultdict(list)

        for post in posts:
            original_author = post.get("author", "unknown")
            anon_author = self.anonymizer.anonymize(original_author)
            parent_id = post.get("parent_id", "")

            if parent_id.startswith("t3_"):
                thread_id = parent_id
            elif parent_id.startswith("t1_"):
                thread_id = parent_id
            else:
                continue

            threads[thread_id].append(anon_author)

        for _thread_id, authors in threads.items():
            unique_authors = list(set(authors))
            if len(unique_authors) < 2:
                continue

            for i, author1 in enumerate(unique_authors):
                for author2 in unique_authors[i + 1 :]:
                    if author1 != author2:
                        user1_toxic = self.users[author1].toxic_count if author1 in self.users else 0
                        user2_toxic = self.users[author2].toxic_count if author2 in self.users else 0
                        weight = 1.0 + 0.5 * (user1_toxic > 0) + 0.5 * (user2_toxic > 0)

                        if author2 not in self.out_edges[author1]:
                            self.out_edges[author1].add(author2)
                            self.in_edges[author2].add(author1)
                            self.edge_weights[(author1, author2)] = weight

                        if author1 not in self.out_edges[author2]:
                            self.out_edges[author2].add(author1)
                            self.in_edges[author1].add(author2)
                            self.edge_weights[(author2, author1)] = weight

    def get_neighbors(self, user: str) -> set[str]:
        return self.out_edges.get(user, set())

    def get_out_degree(self, user: str) -> int:
        return len(self.out_edges.get(user, set()))

    def get_in_degree(self, user: str) -> int:
        return len(self.in_edges.get(user, set()))

    def get_candidate_users(
        self, min_toxic_posts: int = 1, min_connections: int = 0, max_candidates: int | None = None
    ) -> list[str]:
        candidates = []

        for username, user in self.users.items():
            if user.toxic_count < min_toxic_posts:
                continue

            connections = self.get_out_degree(username) + self.get_in_degree(username)
            if connections < min_connections:
                continue

            score = (
                user.toxic_count * 2
                + user.toxicity_ratio * 10
                + self.get_out_degree(username) * 1.5
                + self.get_in_degree(username) * 0.5
            )
            candidates.append((username, score))

        candidates.sort(key=lambda x: x[1], reverse=True)

        if max_candidates and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        return [username for username, _ in candidates]


class IndependentCascadeModel:

    def __init__(self, graph: ToxicInfluenceGraph, propagation_prob: float = 0.1):
        self.graph = graph
        self.base_prob = propagation_prob
        self._prob_cache: dict[tuple[str, str], float] = {}

    def get_propagation_probability(self, u: str, v: str) -> float:
        cache_key = (u, v)
        if cache_key in self._prob_cache:
            return self._prob_cache[cache_key]

        edge_weight = self.graph.edge_weights.get((u, v), 1.0)
        user_toxicity = self.graph.users[u].toxicity_ratio if u in self.graph.users else 0
        prob = self.base_prob * edge_weight * (1 + user_toxicity)
        prob = min(prob, 0.5)

        self._prob_cache[cache_key] = prob
        return prob

    def simulate_spread(self, seeds: set[str], num_simulations: int = 50) -> float:
        if not seeds:
            return 0.0

        total_spread = 0

        for _ in range(num_simulations):
            activated = set(seeds)
            newly_activated = set(seeds)
            max_iterations = 20
            iteration = 0

            while newly_activated and iteration < max_iterations:
                iteration += 1
                next_activated = set()

                for node in newly_activated:
                    neighbors = self.graph.get_neighbors(node)
                    for neighbor in neighbors:
                        if neighbor not in activated:
                            prob = self.get_propagation_probability(node, neighbor)
                            if random.random() < prob:
                                next_activated.add(neighbor)

                activated.update(next_activated)
                newly_activated = next_activated

            total_spread += len(activated)

        return total_spread / num_simulations


@dataclass(order=True)
class CELFNode:
    priority: float
    user: str = field(compare=False)
    mg1: float = field(compare=False)
    prev_best: str = field(compare=False)
    mg2: float = field(compare=False)
    flag: int = field(compare=False)


class CELFPlusPlus:

    def __init__(
        self,
        graph: ToxicInfluenceGraph,
        propagation_model: IndependentCascadeModel,
        num_simulations: int = 50,
    ):
        self.graph = graph
        self.model = propagation_model
        self.num_simulations = num_simulations
        self._spread_cache: dict[frozenset, float] = {}

    def _get_spread(self, seeds: set[str]) -> float:
        key = frozenset(seeds)
        if key not in self._spread_cache:
            self._spread_cache[key] = self.model.simulate_spread(seeds, self.num_simulations)
        return self._spread_cache[key]

    def compute_marginal_gain(self, seeds: set[str], candidate: str) -> float:
        current_spread = self._get_spread(seeds) if seeds else 0
        new_spread = self._get_spread(seeds | {candidate})
        return new_spread - current_spread

    def find_top_k_influential(
        self, k: int, candidates: list[str] | None = None, verbose: bool = True
    ) -> list[tuple[str, float]]:
        if candidates is None:
            candidates = self.graph.get_candidate_users(min_toxic_posts=1)

        if not candidates:
            print("No candidate users found")
            return []

        if verbose:
            print(f"Analyzing {len(candidates)} candidate users...")

        seeds: set[str] = set()
        results: list[tuple[str, float]] = []
        heap: list[CELFNode] = []

        start_time = time.time()

        if verbose:
            print("Computing initial influence scores...")

        for i, user in enumerate(candidates):
            mg1 = self.compute_marginal_gain(set(), user)
            node = CELFNode(priority=-mg1, user=user, mg1=mg1, prev_best="", mg2=mg1, flag=0)
            heapq.heappush(heap, node)

            if verbose and (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 1
                remaining = (len(candidates) - i - 1) / rate
                print(f"  Processed {i + 1}/{len(candidates)} users (~{remaining:.1f}s remaining)")

        if verbose:
            elapsed = time.time() - start_time
            print(f"  Initial pass completed in {elapsed:.1f}s")
            print(f"\nSelecting top {k} influential users...")

        iteration = 0
        last_seed = ""
        cur_best = heap[0] if heap else None

        while len(seeds) < k and heap:
            iteration += 1
            node = heapq.heappop(heap)

            if node.flag == len(seeds):
                seeds.add(node.user)
                results.append((node.user, node.mg1))
                last_seed = node.user

                if len(self._spread_cache) > 1000:
                    self._spread_cache.clear()

                if verbose:
                    user_data = self.graph.users[node.user]
                    print(f"\n  #{len(seeds)}: {node.user}")
                    print(f"     Influence Score: {node.mg1:.2f}")
                    print(f"     Toxic Posts: {user_data.toxic_count}/{user_data.total_posts}")
                    print(f"     Out-degree: {self.graph.get_out_degree(node.user)}")

                if heap:
                    cur_best = heap[0]
            else:
                if node.prev_best == last_seed and node.mg2 > 0:
                    node.mg1 = node.mg2
                else:
                    node.mg1 = self.compute_marginal_gain(seeds, node.user)

                if cur_best and cur_best.user != node.user:
                    node.prev_best = cur_best.user
                    node.mg2 = self.compute_marginal_gain(seeds | {cur_best.user}, node.user)

                node.flag = len(seeds)
                node.priority = -node.mg1
                heapq.heappush(heap, node)

        return results
