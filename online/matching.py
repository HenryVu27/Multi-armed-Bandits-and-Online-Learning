"""Online fractional bipartite matching: water-filling against greedy.

Offline vertices u (capacity 1 each) are known in advance. Online vertices v arrive one at a time
with their edges, and each brings one unit that must be allocated to its neighbors at once.

Sources: the online primal-dual method of Henry Vu's deck "Online Algorithm and Online
Optimization" (online_learning/, slides 21-22: keep alpha * Delta P >= Delta D at every step), and
Buchbinder, Jain and Naor, "Online primal-dual algorithms for maximizing ad-auctions revenue",
ESA 2007, whose fractional allocation with unit bids and budgets is water-filling.

Water-filling pours v's unit into its least-filled neighbors, raising them to a common level h.
Its dual certificate sets alpha_u = g(level_u) with g(x) = (e^x - 1) / (e - 1) and
beta_v = 1 - g(h_v). This dual is feasible (alpha_u + beta_v >= 1 on every edge) and each arrival
raises it by at most e / (e - 1) times the primal gain, so ALG >= (1 - 1/e) OPT.
Greedy matches v whole to one free neighbor and only guarantees ALG >= OPT / 2.

Ratios here are ALG / OPT <= 1, the convention of the matching literature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

__all__ = [
    "WaterFillingResult",
    "water_filling",
    "greedy",
    "offline_maximum_matching",
    "upper_triangular_instance",
    "random_instance",
    "WATER_FILLING_RATIO",
    "GREEDY_RATIO",
]

WATER_FILLING_RATIO = 1.0 - 1.0 / np.e
GREEDY_RATIO = 0.5


@dataclass(frozen=True)
class WaterFillingResult:
    """allocation[v, u] is the fraction of online vertex v sent to offline vertex u."""

    allocation: np.ndarray
    alpha: np.ndarray  # dual variable of each offline vertex
    beta: np.ndarray  # dual variable of each online vertex

    @property
    def value(self) -> float:
        return float(self.allocation.sum())

    @property
    def dual_value(self) -> float:
        return float(self.alpha.sum() + self.beta.sum())


def _g(x: np.ndarray | float) -> np.ndarray:
    return (np.exp(x) - 1.0) / (np.e - 1.0)


def _pour(levels: np.ndarray, amount: float = 1.0) -> float:
    """Common water level h after pouring `amount` into bins at `levels`, each of capacity 1."""
    if np.sum(1.0 - levels) <= amount:
        return 1.0
    s = np.sort(levels)
    k = np.arange(1, s.size + 1)
    h = (amount + np.cumsum(s)) / k  # level if the k lowest bins share the water
    upper = np.append(s[1:], np.inf)
    valid = (h >= s) & (h <= upper)
    return float(h[np.argmax(valid)])


def water_filling(adjacency: np.ndarray) -> WaterFillingResult:
    """Run water-filling with online vertices arriving in row order of `adjacency[v, u]`."""
    adjacency = np.asarray(adjacency, dtype=bool)
    n_online, n_offline = adjacency.shape
    level = np.zeros(n_offline)
    allocation = np.zeros((n_online, n_offline))
    beta = np.zeros(n_online)
    for v in range(n_online):
        nbrs = np.flatnonzero(adjacency[v])
        if nbrs.size == 0:
            continue
        h = _pour(level[nbrs])
        poured = np.maximum(0.0, h - level[nbrs])
        allocation[v, nbrs] = poured
        level[nbrs] += poured
        beta[v] = 1.0 - _g(h)
    return WaterFillingResult(allocation, _g(level), beta)


def greedy(
    adjacency: np.ndarray,
    tie_break: str = "first",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Integral greedy: match each arriving vertex to a free neighbor if one exists.

    tie_break picks among free neighbors: "first" (lowest index), "last" (highest index, the
    adversarial choice on the upper-triangular instance) or "random" (needs `rng`).
    Returns a 0/1 allocation matrix.
    """
    adjacency = np.asarray(adjacency, dtype=bool)
    if tie_break == "random" and rng is None:
        raise ValueError("tie_break='random' needs an rng")
    n_online, n_offline = adjacency.shape
    free = np.ones(n_offline, dtype=bool)
    allocation = np.zeros((n_online, n_offline))
    for v in range(n_online):
        options = np.flatnonzero(adjacency[v] & free)
        if options.size == 0:
            continue
        if tie_break == "first":
            u = options[0]
        elif tie_break == "last":
            u = options[-1]
        elif tie_break == "random":
            u = rng.choice(options)
        else:
            raise ValueError(f"unknown tie_break {tie_break!r}")
        allocation[v, u] = 1.0
        free[u] = False
    return allocation


def offline_maximum_matching(adjacency: np.ndarray) -> int:
    """Size of a maximum matching, which equals the fractional optimum on bipartite graphs."""
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.size == 0:
        return 0
    rows, cols = linear_sum_assignment(adjacency.astype(float), maximize=True)
    return int(adjacency[rows, cols].sum())


def upper_triangular_instance(n: int) -> np.ndarray:
    """Online vertex v (0-based) is adjacent to offline vertices v, v + 1, ..., n - 1.

    The perfect matching v -> v exists, but an algorithm that spreads early arrivals evenly or
    commits them to high-index vertices strands the late arrivals.
    """
    return np.triu(np.ones((n, n), dtype=bool))


def random_instance(
    rng: np.random.Generator,
    n: int,
    edge_probability: float,
    planted: bool = True,
) -> np.ndarray:
    """Random bipartite graph with n online and n offline vertices.

    With `planted`, each online vertex also gets an edge to a distinct hidden partner, so a perfect
    matching exists and OPT = n.
    """
    adjacency = rng.random((n, n)) < edge_probability
    if planted:
        adjacency[np.arange(n), rng.permutation(n)] = True
    return adjacency
