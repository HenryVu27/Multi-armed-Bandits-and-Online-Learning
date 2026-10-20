"""Tests for online/matching.py: water-filling is (1 - 1/e)-competitive with a primal-dual certificate."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

from online.matching import (
    GREEDY_RATIO,
    WATER_FILLING_RATIO,
    greedy,
    offline_maximum_matching,
    random_instance,
    upper_triangular_instance,
    water_filling,
)

TOL = 1e-9


def random_graphs(rng, count=250):
    for _ in range(count):
        n_online, n_offline = int(rng.integers(1, 50)), int(rng.integers(1, 50))
        p = rng.uniform(0.01, 0.4)
        adjacency = rng.random((n_online, n_offline)) < p
        if rng.random() < 0.5 and n_online == n_offline:
            adjacency[np.arange(n_online), rng.permutation(n_offline)] = True
        yield adjacency


def check_certificate(adjacency, result):
    """Feasible primal, feasible dual, and P >= (1 - 1/e) D, which with weak duality gives the ratio."""
    alloc = result.allocation
    assert np.all(alloc >= -TOL)
    assert np.all(alloc[~adjacency] == 0)
    assert np.all(alloc.sum(axis=0) <= 1 + TOL) and np.all(alloc.sum(axis=1) <= 1 + TOL)
    assert np.all(result.alpha >= -TOL) and np.all(result.beta >= -TOL)
    edge_cover = (result.alpha[None, :] + result.beta[:, None])[adjacency]
    assert np.all(edge_cover >= 1 - TOL)
    assert result.value >= WATER_FILLING_RATIO * result.dual_value - TOL


def test_offline_optimum_matches_hopcroft_karp():
    rng = np.random.default_rng(0)
    for adjacency in random_graphs(rng, 100):
        expected = int((maximum_bipartite_matching(csr_matrix(adjacency), perm_type="column") >= 0).sum())
        assert offline_maximum_matching(adjacency) == expected


def test_water_filling_and_greedy_on_random_graphs():
    rng = np.random.default_rng(1)
    for adjacency in random_graphs(rng):
        opt = offline_maximum_matching(adjacency)
        result = water_filling(adjacency)
        check_certificate(adjacency, result)
        assert result.dual_value >= opt - TOL  # weak duality
        if opt == 0:
            continue
        assert result.value / opt >= WATER_FILLING_RATIO - TOL
        for tie_break in ("first", "last", "random"):
            matched = greedy(adjacency, tie_break, rng)
            assert matched.sum(axis=0).max(initial=0) <= 1 and matched.sum(axis=1).max(initial=0) <= 1
            assert matched.sum() / opt >= GREEDY_RATIO - TOL


def test_upper_triangular_hard_instance():
    previous = 1.0
    for n in [2, 5, 10, 50, 200, 600]:
        adjacency = upper_triangular_instance(n)
        assert offline_maximum_matching(adjacency) == n
        result = water_filling(adjacency)
        check_certificate(adjacency, result)
        ratio = result.value / n
        assert WATER_FILLING_RATIO - TOL <= ratio <= previous + TOL  # decreases toward 1 - 1/e
        previous = ratio
        # Adversarial tie-breaking sends each arrival to the highest free index and strands half of them.
        assert greedy(adjacency, "last").sum() == pytest.approx(np.ceil(n / 2))
        assert greedy(adjacency, "first").sum() == n
    assert previous == pytest.approx(WATER_FILLING_RATIO, abs=2e-3)


def test_water_filling_spreads_evenly():
    """The first arrival on the upper-triangular instance splits its unit equally over all n offline vertices."""
    n = 8
    result = water_filling(upper_triangular_instance(n))
    assert np.allclose(result.allocation[0], 1.0 / n)


def test_random_instance_has_perfect_matching():
    rng = np.random.default_rng(2)
    for n in [5, 30, 80]:
        assert offline_maximum_matching(random_instance(rng, n, 0.02)) == n
