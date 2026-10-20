"""Tests for online/ski_rental.py against Henry's deck and Bamas, Maggiori and Svensson (NeurIPS 2020) Theorem 2."""

import numpy as np
import pytest

from online.ski_rental import (
    break_even_cost,
    break_even_ratio,
    buy_day_distribution,
    follow_prediction_cost,
    offline_cost,
    pdla,
    pdla_bounds,
    pdla_cost_curve,
    primal_dual,
    primal_dual_ratio,
    randomized_buy_day_expected_cost,
    round_online,
)

RTOL = 1e-9
# Theorem 2 assumes lam * B and B / lam are integers; these pairs satisfy both.
CASES = [(100, 0.2), (100, 0.25), (100, 0.5), (100, 0.8), (100, 1.0), (40, 0.5), (8, 0.5), (20, 0.2)]


def test_break_even_is_two_minus_one_over_B_competitive():
    for B in [1, 2, 5, 20, 100]:
        ratios = [break_even_cost(N, B) / offline_cost(N, B) for N in range(1, 4 * B + 2)]
        assert max(ratios) == pytest.approx(break_even_ratio(B))
        assert break_even_cost(B, B) / offline_cost(B, B) == pytest.approx(break_even_ratio(B))


def test_deck_buy_day_distribution():
    """Slide 11: the distribution sums to 1 and equalizes the ratio at alpha = 1 / (1 - (1 - 1/B)^B)."""
    for B in [2, 4, 10, 100, 1000]:
        p = buy_day_distribution(B)
        assert p.sum() == pytest.approx(1.0)
        alpha = 1.0 / (1.0 - (1.0 - 1.0 / B) ** B)
        ratios = [randomized_buy_day_expected_cost(N, B) / offline_cost(N, B) for N in range(1, 2 * B + 2)]
        assert max(ratios) == pytest.approx(alpha)
        assert alpha <= np.e / (np.e - 1)
    assert randomized_buy_day_expected_cost(1, 4) / 1 == pytest.approx(1.0 / (1.0 - 0.75**4))


def test_primal_dual_is_e_over_e_minus_one_competitive():
    for B in [5, 20, 100, 400]:
        bound = primal_dual_ratio(B)
        for N in range(1, 4 * B):
            run = primal_dual(N, B)
            assert run.cost <= bound * offline_cost(N, B) * (1 + RTOL)
            assert run.dual_value <= B + 1e-9  # dual feasibility: sum of y_j <= B
            assert np.all(np.minimum(run.x, 1.0) + run.f >= 1.0 - 1e-12)  # every day is covered
        # The bound is reached at N = B and tends to e / (e - 1).
        assert primal_dual(B, B).cost / B == pytest.approx(bound, rel=1e-9)
    assert primal_dual_ratio(10**6) == pytest.approx(np.e / (np.e - 1), rel=1e-5)


def test_online_rounding_matches_fractional_cost_in_expectation():
    rng = np.random.default_rng(0)
    for N, B, N_pred, lam in [(30, 20, 50, 0.5), (150, 100, 40, 0.5), (7, 20, 5, 1.0)]:
        run = pdla(N, B, N_pred, lam)
        draws = np.array([round_online(run, rng) for _ in range(20000)])
        assert draws.mean() == pytest.approx(run.cost, rel=0.02)
        assert set(np.unique(draws)) <= {N} | {k + B for k in range(1, N + 1)}


@pytest.mark.parametrize("B, lam", CASES)
def test_pdla_robustness_for_every_prediction(B, lam):
    consistency, robustness = pdla_bounds(lam, B)
    n_max = 5 * B
    N = np.arange(1, n_max + 1)
    opt = np.minimum(N, B)
    for N_pred in [1, B // 2, B - 1, B, B + 1, 3 * B, n_max]:
        cost = pdla_cost_curve(B, N_pred, lam, n_max)
        assert np.max(cost / opt) <= robustness * (1 + RTOL)
        follow = np.array([follow_prediction_cost(n, B, N_pred) for n in N])
        assert np.max(cost / follow) <= consistency * (1 + RTOL)


@pytest.mark.parametrize("B, lam", CASES)
def test_pdla_consistency_with_a_perfect_prediction(B, lam):
    consistency, robustness = pdla_bounds(lam, B)
    ratios = [pdla(N, B, N, lam).cost / offline_cost(N, B) for N in range(1, 4 * B)]
    assert max(ratios) <= consistency * (1 + RTOL)
    assert max(ratios) == pytest.approx(consistency, rel=1e-9)  # reached once N >= B


@pytest.mark.parametrize("B, lam", CASES)
def test_pdla_adversarial_prediction_reaches_robustness(B, lam):
    """Predict a long season, stop skiing on the day the fast schedule finishes buying (N = lam B)."""
    _, robustness = pdla_bounds(lam, B)
    N = round(lam * B)
    assert pdla(N, B, 10 * B, lam).cost / offline_cost(N, B) == pytest.approx(robustness, rel=1e-9)


def test_lambda_one_recovers_primal_dual():
    for B in [10, 100]:
        for N in [1, B // 2, B, 3 * B]:
            for N_pred in [1, B, 5 * B]:
                assert pdla(N, B, N_pred, 1.0).cost == pytest.approx(primal_dual(N, B).cost)
        assert pdla_bounds(1.0, B) == pytest.approx((primal_dual_ratio(B),) * 2)


def test_bounds_limit_and_tradeoff():
    for lam in [0.1, 0.3, 0.6, 1.0]:
        c, r = pdla_bounds(lam)
        assert c == pytest.approx(lam / (1 - np.exp(-lam)))
        assert r == pytest.approx(1 / (1 - np.exp(-lam)))
        assert 1.0 <= c <= np.e / (np.e - 1) + 1e-12 <= r + 1e-12
    with pytest.raises(ValueError):
        pdla_bounds(0.0)
