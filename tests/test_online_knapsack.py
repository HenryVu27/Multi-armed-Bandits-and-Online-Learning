"""Tests for online/knapsack.py against the Zhou-Chakrabarty-Lukose ratio 1 + ln(U / L)."""

import numpy as np
import pytest
from scipy.optimize import linprog

from online.knapsack import (
    flood_instance,
    greedy_knapsack,
    offline_fractional_knapsack,
    ramp_instance,
    random_instance,
    threshold_knapsack,
    zcl_beta,
    zcl_competitive_ratio,
    zcl_threshold,
    zcl_utilization_at,
)

RTOL = 1e-9
RANGES = [(1.0, 1.2), (1.0, 2.0), (1.0, 10.0), (3.0, 300.0), (1.0, 1000.0)]


def test_threshold_satisfies_the_primal_dual_design():
    """Slides 32-33: Psi = L up to beta, Psi(1) = U, and Psi(z) = alpha * integral of Psi from 0 to z after beta."""
    for L, U in RANGES:
        alpha, beta = zcl_competitive_ratio(L, U), zcl_beta(L, U)
        assert alpha * beta == pytest.approx(1.0)
        assert zcl_threshold(beta, L, U) == pytest.approx(L)
        assert zcl_threshold(1.0, L, U) == pytest.approx(U)
        z = np.linspace(beta, 1.0, 50)
        integral = L * beta + (zcl_threshold(z, L, U) - L) / np.log(U * np.e / L)
        assert np.allclose(zcl_threshold(z, L, U), alpha * integral)
        d = np.linspace(L, U, 200)
        assert np.allclose(zcl_threshold(zcl_utilization_at(d, L, U), L, U), d)


def test_offline_optimum_matches_linear_program():
    rng = np.random.default_rng(0)
    for _ in range(30):
        values, weights = random_instance(rng, 1.0, 20.0, n_items=12, total_weight=rng.uniform(0.3, 4.0))
        lp = linprog(-values, A_ub=weights[None, :], b_ub=[1.0], bounds=[(0, 1)] * values.size)
        assert offline_fractional_knapsack(values, weights).value == pytest.approx(-lp.fun, rel=1e-7)


@pytest.mark.parametrize("L, U", RANGES)
def test_threshold_rule_stays_within_one_plus_log_theta(L, U):
    rng = np.random.default_rng(1)
    bound = zcl_competitive_ratio(L, U)
    ratios = []
    for _ in range(300):
        values, weights = random_instance(rng, L, U, int(rng.integers(5, 80)), rng.uniform(0.2, 6.0))
        result = threshold_knapsack(values, weights, L, U)
        assert np.all((0 <= result.fractions) & (result.fractions <= 1 + 1e-12))
        assert np.dot(result.fractions, weights) <= 1 + 1e-9
        ratios.append(offline_fractional_knapsack(values, weights).value / result.value)
    # Adversarial: densities rise to p then weight 1 arrives at p; also in increasing and decreasing order.
    for p in np.linspace(L, U, 25):
        values, weights = ramp_instance(p, L, n_steps=500)
        opt = offline_fractional_knapsack(values, weights).value
        ratios.append(opt / threshold_knapsack(values, weights, L, U).value)
    for order in (np.argsort, lambda d: np.argsort(-d)):
        values, weights = random_instance(rng, L, U, 200, 3.0)
        idx = order(values / weights)
        ratios.append(offline_fractional_knapsack(values, weights).value
                      / threshold_knapsack(values[idx], weights[idx], L, U).value)
    assert max(ratios) <= bound * (1 + RTOL)


@pytest.mark.parametrize("L, U", RANGES)
def test_ratio_is_tight_on_the_ramp(L, U):
    bound = zcl_competitive_ratio(L, U)
    for p in [L, np.sqrt(L * U), U]:
        values, weights = ramp_instance(p, L, n_steps=4000)
        ratio = offline_fractional_knapsack(values, weights).value / threshold_knapsack(values, weights, L, U).value
        assert ratio == pytest.approx(bound, rel=1e-3)


def test_large_items_are_fine_in_the_fractional_version():
    """One item heavier than the knapsack: the rule takes the part that keeps the threshold below its density."""
    L, U = 1.0, 50.0
    for density in [L, 5.0, U]:
        result = threshold_knapsack(np.array([3.0 * density]), np.array([3.0]), L, U)
        assert result.value / (density * 1.0) <= zcl_competitive_ratio(L, U) * (1 + RTOL)
        assert result.fractions[0] * 3.0 == pytest.approx(zcl_utilization_at(density, L, U))


def test_greedy_baseline_can_be_theta_bad():
    for L, U in RANGES:
        values, weights = flood_instance(L, U)
        opt = offline_fractional_knapsack(values, weights).value
        assert opt / greedy_knapsack(values, weights).value == pytest.approx(U / L)
        ratio = opt / threshold_knapsack(values, weights, L, U).value
        assert ratio <= zcl_competitive_ratio(L, U) * (1 + RTOL)
