"""Tests for online/conversion.py against Sun et al. (NeurIPS 2021) Theorems 3.5, 3.6, 4.1 and 4.2."""

import numpy as np
import pytest

from online.conversion import (
    alpha_star,
    classical_one_way_threshold,
    classical_reservation_price,
    naive_one_max_search_bounds,
    naive_reservation_price,
    offline_optimum,
    one_max_search,
    one_max_search_bounds,
    one_way_trades,
    one_way_trading,
    one_way_trading_bounds,
    p_instance,
    pareto_one_way_threshold,
    pareto_reservation_price,
    random_instances,
    reservation_price_ratio,
    threshold_ratio,
)

RTOL = 1e-9
PRICE_RANGES = [(1.0, 1.5), (2.0, 10.0), (1.0, 20.0), (1.0, 200.0)]
LAMBDAS = [0.05, 0.2, 0.5, 0.8, 0.99]


def instances(rng, L, U, n_random=200):
    """Random walks plus the adversarial p-instances (Definition 4.3) for maximum prices across [L, U]."""
    adversarial = p_instance(np.linspace(L, U, 41), L, n_steps=400)
    random = random_instances(rng, L, U, n_random, n_steps=adversarial.shape[1])
    return np.vstack([adversarial, random])


def test_alpha_star_solves_its_fixed_point():
    for theta in [1.5, 5.0, 20.0, 1000.0]:
        a = alpha_star(theta)
        assert (a - 1.0) * np.exp(a) == pytest.approx(theta - 1.0, rel=1e-12)
        assert 1.0 < a <= np.sqrt(theta)  # fractional trading beats the best reservation price


def test_bounds_endpoints_and_pareto_relations():
    for L, U in PRICE_RANGES:
        theta = U / L
        assert one_max_search_bounds(1.0, theta) == pytest.approx((np.sqrt(theta), np.sqrt(theta)))
        assert one_max_search_bounds(0.0, theta) == pytest.approx((1.0, theta))
        assert one_way_trading_bounds(1.0, theta) == pytest.approx((alpha_star(theta),) * 2)
        assert one_way_trading_bounds(0.0, theta) == pytest.approx((1.0, theta))
        for lam in LAMBDAS:
            eta, gamma = one_max_search_bounds(lam, theta)
            assert eta == pytest.approx(theta / gamma)  # Theorem 4.1 lower bound, met with equality
            assert eta == pytest.approx(lam * gamma + 1.0 - lam)  # equation (5)
            eta_owt, gamma_owt = one_way_trading_bounds(lam, theta)
            assert 1.0 <= eta_owt <= gamma_owt <= theta
            # Figure 2 of the paper: at the same robustness gamma, one-way trading is more consistent.
            lam_same_gamma = 1.0 - (gamma - alpha_star(theta)) / (theta - alpha_star(theta))
            assert one_way_trading_bounds(lam_same_gamma, theta).consistency <= eta + 1e-12


def test_classical_algorithms_stay_within_their_optimal_ratios():
    rng = np.random.default_rng(0)
    for L, U in PRICE_RANGES:
        prices = instances(rng, L, U)
        opt = offline_optimum(prices)
        ratio_1max = opt / one_max_search(prices, classical_reservation_price(L, U))
        ratio_owt = opt / one_way_trading(prices, classical_one_way_threshold(L, U))
        assert ratio_1max.max() <= np.sqrt(U / L) * (1 + RTOL)
        assert ratio_owt.max() <= alpha_star(U / L) * (1 + RTOL)
        # The p-instances come close to the bound, so the ratios are tight.
        V = np.linspace(L, U, 401)
        assert threshold_ratio(V, classical_one_way_threshold(L, U)).max() == pytest.approx(alpha_star(U / L))
        assert ratio_owt.max() >= 0.98 * alpha_star(U / L)


def test_one_way_trades_are_feasible():
    rng = np.random.default_rng(1)
    prices = random_instances(rng, 1.0, 20.0, 100)
    for lam in LAMBDAS:
        trades = one_way_trades(prices, pareto_one_way_threshold(7.0, 1.0, 20.0, lam))
        assert trades.min() >= -1e-12
        assert np.allclose(trades.sum(axis=1), 1.0)


@pytest.mark.parametrize("L, U", PRICE_RANGES)
def test_pareto_one_max_search_consistency_and_robustness(L, U):
    rng = np.random.default_rng(2)
    prices = instances(rng, L, U)
    opt = offline_optimum(prices)
    V_grid = np.linspace(L, U, 801)
    for lam in LAMBDAS:
        eta, gamma = one_max_search_bounds(lam, U / L)
        worst_consistency = 0.0
        for P in np.linspace(L, U, 61):
            phi = pareto_reservation_price(P, L, U, lam)
            # Robustness: any instance, whatever the prediction.
            assert (opt / one_max_search(prices, phi)).max() <= gamma * (1 + RTOL)
            assert reservation_price_ratio(V_grid, phi, L).max() <= gamma * (1 + RTOL)
            # Consistency: instances whose maximum price is exactly P.
            exact = p_instance(P, L, n_steps=400)
            ratio = offline_optimum(exact) / one_max_search(exact, phi)
            assert ratio.max() <= eta * (1 + RTOL)
            worst_consistency = max(worst_consistency, float(reservation_price_ratio(P, phi, L)))
        assert worst_consistency == pytest.approx(eta, rel=1e-9)  # the bound is reached


@pytest.mark.parametrize("L, U", PRICE_RANGES)
def test_pareto_one_way_trading_consistency_and_robustness(L, U):
    rng = np.random.default_rng(3)
    prices = instances(rng, L, U)
    opt = offline_optimum(prices)
    V_grid = np.linspace(L, U, 801)
    for lam in LAMBDAS:
        eta, gamma = one_way_trading_bounds(lam, U / L)
        worst_consistency = worst_robustness = 0.0
        for P in np.linspace(L, U, 41):
            phi = pareto_one_way_threshold(P, L, U, lam)
            assert (opt / one_way_trading(prices, phi)).max() <= gamma * (1 + RTOL)
            worst_robustness = max(worst_robustness, threshold_ratio(V_grid, phi).max())
            exact = p_instance(P, L, n_steps=400)
            assert (offline_optimum(exact) / one_way_trading(exact, phi)).max() <= eta * (1 + RTOL)
            worst_consistency = max(worst_consistency, float(threshold_ratio(P, phi)))
        assert worst_consistency <= eta * (1 + RTOL)
        assert worst_robustness <= gamma * (1 + RTOL)
        assert worst_consistency == pytest.approx(eta, rel=1e-6)
        assert worst_robustness == pytest.approx(gamma, rel=1e-3)


def test_adversarial_prediction_drives_ratio_to_robustness():
    L, U = 2.0, 10.0
    for lam in [0.2, 0.5, 0.8]:
        eta, gamma = one_max_search_bounds(lam, U / L)
        # Prediction U while the market tops out just below the reservation price L gamma.
        phi = pareto_reservation_price(U, L, U, lam)
        bad = p_instance(phi * (1 - 1e-9), L, n_steps=400)
        assert offline_optimum(bad)[0] / one_max_search(bad, phi)[0] == pytest.approx(gamma, rel=1e-6)
        eta, gamma = one_way_trading_bounds(lam, U / L)
        phi = pareto_one_way_threshold(U, L, U, lam)
        V = np.linspace(L, U, 4001)
        ratio = offline_optimum(p_instance(V, L, 4000)) / one_way_trading(p_instance(V, L, 4000), phi)
        assert ratio.max() == pytest.approx(gamma, rel=1e-2)


def test_lambda_one_recovers_the_classical_algorithms():
    for L, U in PRICE_RANGES:
        w = np.linspace(0.0, 1.0, 501)
        v = np.linspace(L, U, 1001)
        classical = classical_one_way_threshold(L, U)
        gaps = {lam: 0.0 for lam in (0.99, 0.999, 0.9999)}
        for P in np.linspace(L, U, 25):
            assert pareto_reservation_price(P, L, U, 1.0) == pytest.approx(classical_reservation_price(L, U))
            assert np.allclose(pareto_one_way_threshold(P, L, U, 1.0)(w), classical(w))
            # The general construction of (8)-(10) converges to the classical schedule as lambda -> 1:
            # the utilization reached at each price approaches the classical one.
            for lam in gaps:
                schedule = pareto_one_way_threshold(P, L, U, lam).utilization_at(v)
                gaps[lam] = max(gaps[lam], np.max(np.abs(schedule - classical.utilization_at(v))))
        assert gaps[0.99] > gaps[0.999] > gaps[0.9999]
        assert gaps[0.9999] <= 0.02


def test_lambda_zero_follows_the_prediction():
    L, U = 2.0, 10.0
    for P in np.linspace(L, U, 25):
        assert pareto_reservation_price(P, L, U, 0.0) == pytest.approx(P)
        assert np.allclose(pareto_one_way_threshold(P, L, U, 0.0)(np.linspace(0, 1, 11)), P)


def test_naive_combination_does_not_improve_consistency():
    """Proposition 3.1: lambda sqrt(LU) + (1 - lambda) P stays sqrt(theta)-consistent for lambda > 0."""
    L, U = 2.0, 10.0
    for lam in [0.2, 0.5, 0.8]:
        consistency, robustness = naive_one_max_search_bounds(lam, U / L)
        worst = max(float(reservation_price_ratio(P, naive_reservation_price(P, L, U, lam), L))
                    for P in np.linspace(L, U, 2001))
        assert worst <= consistency * (1 + RTOL)
        assert worst == pytest.approx(np.sqrt(U / L), rel=1e-2)
        assert one_max_search_bounds(lam, U / L).consistency < consistency


def test_threshold_shapes_match_equation_8():
    """phi is non-decreasing in [L, U]; phi(1) = P when P gamma / eta >= U and U otherwise (proof of Thm 3.6)."""
    for L, U in PRICE_RANGES:
        for lam in LAMBDAS:
            eta, gamma = one_way_trading_bounds(lam, U / L)
            first_piece_8a = pareto_one_way_threshold(L, L, U, lam).pieces[0]  # P = L < M uses (8a)
            w = np.linspace(0, 1, 1001)
            for P in np.linspace(L, U, 41):
                phi = pareto_one_way_threshold(P, L, U, lam)
                values = phi(w)
                assert np.all(np.diff(values) >= -1e-9 * U)
                assert values[0] >= L * (1 - 1e-12) and values[-1] <= U * (1 + 1e-12)
                in_8b = len(phi.pieces) == 4
                expected_top = P if in_8b and P * gamma / eta >= U else U
                assert phi(1.0) == pytest.approx(expected_top, rel=1e-9)
                if in_8b:
                    starts = [piece.start for piece in phi.pieces] + [1.0]
                    assert np.all(np.diff(starts) >= -1e-12)  # 0 <= beta1 <= beta1' <= beta2 <= 1
            assert first_piece_8a.rate == pytest.approx(eta)
            assert first_piece_8a.value_at_start == pytest.approx(eta * L)
