"""Tests for mab.bayesian: Beta posterior updates, Thompson sampling, Bayesian greedy and the
Gittins index."""

from functools import lru_cache

import numpy as np
import pytest
from scipy import stats

from mab.bayesian import (
    BayesianGreedy,
    GittinsPolicy,
    ThompsonSampling,
    calibration_value,
    gittins_index,
)
from mab.stochastic import BernoulliBandit, average_regret

# R. Weber, Cambridge lecture slides "Gittins Indices for Bernoulli Bandits, beta = 0.9"; the
# rows and columns are the Beta parameters (alpha, beta), so (1, 1) is the uniform prior.
WEBER_GAMMA_09 = {
    (1, 1): 0.7029,
    (2, 1): 0.8001,
    (8, 1): 0.9221,
    (1, 2): 0.5001,
    (2, 3): 0.5163,
    (6, 7): 0.5156,
    (1, 8): 0.1591,
    (8, 8): 0.5454,
}

# S. Baas, R. Boucherie and A. Braaksma, "A sampling-based Gittins index approximation" (2023),
# Table 1, calibration column (gamma = 0.8, truncation error at most 0.0005).
BAAS_GAMMA_08 = {(1, 1): 0.641, (1, 6): 0.183, (3, 3): 0.566, (6, 1): 0.888, (6, 6): 0.537}


def test_beta_posterior_update_counts_successes_and_failures():
    policy = BayesianGreedy(prior_alpha=1.0, prior_beta=1.0)
    policy.reset(2)
    for reward in [1.0, 1.0, 0.0, 1.0, 0.0]:
        policy.update(0, reward)
    assert (policy.alpha[0], policy.beta[0]) == (4.0, 3.0)
    assert (policy.alpha[1], policy.beta[1]) == (1.0, 1.0)
    assert policy.posterior_means[0] == pytest.approx(4 / 7)


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_beta_update_is_bayes_rule(reward):
    alpha, beta = 2.0, 3.0
    theta = np.linspace(0.001, 0.999, 999)
    unnormalised = theta**reward * (1 - theta) ** (1 - reward) * stats.beta.pdf(theta, alpha, beta)
    posterior = unnormalised / np.trapezoid(unnormalised, theta)
    policy = ThompsonSampling(alpha, beta)
    policy.reset(1)
    policy.update(0, reward)
    expected = stats.beta.pdf(theta, policy.alpha[0], policy.beta[0])
    np.testing.assert_allclose(posterior, expected, rtol=1e-3, atol=1e-3)


def test_thompson_sampling_pulls_the_argmax_of_posterior_samples():
    policy = ThompsonSampling()
    policy.reset(3)
    policy.alpha[:] = [2.0, 9.0, 1.0]
    policy.beta[:] = [5.0, 3.0, 1.0]
    for seed in range(20):
        samples = np.random.default_rng(seed).beta(policy.alpha, policy.beta)
        assert policy.select(1, np.random.default_rng(seed)) == int(np.argmax(samples))


def test_bayesian_greedy_pulls_the_largest_posterior_mean():
    policy = BayesianGreedy()
    policy.reset(3)
    policy.alpha[:] = [2.0, 5.0, 1.0]
    policy.beta[:] = [2.0, 1.0, 1.0]
    assert policy.select(1, np.random.default_rng(0)) == 1


def test_thompson_sampling_regret_is_small_and_sublinear():
    env = BernoulliBandit([0.9, 0.5, 0.2])
    mean, _ = average_regret(env, ThompsonSampling(), 2000, range(30))
    assert mean[-1] < 30
    assert mean[-1] / mean[499] < 2.0


@pytest.mark.parametrize(("state", "value"), WEBER_GAMMA_09.items())
def test_gittins_index_matches_the_published_gamma_09_table(state, value):
    assert gittins_index(*map(float, state), 0.9) == pytest.approx(value, abs=1e-4)


@pytest.mark.parametrize(("state", "value"), BAAS_GAMMA_08.items())
def test_gittins_index_matches_the_published_gamma_08_table(state, value):
    assert gittins_index(*map(float, state), 0.8) == pytest.approx(value, abs=1e-3)


def test_gittins_index_is_at_least_the_posterior_mean():
    for alpha in (1.0, 2.0, 5.0, 20.0):
        for beta in (1.0, 3.0, 10.0):
            assert gittins_index(alpha, beta, 0.9) >= alpha / (alpha + beta)


def test_gittins_index_decreases_in_beta_and_increases_in_alpha():
    by_beta = [gittins_index(2.0, b, 0.9) for b in (1.0, 2.0, 4.0, 8.0)]
    by_alpha = [gittins_index(a, 2.0, 0.9) for a in (1.0, 2.0, 4.0, 8.0)]
    assert np.all(np.diff(by_beta) < 0)
    assert np.all(np.diff(by_alpha) > 0)


def test_gittins_index_increases_with_gamma_and_tends_to_the_mean_as_gamma_vanishes():
    by_gamma = [gittins_index(1.0, 1.0, g) for g in (0.5, 0.7, 0.9, 0.95)]
    assert np.all(np.diff(by_gamma) > 0)
    assert gittins_index(3.0, 2.0, 0.01) == pytest.approx(0.6, abs=0.01)


def test_calibration_value_matches_the_report_recursion():
    gamma, depth = 0.9, 12

    def reference(p: float, alpha: float, beta: float) -> float:
        @lru_cache(maxsize=None)
        def value(a: float, b: float, level: int) -> float:
            mean = a / (a + b)
            if level == depth:
                return max(p, mean) / (1 - gamma)
            pull = mean * (1 + gamma * value(a + 1, b, level + 1)) + (1 - mean) * gamma * value(
                a, b + 1, level + 1
            )
            return max(p / (1 - gamma), pull)

        return value(alpha, beta, 0)

    for p in (0.3, 0.6, 0.7, 0.9):
        assert calibration_value(p, 1.0, 1.0, gamma, depth) == pytest.approx(reference(p, 1.0, 1.0))
        assert calibration_value(p, 3.0, 2.0, gamma, depth) == pytest.approx(reference(p, 3.0, 2.0))


def test_retiring_and_pulling_are_equally_good_at_the_index():
    gamma = 0.9
    index = gittins_index(2.0, 3.0, gamma)
    assert calibration_value(index, 2.0, 3.0, gamma) == pytest.approx(index / (1 - gamma), rel=1e-5)
    assert calibration_value(index - 0.01, 2.0, 3.0, gamma) > (index - 0.01) / (1 - gamma)


def test_gittins_policy_prefers_the_less_explored_arm_at_equal_means():
    policy = GittinsPolicy(gamma=0.9)
    policy.reset(2)
    policy.alpha[:] = [20.0, 1.0]
    policy.beta[:] = [20.0, 1.0]
    assert policy.select(1, np.random.default_rng(0)) == 1
