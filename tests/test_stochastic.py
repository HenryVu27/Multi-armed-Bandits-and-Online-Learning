"""Tests for mab.stochastic: environments, ETC, UCB1, epsilon-greedy and the ETC regret bound."""

import math

import numpy as np
import pytest

from mab.stochastic import (
    BernoulliBandit,
    EpsilonGreedy,
    ExploreThenCommit,
    GaussianBandit,
    UCB1,
    average_regret,
    etc_regret_bound,
    etc_tuned_m,
    run,
    simulate,
    ucb_f,
)


def test_gaussian_arms_expose_means_gaps_and_unit_variance():
    env = GaussianBandit([0.2, 1.0, 0.5])
    np.testing.assert_allclose(env.gaps, [0.8, 0.0, 0.5])
    table = env.sample_rewards(40_000, np.random.default_rng(0))
    assert table.shape == (40_000, 3)
    np.testing.assert_allclose(table.mean(axis=0), env.means, atol=0.02)
    np.testing.assert_allclose(table.std(axis=0), 1.0, atol=0.02)


def test_bernoulli_arms_pay_zero_or_one_at_the_right_rate():
    env = BernoulliBandit([0.1, 0.7])
    table = env.sample_rewards(40_000, np.random.default_rng(1))
    assert set(np.unique(table)) == {0.0, 1.0}
    np.testing.assert_allclose(table.mean(axis=0), env.means, atol=0.01)
    with pytest.raises(ValueError):
        BernoulliBandit([0.5, 1.2])


def test_simulation_is_reproducible_and_policies_see_the_same_rewards():
    env = BernoulliBandit([0.3, 0.6, 0.5])
    first = simulate(env, EpsilonGreedy(0.1), 500, np.random.default_rng(7))
    second = simulate(env, EpsilonGreedy(0.1), 500, np.random.default_rng(7))
    np.testing.assert_array_equal(first, second)

    table = env.sample_rewards(500, np.random.default_rng(3))
    for policy in (ExploreThenCommit(m=20), UCB1()):
        arms, rewards = run(env, policy, 500, np.random.default_rng(3))
        np.testing.assert_array_equal(rewards, table[np.arange(500), arms])


def test_etc_explores_round_robin_then_commits_to_the_best_mean():
    env = GaussianBandit([0.0, 0.0, 3.0])
    m = 4
    arms, _ = run(env, ExploreThenCommit(m), 40, np.random.default_rng(0))
    rounds = np.arange(1, m * 3 + 1)
    np.testing.assert_array_equal(arms[: m * 3], rounds % 3)  # report: (t mod K) + 1 = i
    assert np.all(arms[m * 3 :] == 2)


def test_ucb1_pulls_each_arm_once_then_uses_the_report_index():
    policy = UCB1()
    policy.reset(3)
    rng = np.random.default_rng(0)
    for t, reward in zip(range(1, 4), [1.0, 0.0, 0.5]):
        arm = policy.select(t, rng)
        assert arm == t - 1
        policy.update(arm, reward)
    t = 4
    assert ucb_f(t) == pytest.approx(1 + t * math.log(t) ** 2)
    bonus = math.sqrt(2 * math.log(1 + t * math.log(t) ** 2) / 1)
    np.testing.assert_allclose(policy.index(t), np.array([1.0, 0.0, 0.5]) + bonus)
    assert policy.select(t, rng) == 0


def test_ucb1_regret_grows_sublinearly_on_an_easy_instance():
    env = GaussianBandit([1.0, 0.5, 0.0])
    mean, _ = average_regret(env, UCB1(), 4000, range(20))
    # Linear regret would give a ratio of 4 between rounds 4000 and 1000.
    assert mean[-1] / mean[999] < 1.6
    assert mean[-1] / 4000 < 0.02


def test_epsilon_greedy_regret_is_linear_with_slope_epsilon_times_mean_gap():
    env = GaussianBandit([1.0, 0.0, 0.0, 0.0])
    epsilon = 0.2
    mean, _ = average_regret(env, EpsilonGreedy(epsilon), 4000, range(20))
    slope = (mean[-1] - mean[1999]) / 2000
    assert slope == pytest.approx(epsilon * env.gaps.mean(), abs=0.02)


def test_etc_bound_matches_the_report_formula():
    means, m, n = [1.0, 0.5, 0.0], 10, 100
    expected = m * 1.5 + (n - m * 3) * (0.5 * math.exp(-m * 0.25 / 4) + 1.0 * math.exp(-m / 4))
    assert etc_regret_bound(means, m, n) == pytest.approx(expected)
    curve = etc_regret_bound(means, m, np.array([30, 100]))
    assert curve[0] == pytest.approx(m * 1.5) and curve[1] == pytest.approx(expected)
    with pytest.raises(ValueError):
        etc_regret_bound(means, m, 29)


def test_etc_tuned_m_follows_the_two_arm_formula():
    assert etc_tuned_m(0.2, 10_000) == math.ceil(4 / 0.04 * math.log(10_000 * 0.04 / 4))
    assert etc_tuned_m(0.01, 10) == 1


def test_etc_average_regret_stays_below_the_bound():
    env = GaussianBandit([0.5, 0.0, -0.2])
    m, n = 8, 2000
    mean, stderr = average_regret(env, ExploreThenCommit(m), n, range(200))
    rounds = np.arange(m * 3, n + 1)
    bound = etc_regret_bound(env.means, m, rounds)
    # At t = mK both sides equal m * sum(gaps); afterwards the bound should hold with room to spare.
    assert np.all(mean[rounds - 1] + 3 * stderr[rounds - 1] <= bound + 1e-9)
    assert mean[-1] + 3 * stderr[-1] < 0.5 * bound[-1]
