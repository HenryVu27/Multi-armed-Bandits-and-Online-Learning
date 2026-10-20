import numpy as np
import pytest

from mab.adversarial import (
    Exp3,
    Exp3IX,
    Hedge,
    LeastLikelyAdversary,
    ObliviousAdversary,
    UCB1,
    exp3_bound,
    exp3ix_bound,
    hedge_bound,
    play,
    sample_categorical,
    switching_table,
)

N, K = 3000, 5


def stochastic_table(rng: np.random.Generator, n_runs: int) -> np.ndarray:
    means = np.array([0.5, 0.6, 0.4, 0.45, 0.55])
    return (rng.random((n_runs, N, K)) < means).astype(np.int8)


def test_sample_categorical_matches_probabilities():
    rng = np.random.default_rng(0)
    probs = np.tile([0.0, 0.2, 0.5, 0.0, 0.3], (200_000, 1))
    counts = np.bincount(sample_categorical(probs, rng), minlength=5) / len(probs)
    assert counts[0] == 0 and counts[3] == 0
    np.testing.assert_allclose(counts, probs[0], atol=0.005)


@pytest.mark.parametrize(
    "make", [lambda: Exp3(K, N, n_runs=4), lambda: Exp3IX(K, N, n_runs=4), lambda: Hedge(K, N, n_runs=4)]
)
def test_probabilities_sum_to_one_and_stay_positive(make):
    table = switching_table(N, K, np.random.default_rng(1), n_runs=4)
    run = play(make(), ObliviousAdversary(table), N, np.random.default_rng(2), record_probs=True)
    np.testing.assert_allclose(run.probs.sum(axis=2), 1.0, atol=1e-12)
    assert (run.probs > 0).all()


def test_exp3_follows_report_algorithm_3():
    rng = np.random.default_rng(3)
    table = rng.random((50, K))
    eta = np.sqrt(2 * np.log(K) / (50 * K))
    learner, S_hat = Exp3(K, 50), np.zeros(K)
    for t in range(50):
        P = np.exp(eta * S_hat) / np.exp(eta * S_hat).sum()
        np.testing.assert_allclose(learner.probabilities()[0], P, rtol=1e-12)
        arm = int(rng.choice(K, p=P))
        X = table[t, arm]
        S_hat = S_hat + 1 - (np.arange(K) == arm) * (1 - X) / P
        learner.update(np.array([arm]), np.array([X]), P[None])


@pytest.mark.parametrize("table_kind", ["stochastic", "switching"])
def test_exp3_mean_regret_below_report_bound(table_kind):
    rng = np.random.default_rng(4)
    if table_kind == "stochastic":
        table = stochastic_table(rng, 64)
    else:
        table = switching_table(N, K, rng, n_runs=64)
    run = play(Exp3(K, N, n_runs=64), ObliviousAdversary(table), N, np.random.default_rng(5))
    assert run.regret[:, -1].mean() < exp3_bound(N, K)
    assert run.expected_regret.mean() < exp3_bound(N, K)


def test_hedge_expected_regret_below_bound_on_every_table():
    rng = np.random.default_rng(6)
    alternating = np.zeros((N, K))
    alternating[::2, 0] = 1.0
    alternating[1::2, 1] = 1.0
    tables = [stochastic_table(rng, 8), switching_table(N, K, rng, n_runs=8), rng.random((N, K))]
    for table in tables + [alternating]:
        run = play(Hedge(K, N, n_runs=8), ObliviousAdversary(table), N, np.random.default_rng(7))
        assert (run.expected_regret <= hedge_bound(N, K)).all()


def test_exp3ix_high_probability_bound():
    table = switching_table(N, K, np.random.default_rng(8), n_runs=200)
    run = play(Exp3IX(K, N, n_runs=200), ObliviousAdversary(table), N, np.random.default_rng(9))
    delta = 0.05
    assert np.mean(run.regret[:, -1] >= exp3ix_bound(N, K, delta)) <= delta


def test_adaptive_adversary_gives_ucb_linear_regret_but_not_exp3():
    ucb = play(UCB1(K, n_runs=2), LeastLikelyAdversary(), N, np.random.default_rng(10))
    assert (ucb.learner_total == 0).all()
    assert (ucb.regret[:, -1] >= N / K).all()
    exp3 = play(Exp3(K, N, n_runs=32), LeastLikelyAdversary(), N, np.random.default_rng(11))
    assert exp3.regret[:, -1].mean() < exp3_bound(N, K)


def test_ucb1_finds_best_arm_on_stochastic_table():
    table = stochastic_table(np.random.default_rng(12), 8)
    run = play(UCB1(K, n_runs=8), ObliviousAdversary(table), N, np.random.default_rng(13))
    assert run.regret[:, -1].mean() < 0.05 * N
