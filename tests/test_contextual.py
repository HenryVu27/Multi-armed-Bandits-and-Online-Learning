from pathlib import Path

import numpy as np

from mab.adversarial import Exp3, sample_categorical
from mab.contextual import (
    ContextFreeUCB,
    EpsilonGreedy,
    Exp4,
    LinUCB,
    NewsLog,
    PerContextExp3,
    load_news_log,
    make_advice_problem,
    play_advice_problem,
    replay_evaluate,
)

DATASET = Path(__file__).resolve().parents[1] / "project" / "dataset.txt"


def test_exp4_with_unit_vector_experts_is_exp3():
    n, K = 2000, 4
    rng = np.random.default_rng(0)
    table = rng.random((n, K))
    advice = np.eye(K)[None]  # (1, M = K, K): expert m always recommends arm m
    exp3, exp4 = Exp3(K, n), Exp4(K, K, n)
    assert exp4.eta == exp3.eta
    for t in range(n):
        p3 = exp3.probabilities()
        p4 = exp4.probabilities(None, advice)
        np.testing.assert_allclose(p4, p3, rtol=1e-10, atol=1e-15)
        arm = sample_categorical(p3, rng)
        reward = table[t, arm]
        exp3.update(arm, reward, p3)
        exp4.update(None, advice, arm, reward, p4)


def test_exp4_probabilities_are_q_times_e():
    rng = np.random.default_rng(1)
    advice = rng.dirichlet(np.ones(5), size=(3, 7))  # (runs, M, K)
    learner = Exp4(7, 5, 100, n_runs=3)
    learner.S = rng.normal(size=(3, 7)) * 50
    P = learner.probabilities(None, advice)
    Q = learner.expert_weights()
    np.testing.assert_allclose(P, np.stack([Q[r] @ advice[r] for r in range(3)]))
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert (P > 0).all()


def test_per_context_exp3_keeps_contexts_separate():
    learner = PerContextExp3(n_contexts=3, n_arms=4, n_runs=1)
    before = learner.probabilities(np.array([1]))
    for _ in range(3):
        p = learner.probabilities(np.array([0]))
        learner.update(np.array([0]), None, np.array([2]), np.array([0.0]), p)
    np.testing.assert_allclose(learner.probabilities(np.array([1])), before)
    assert learner.probabilities(np.array([0]))[0, 2] < 0.25


def test_exp4_regret_below_bound_on_advice_problem():
    n, C, K, M = 3000, 10, 4, 16
    problem = make_advice_problem(C, K, M, np.random.default_rng(2))
    run = play_advice_problem(Exp4(M, K, n, n_runs=32), problem, n, np.random.default_rng(3))
    bound = np.sqrt(2 * n * min(M, K) * np.log(M))
    assert run.regret_to_best_expert[:, -1].mean() < bound
    probs_ok = PerContextExp3(C, K, n_runs=32)
    pc = play_advice_problem(probs_ok, problem, n, np.random.default_rng(3))
    assert pc.regret_to_best_mapping.mean() < np.sqrt(2 * C * n * K * np.log(K))


def test_linucb_inverse_matches_direct_ridge_solution():
    rng = np.random.default_rng(4)
    learner = LinUCB(n_arms=3, dim=6, alpha=0.5)
    A = np.tile(np.eye(6), (3, 1, 1))
    for _ in range(200):
        context = rng.normal(size=(3, 6))
        arm = int(rng.integers(3))
        learner.update(arm, float(rng.random()), context)
        A[arm] += np.outer(context[arm], context[arm])
    np.testing.assert_allclose(learner.A_inv, np.linalg.inv(A), atol=1e-10)


def test_replay_of_fixed_arm_policy_recovers_logged_click_rate():
    log = load_news_log(DATASET)

    class Always:
        def select(self, context, rng):
            return 3

        def update(self, arm, reward, context):
            pass

    matched = replay_evaluate(Always(), log, np.random.default_rng(5))
    assert len(matched) == np.sum(log.arms == 3)
    assert matched.mean() == log.rewards[log.arms == 3].mean()


def test_linucb_replay_reproduces_notebook_number():
    log = load_news_log(DATASET)
    matched = replay_evaluate(LinUCB(10, 10, alpha=1.0), log, np.random.default_rng(6), max_matches=800)
    assert len(matched) == 800
    assert abs(matched.mean() - 0.53) < 0.01  # project/MABs.ipynb: 0.5300
    small = NewsLog(log.arms[:2000], log.rewards[:2000], log.contexts[:2000])
    for policy in (EpsilonGreedy(10, 0.1), ContextFreeUCB(10)):
        assert 0.0 <= replay_evaluate(policy, small, np.random.default_rng(7)).mean() <= 1.0
