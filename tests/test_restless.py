import itertools

import numpy as np

from mab.restless import (
    RestlessArm,
    average_reward,
    indexable,
    is_indexable,
    myopic_policy,
    random_policy,
    random_two_state_arm,
    relaxation_upper_bound,
    simulate,
    subsidy_q_values,
    two_state_arm,
    whittle_index,
    whittle_indices,
    whittle_policy,
)

GAMMA = 0.9


def exact_q_gap(arm: RestlessArm, lam: float, gamma: float) -> np.ndarray:
    """Q_passive - Q_active from exact evaluation of every deterministic policy."""
    n = len(arm.reward)
    best = None
    for policy in itertools.product([False, True], repeat=n):
        active = np.array(policy)
        transitions = np.where(active[:, None], arm.active, arm.passive)
        rewards = np.where(active, arm.reward, lam)
        values = np.linalg.solve(np.eye(n) - gamma * transitions, rewards)
        best = values if best is None else np.maximum(best, values)
    return lam + gamma * arm.passive @ best - (arm.reward + gamma * arm.active @ best)


def test_value_iteration_matches_exact_policy_evaluation():
    arm = random_two_state_arm(np.random.default_rng(0))
    for lam in (-0.5, 0.2, 0.7):
        q_active, q_passive, _ = subsidy_q_values([arm], np.array([[lam]]), GAMMA)
        np.testing.assert_allclose(q_passive[0, 0] - q_active[0, 0], exact_q_gap(arm, lam, GAMMA), atol=1e-8)


def test_index_equals_reward_when_action_does_not_change_dynamics():
    rng = np.random.default_rng(1)
    transitions = rng.dirichlet(np.ones(3), size=3)
    reward = np.array([0.3, -0.2, 0.9])
    arm = RestlessArm(transitions, transitions, reward)
    np.testing.assert_allclose(whittle_index(arm, GAMMA), reward, atol=1e-8)


def test_frozen_passive_arm_matches_gittins_closed_form():
    # Resting freezes the state, so the Whittle index is the Gittins index. With active
    # transitions P(0 -> 1) = p and P(1 -> 0) = q and rewards (0, 1):
    # W(1) = 1 and W(0) = gamma p / (1 - gamma (1 - p - q)).
    for p, q in [(0.3, 0.4), (0.1, 0.9), (0.8, 0.05)]:
        arm = two_state_arm(active_to_1=(p, 1 - q), passive_to_1=(0.0, 1.0))
        expected = [GAMMA * p / (1 - GAMMA * (1 - p - q)), 1.0]
        np.testing.assert_allclose(whittle_index(arm, GAMMA), expected, atol=1e-8)
        assert is_indexable(arm, GAMMA)


def test_fresh_state_index_is_one_for_every_two_state_arm():
    # For a subsidy below 1, pulling in state 1 always beats resting, so W(1) = r(1) = 1.
    rng = np.random.default_rng(2)
    arms = [random_two_state_arm(rng) for _ in range(40)]
    index = whittle_indices(arms, GAMMA)
    np.testing.assert_allclose(index[:, 1], 1.0, atol=1e-8)
    assert (index[:, 0] < 1.0).all()


def test_index_is_monotone_in_the_state_reward():
    base = random_two_state_arm(np.random.default_rng(3))
    for state in (0, 1):
        levels = np.linspace(-0.5, 1.5, 9)
        arms = []
        for level in levels:
            reward = base.reward.copy()
            reward[state] = level
            arms.append(RestlessArm(base.active, base.passive, reward))
        index = whittle_indices(arms, GAMMA)[:, state]
        assert np.all(np.diff(index) > 0)


def test_indexability_check():
    rng = np.random.default_rng(4)
    assert indexable([random_two_state_arm(rng) for _ in range(20)], GAMMA).all()
    # A three-state arm whose state 0 enters the passive set near lambda = -0.62, leaves near
    # 0.50 and re-enters at 0.79 (confirmed with exact_q_gap), so it is not indexable.
    arm = RestlessArm(
        active=np.array([[0.0, 0.99, 0.01], [0.84, 0.16, 0.0], [0.02, 0.0, 0.98]]),
        passive=np.array([[0.0, 0.2, 0.8], [0.06, 0.87, 0.07], [0.96, 0.01, 0.03]]),
        reward=np.array([0.79, 0.01, 0.58]),
    )
    assert exact_q_gap(arm, 0.0, GAMMA)[0] > 0 and exact_q_gap(arm, 0.6, GAMMA)[0] < -0.05
    assert not is_indexable(arm, GAMMA)


def test_policies_respect_budget_and_bound():
    rng = np.random.default_rng(5)
    arms = [random_two_state_arm(rng) for _ in range(6)]
    bound = relaxation_upper_bound(arms, n_active=2)
    results = {}
    for name, policy in [
        ("whittle", whittle_policy(arms, 0.99)),
        ("myopic", myopic_policy(arms)),
        ("random", random_policy()),
    ]:
        totals = simulate(arms, policy, 2, 2000, np.random.default_rng(6), n_runs=4)
        assert totals.max() <= 2
        results[name] = average_reward(totals, burn_in=200).mean()
    assert results["whittle"] <= bound + 0.02
    assert results["whittle"] > results["random"]
