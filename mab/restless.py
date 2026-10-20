"""Restless bandits: Markov arms with active and passive dynamics, the Whittle index, index policies.

Sources: Henry Vu, Final Report, Section 4 (4.2 relaxed constraints, 4.3 decoupled problem);
slides "Whittle Index and Restless Bandits" and "Markov Process and Gittins Index"; Whittle,
"Restless bandits: activity allocation in a changing world" (Journal of Applied Probability,
1988); Weber and Weiss, "On an index policy for restless bandits" (1990).

The report's decoupled problem charges lambda for every active round (r - lambda u). This module
pays lambda as a subsidy for every passive round instead, as Whittle did. The two objectives
differ by the constant lambda per round, so they share optimal policies, passive sets P(lambda)
and indices. Each arm pays ``reward[s]`` when pulled in state s and nothing when rested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class RestlessArm:
    active: np.ndarray  # (S, S) transition matrix when pulled
    passive: np.ndarray  # (S, S) transition matrix when rested
    reward: np.ndarray  # (S,) reward for pulling in each state


def two_state_arm(
    active_to_1: tuple[float, float], passive_to_1: tuple[float, float]
) -> RestlessArm:
    """Arm on states {0, 1} that pays its state when pulled. ``active_to_1[s]`` is
    P(next state = 1 | state s, pulled); ``passive_to_1[s]`` is the same when rested."""

    def matrix(to_1: tuple[float, float]) -> np.ndarray:
        p = np.asarray(to_1, dtype=float)
        return np.stack([1 - p, p], axis=1)

    return RestlessArm(matrix(active_to_1), matrix(passive_to_1), np.array([0.0, 1.0]))


def random_two_state_arm(rng: np.random.Generator, low: float = 0.05, high: float = 0.95) -> RestlessArm:
    """Draw the four transition probabilities of a two-state arm uniformly from [low, high]."""
    p = rng.uniform(low, high, size=4)
    return two_state_arm((p[0], p[1]), (p[2], p[3]))


def random_fatigue_arm(rng: np.random.Generator, low: float = 0.05, high: float = 0.95) -> RestlessArm:
    """Work tires, rest recovers (slides, "Restless Bandits"). The passive probabilities of moving
    to state 1 are uniform on [low, high]; pulling multiplies each by an independent U(0, 1)
    factor, so pulling never makes the good state more likely."""
    passive = rng.uniform(low, high, size=2)
    active = passive * rng.uniform(0.0, 1.0, size=2)
    return two_state_arm((active[0], active[1]), (passive[0], passive[1]))


# ---------------------------------------------------------------------------------------------
# Single-arm subsidy problem
# ---------------------------------------------------------------------------------------------


def _stack(arms: Sequence[RestlessArm]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([arm.active for arm in arms]),
        np.stack([arm.passive for arm in arms]),
        np.stack([arm.reward for arm in arms]),
    )


def subsidy_q_values(
    arms: Sequence[RestlessArm],
    subsidies: np.ndarray,
    gamma: float,
    tol: float = 1e-10,
    initial_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Value iteration on V(s) = max(r(s) + gamma P1[s] V, lambda + gamma P0[s] V).

    ``subsidies`` has shape (N, L): L subsidy levels per arm, all solved together. Returns
    Q_active, Q_passive and V, each of shape (N, L, S). ``initial_values`` warm-starts V.
    """
    active, passive, reward = _stack(arms)
    active_t, passive_t = np.swapaxes(active, 1, 2), np.swapaxes(passive, 1, 2)
    reward = reward[:, None, :]
    lam = np.asarray(subsidies, dtype=float)[..., None]
    values = np.zeros(lam.shape[:2] + reward.shape[2:]) if initial_values is None else initial_values
    while True:
        q_active = reward + gamma * values @ active_t
        q_passive = lam + gamma * values @ passive_t
        new_values = np.maximum(q_active, q_passive)
        if np.abs(new_values - values).max() < tol:
            return q_active, q_passive, new_values
        values = new_values


def passive_sets(arms: Sequence[RestlessArm], subsidies: np.ndarray, gamma: float) -> np.ndarray:
    """Boolean (N, L, S): s is in P(lambda) when resting is optimal (ties count as passive)."""
    q_active, q_passive, _ = subsidy_q_values(arms, subsidies, gamma)
    return q_passive >= q_active


def _subsidy_bracket(arms: Sequence[RestlessArm], gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Per arm: below lo every state is active, above hi every state is passive."""
    reward = np.stack([arm.reward for arm in arms])
    r_min, r_max = reward.min(axis=1), reward.max(axis=1)
    return r_min - gamma * (r_max - r_min) / (1 - gamma) - 1.0, r_max + 1.0


def whittle_indices(arms: Sequence[RestlessArm], gamma: float, iters: int = 50) -> np.ndarray:
    """W(s) for every arm and state, shape (N, S): the smallest passive subsidy at which resting
    is optimal in state s, found by bisection on lambda. Valid for indexable arms."""
    n_states = len(arms[0].reward)
    lo, hi = _subsidy_bracket(arms, gamma)
    lo = np.repeat(lo[:, None], n_states, axis=1)
    hi = np.repeat(hi[:, None], n_states, axis=1)
    states = np.arange(n_states)
    values = None
    for _ in range(iters):
        mid = (lo + hi) / 2  # problem (n, s) has subsidy mid[n, s]
        q_active, q_passive, values = subsidy_q_values(arms, mid, gamma, initial_values=values)
        passive = (q_passive >= q_active)[:, states, states]
        hi = np.where(passive, mid, hi)
        lo = np.where(passive, lo, mid)
    return (lo + hi) / 2


def whittle_index(arm: RestlessArm, gamma: float) -> np.ndarray:
    return whittle_indices([arm], gamma)[0]


def indexable(arms: Sequence[RestlessArm], gamma: float, grid_size: int = 401) -> np.ndarray:
    """Indexability (slides, "Indexability"): P(lambda) grows monotonically from the empty set
    to the whole state space as lambda increases. Checked per arm on a grid that also straddles
    each computed index, so a violation narrower than the grid spacing could be missed."""
    lo, hi = _subsidy_bracket(arms, gamma)
    index = whittle_indices(arms, gamma)
    grid = np.linspace(lo, hi, grid_size, axis=1)
    grid = np.sort(np.concatenate([grid, index - 1e-6, index + 1e-6], axis=1), axis=1)
    sets = passive_sets(arms, grid, gamma)  # (N, L, S)
    nested = ~np.any(sets[:, :-1] & ~sets[:, 1:], axis=(1, 2))
    return nested & ~sets[:, 0].any(axis=1) & sets[:, -1].all(axis=1)


def is_indexable(arm: RestlessArm, gamma: float) -> bool:
    return bool(indexable([arm], gamma)[0])


# ---------------------------------------------------------------------------------------------
# Relaxation bound (report Section 4.2) for the long-run average reward
# ---------------------------------------------------------------------------------------------


def _stationary(transitions: np.ndarray) -> np.ndarray:
    n = len(transitions)
    system = np.vstack([transitions.T - np.eye(n), np.ones(n)])
    target = np.zeros(n + 1)
    target[-1] = 1.0
    return np.linalg.lstsq(system, target, rcond=None)[0]


def _policy_lines(arm: RestlessArm) -> np.ndarray:
    """For each deterministic stationary policy of the subsidy problem, (a, b) such that its
    long-run average reward is a + b * lambda; b is the fraction of rounds spent resting."""
    n_states = len(arm.reward)
    lines = []
    for code in range(2**n_states):
        active = np.array([(code >> s) & 1 for s in range(n_states)], dtype=bool)
        transitions = np.where(active[:, None], arm.active, arm.passive)
        mu = _stationary(transitions)
        lines.append((mu @ (arm.reward * active), mu @ ~active))
    return np.array(lines)


def relaxation_upper_bound(arms: Sequence[RestlessArm], n_active: int) -> float:
    """Whittle's Lagrangian bound on the long-run average reward of any policy that pulls
    ``n_active`` arms per round: min over lambda of sum_i g_i(lambda) - lambda (N - n_active),
    where g_i(lambda) is the best average reward of arm i alone under subsidy lambda.

    Each g_i is a maximum of lines, so the minimum sits where two lines of one arm cross.
    Assumes every policy's chain has a single recurrent class.
    """
    lines = [_policy_lines(arm) for arm in arms]
    candidates = []
    for arm_lines in lines:
        for i in range(len(arm_lines)):
            for j in range(i + 1, len(arm_lines)):
                (a1, b1), (a2, b2) = arm_lines[i], arm_lines[j]
                if abs(b1 - b2) > 1e-12:
                    candidates.append((a2 - a1) / (b1 - b2))
    lam = np.array(candidates)
    total = sum((arm_lines[:, :1] + arm_lines[:, 1:] * lam).max(axis=0) for arm_lines in lines)
    return float((total - lam * (len(arms) - n_active)).min())


# ---------------------------------------------------------------------------------------------
# Policies and simulator
# ---------------------------------------------------------------------------------------------

Priority = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def whittle_policy(arms: Sequence[RestlessArm], gamma: float) -> Priority:
    """Pull the arms whose current state has the largest Whittle index."""
    table = whittle_indices(arms, gamma)  # (N, S)
    return lambda states, rng: table[np.arange(len(arms)), states]


def myopic_policy(arms: Sequence[RestlessArm]) -> Priority:
    """Pull the arms with the largest expected immediate reward, which is the state itself here."""
    table = np.stack([arm.reward for arm in arms])
    return lambda states, rng: table[np.arange(len(arms)), states]


def random_policy() -> Priority:
    return lambda states, rng: rng.random(states.shape)


def simulate(
    arms: Sequence[RestlessArm],
    policy: Priority,
    n_active: int,
    horizon: int,
    rng: np.random.Generator,
    n_runs: int = 1,
) -> np.ndarray:
    """Pull the ``n_active`` highest-priority arms each round (ties broken at random).

    Initial states are drawn uniformly. Returns the total reward per round, shape (n_runs, horizon).
    """
    n_arms = len(arms)
    n_states = len(arms[0].reward)
    active, passive, reward = _stack(arms)
    arm_ids = np.arange(n_arms)
    states = rng.integers(n_states, size=(n_runs, n_arms))
    totals = np.zeros((n_runs, horizon))
    for t in range(horizon):
        priority = policy(states, rng) + 1e-9 * rng.random((n_runs, n_arms))
        chosen = np.argpartition(-priority, n_active - 1, axis=1)[:, :n_active]
        pulled = np.zeros((n_runs, n_arms), dtype=bool)
        np.put_along_axis(pulled, chosen, True, axis=1)
        totals[:, t] = np.where(pulled, reward[arm_ids, states], 0.0).sum(axis=1)
        rows = np.where(pulled[..., None], active[arm_ids, states], passive[arm_ids, states])
        cdf = np.cumsum(rows, axis=-1)
        u = rng.random((n_runs, n_arms, 1)) * cdf[..., -1:]
        states = np.minimum((cdf <= u).sum(axis=-1), n_states - 1)
    return totals


def average_reward(totals: np.ndarray, burn_in: int = 0) -> np.ndarray:
    """Long-run average reward per round for each run, skipping the first ``burn_in`` rounds."""
    return totals[:, burn_in:].mean(axis=1)


def discounted_return(totals: np.ndarray, gamma: float) -> np.ndarray:
    return totals @ gamma ** np.arange(totals.shape[1])
