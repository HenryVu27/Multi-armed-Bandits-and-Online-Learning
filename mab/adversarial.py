"""Adversarial bandits: reward-table adversaries, Hedge, Exp3, Exp3-IX and a UCB1 baseline.

Sources: Henry Vu, Final Report, Section 1 (Algorithm 1 Hedge, Algorithm 2 setup,
Algorithm 3 Exp3 and the Exp3-IX theorem); slides "Adversarial Bandits, Exp3 Algorithm";
Lattimore and Szepesvari, Bandit Algorithms (2020), Chapters 11 and 12; Cesa-Bianchi and
Lugosi, Prediction, Learning, and Games (2006), Chapter 2; Neu, "Explore no more" (NeurIPS 2015).

Every learner runs ``n_runs`` independent copies side by side, so its state has a leading run
axis and ``probabilities()`` returns an array of shape (n_runs, K). Rewards lie in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax, shifted by the row maximum for numerical stability."""
    z = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw one index per row of ``probs`` by inverting the cumulative distribution."""
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random(probs.shape[:-1] + (1,)) * cdf[..., -1:]
    return np.minimum((cdf <= u).sum(axis=-1), probs.shape[-1] - 1)


def one_hot(indices: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(indices), n))
    out[np.arange(len(indices)), indices] = 1.0
    return out


# ---------------------------------------------------------------------------------------------
# Adversaries
# ---------------------------------------------------------------------------------------------


class Adversary(Protocol):
    def reward_vectors(self, t: int, probs: np.ndarray) -> np.ndarray:
        """Reward vector x_t for every run, shape (n_runs, K), chosen before A_t is drawn."""
        ...


@dataclass
class ObliviousAdversary:
    """A reward table fixed before the game: shape (n, K), or (n_runs, n, K) for one per run."""

    table: np.ndarray

    def reward_vectors(self, t: int, probs: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.table[..., t, :], probs.shape).astype(float)


class LeastLikelyAdversary:
    """Adaptive adversary: reward 1 to the arm the learner is least likely to pull, 0 elsewhere.

    It sees P_t before A_t is drawn (Algorithm 2) and breaks ties toward the lowest index.
    Against a deterministic learner such as UCB1, P_t is a point mass, so the rewarded arm is
    never pulled: the learner earns 0 while some arm earns at least n / K, which is linear
    regret. For a deterministic learner the same table could be written down in advance by
    simulating it, so obliviousness alone does not save it (Lattimore and Szepesvari, 11.1).
    """

    def reward_vectors(self, t: int, probs: np.ndarray) -> np.ndarray:
        return one_hot(probs.argmin(axis=1), probs.shape[1])


def switching_table(
    horizon: int,
    n_arms: int,
    rng: np.random.Generator,
    n_runs: int = 1,
    phase_ends: tuple[float, ...] = (0.2, 0.5, 1.0),
    good_arms: tuple[int, ...] = (0, 1, 2),
    high: float = 0.8,
    low: float = 0.2,
) -> np.ndarray:
    """Oblivious Bernoulli table whose best arm switches between phases.

    Phase j covers rounds up to ``phase_ends[j] * horizon``; its good arm ``good_arms[j]`` pays
    Bernoulli(high) and every other arm pays Bernoulli(low). Returns shape (n_runs, n, K), int8.
    """
    means = np.full((horizon, n_arms), low)
    start = 0
    for end_fraction, arm in zip(phase_ends, good_arms):
        end = int(round(end_fraction * horizon))
        means[start:end, arm] = high
        start = end
    return (rng.random((n_runs, horizon, n_arms)) < means).astype(np.int8)


# ---------------------------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------------------------


class Hedge:
    """Report Algorithm 1: full-information exponential weights on losses 1 - x_t."""

    full_information = True

    def __init__(self, n_experts: int, horizon: int, epsilon: float | None = None, n_runs: int = 1):
        self.epsilon = np.sqrt(8 * np.log(n_experts) / horizon) if epsilon is None else epsilon
        self.log_weights = np.zeros((n_runs, n_experts))  # log W_t(i), W_1(i) = 1

    def probabilities(self) -> np.ndarray:
        return softmax(self.log_weights)

    def update_full(self, reward_vectors: np.ndarray) -> None:
        self.log_weights -= self.epsilon * (1.0 - reward_vectors)


class Exp3:
    """Report Algorithm 3: exponential weights on the loss-based importance-weighted estimate.

    S_t,i = S_t-1,i + 1 - 1{A_t = i}(1 - X_t) / P_t,i, and P_t = softmax(eta S_t-1).
    """

    full_information = False

    def __init__(self, n_arms: int, horizon: int, eta: float | None = None, n_runs: int = 1):
        self.eta = np.sqrt(2 * np.log(n_arms) / (horizon * n_arms)) if eta is None else eta
        self.S = np.zeros((n_runs, n_arms))

    def probabilities(self) -> np.ndarray:
        return softmax(self.eta * self.S)

    def update(self, arms: np.ndarray, rewards: np.ndarray, probs: np.ndarray) -> None:
        rows = np.arange(len(arms))
        self.S += 1.0
        self.S[rows, arms] -= (1.0 - rewards) / probs[rows, arms]


class Exp3IX(Exp3):
    """Exp3 with implicit exploration: the played arm's loss is divided by P_t,i + gamma.

    Defaults follow the report's first Exp3-IX statement: eta_1 = sqrt(2 log(K+1) / (nK)) and
    gamma = eta / 2. The estimate is biased low on losses, which caps how far one bad draw can
    push an arm's score down at 1 / gamma.
    """

    def __init__(
        self,
        n_arms: int,
        horizon: int,
        eta: float | None = None,
        gamma: float | None = None,
        n_runs: int = 1,
    ):
        eta = np.sqrt(2 * np.log(n_arms + 1) / (horizon * n_arms)) if eta is None else eta
        super().__init__(n_arms, horizon, eta=eta, n_runs=n_runs)
        self.gamma = eta / 2 if gamma is None else gamma

    def update(self, arms: np.ndarray, rewards: np.ndarray, probs: np.ndarray) -> None:
        rows = np.arange(len(arms))
        self.S += 1.0
        self.S[rows, arms] -= (1.0 - rewards) / (probs[rows, arms] + self.gamma)


class UCB1:
    """UCB1 (Auer et al. 2002): mean + sqrt(2 log t / T_i), each arm tried once first.

    Ties go to the lowest index, so the policy is deterministic and P_t is a point mass.
    """

    full_information = False

    def __init__(self, n_arms: int, n_runs: int = 1):
        self.counts = np.zeros((n_runs, n_arms))
        self.sums = np.zeros((n_runs, n_arms))
        self.t = 0

    def probabilities(self) -> np.ndarray:
        t = self.t + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            index = self.sums / self.counts + np.sqrt(2 * np.log(t) / self.counts)
        index[self.counts == 0] = np.inf
        return one_hot(index.argmax(axis=1), self.counts.shape[1])

    def update(self, arms: np.ndarray, rewards: np.ndarray, probs: np.ndarray) -> None:
        rows = np.arange(len(arms))
        self.counts[rows, arms] += 1
        self.sums[rows, arms] += rewards
        self.t += 1


# ---------------------------------------------------------------------------------------------
# Game loop and regret
# ---------------------------------------------------------------------------------------------


@dataclass
class AdversarialRun:
    regret: np.ndarray  # (n_runs, n): best fixed arm on rounds 1..t minus the learner's reward
    expected_regret: np.ndarray  # (n_runs,): final regret with X_t replaced by <P_t, x_t>
    arm_totals: np.ndarray  # (n_runs, K): total reward of each arm on the realized table
    learner_total: np.ndarray  # (n_runs,)
    probs: np.ndarray | None = None  # (n_runs, n, K) when recorded


def play(
    learner,
    adversary: Adversary,
    horizon: int,
    rng: np.random.Generator,
    record_probs: bool = False,
) -> AdversarialRun:
    """Run Algorithm 2 for ``horizon`` rounds and measure regret against the best fixed arm."""
    probs = learner.probabilities()
    n_runs, n_arms = probs.shape
    arm_totals = np.zeros((n_runs, n_arms))
    learner_total = np.zeros(n_runs)
    expected_total = np.zeros(n_runs)
    regret = np.zeros((n_runs, horizon))
    recorded = np.zeros((n_runs, horizon, n_arms)) if record_probs else None
    rows = np.arange(n_runs)
    for t in range(horizon):
        if t > 0:
            probs = learner.probabilities()
        x = adversary.reward_vectors(t, probs)
        arms = sample_categorical(probs, rng)
        rewards = x[rows, arms]
        if learner.full_information:
            learner.update_full(x)
        else:
            learner.update(arms, rewards, probs)
        arm_totals += x
        learner_total += rewards
        expected_total += (probs * x).sum(axis=1)
        regret[:, t] = arm_totals.max(axis=1) - learner_total
        if recorded is not None:
            recorded[:, t] = probs
    return AdversarialRun(
        regret=regret,
        expected_regret=arm_totals.max(axis=1) - expected_total,
        arm_totals=arm_totals,
        learner_total=learner_total,
        probs=recorded,
    )


def hedge_bound(horizon: int, n_experts: int) -> float:
    """Expected regret bound of Hedge with epsilon = sqrt(8 log K / n): sqrt(n log K / 2)."""
    return float(np.sqrt(horizon * np.log(n_experts) / 2))


def exp3_bound(horizon: int, n_arms: int) -> float:
    """Report Theorem 1 (Lattimore and Szepesvari Theorem 11.2): sqrt(2 n K log K)."""
    return float(np.sqrt(2 * horizon * n_arms * np.log(n_arms)))


def exp3ix_bound(horizon: int, n_arms: int, delta: float) -> float:
    """Exp3-IX with eta_1, gamma = eta_1 / 2: regret exceeds this with probability at most delta."""
    nk, log_k1 = horizon * n_arms, np.log(n_arms + 1)
    return float(np.sqrt(8.5 * nk * log_k1) + (np.sqrt(nk / (2 * log_k1)) + 1) * np.log(1 / delta))
