"""Stochastic multi-armed bandits: Gaussian and Bernoulli arms, explore-then-commit, UCB1 and
epsilon-greedy, measured by pseudo-regret.

Sources: Henry Vu's Final Report (pseudo-regret, the explore-then-commit rule A_t and its regret
bound, the UCB index with f(t) = 1 + t log^2 t); the slide decks "Multi-armed Bandits -
Introduction" and "Optimisim, UCB and ETC"; T. Lattimore and C. Szepesvari, Bandit Algorithms
(Cambridge University Press, 2020), chapters 6 to 8; P. Auer, N. Cesa-Bianchi and P. Fischer,
"Finite-time analysis of the multiarmed bandit problem", Machine Learning 47 (2002).

Rounds are numbered t = 1, ..., n as in the report. Arms are numbered 0, ..., K - 1, so the
report's arm i is arm i - 1 here.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np


class BanditEnvironment(ABC):
    """K independent arms with fixed means.

    Each round draws a reward vector X_t = (X_{t,1}, ..., X_{t,K}) and the learner observes only
    the entry of the arm it pulled. Drawing the whole table up front means two policies run with
    the same seed face exactly the same rewards.
    """

    def __init__(self, means: Sequence[float]) -> None:
        self.means = np.asarray(means, dtype=float)
        if self.means.ndim != 1 or self.means.size < 2:
            raise ValueError("need a 1-D array of at least two arm means")

    @property
    def num_arms(self) -> int:
        return self.means.size

    @property
    def gaps(self) -> np.ndarray:
        """Suboptimality gaps Delta_i = mu* - mu_i."""
        return self.means.max() - self.means

    @abstractmethod
    def sample_rewards(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Reward table of shape (n, K); row t - 1 is the reward vector of round t."""


class GaussianBandit(BanditEnvironment):
    """Arms with unit-variance Gaussian rewards, so X_t - E[X_t] is 1-subgaussian."""

    def sample_rewards(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.means + rng.standard_normal((n, self.num_arms))


class BernoulliBandit(BanditEnvironment):
    """Arms that pay 1 with probability mu_i and 0 otherwise."""

    def __init__(self, means: Sequence[float]) -> None:
        super().__init__(means)
        if np.any((self.means < 0) | (self.means > 1)):
            raise ValueError("Bernoulli means must lie in [0, 1]")

    def sample_rewards(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return (rng.random((n, self.num_arms)) < self.means).astype(float)


def argmax_random(values: np.ndarray, rng: np.random.Generator) -> int:
    """Index of the largest value, with ties broken uniformly at random."""
    best = np.flatnonzero(values == values.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


class Policy(ABC):
    """A bandit algorithm. `reset` starts a fresh run, `select` picks A_t, `update` sees X_t."""

    name = "policy"

    def reset(self, num_arms: int) -> None:
        self.counts = np.zeros(num_arms, dtype=int)  # T_i(t - 1)
        self.empirical_means = np.zeros(num_arms)  # mu_hat_i(t - 1)

    @abstractmethod
    def select(self, t: int, rng: np.random.Generator) -> int:
        """Arm to pull in round t (1-indexed)."""

    def update(self, arm: int, reward: float) -> None:
        self.counts[arm] += 1
        self.empirical_means[arm] += (reward - self.empirical_means[arm]) / self.counts[arm]


class ExploreThenCommit(Policy):
    """Explore-then-commit (ETC) as defined in the report.

    A_t = i if (t mod K) + 1 = i and t <= mK, and A_t = argmax_i mu_hat_i(mK) for t > mK.
    With 0-indexed arms the exploration rule is simply A_t = t mod K.
    """

    name = "Explore-then-commit"

    def __init__(self, m: int) -> None:
        if m < 1:
            raise ValueError("m must be at least 1")
        self.m = m

    def reset(self, num_arms: int) -> None:
        super().reset(num_arms)
        self.committed_arm: int | None = None

    def select(self, t: int, rng: np.random.Generator) -> int:
        num_arms = self.counts.size
        if t <= self.m * num_arms:
            return t % num_arms
        if self.committed_arm is None:
            self.committed_arm = argmax_random(self.empirical_means, rng)
        return self.committed_arm


def ucb_f(t: int | np.ndarray) -> float | np.ndarray:
    """f(t) = 1 + t log^2(t), the confidence schedule of Lattimore and Szepesvari, chapter 8."""
    return 1.0 + t * np.log(t) ** 2


class UCB1(Policy):
    """Upper confidence bound policy with the report's index.

    A_t = t for t <= K; afterwards A_t = argmax_i mu_hat_i(t-1) + sqrt(2 log f(t) / T_i(t-1)).
    Auer et al.'s original UCB1 uses log t in place of log f(t); the slower-growing f(t) of
    Lattimore and Szepesvari gives the bound sum_i (Delta_i + (8 log f(n) + 8 sqrt(pi log f(n))
    + 28) / Delta_i) quoted in the report.
    """

    name = "UCB1"

    def index(self, t: int) -> np.ndarray:
        """UCB index of every arm in round t (requires every arm pulled at least once)."""
        return self.empirical_means + np.sqrt(2.0 * math.log(ucb_f(t)) / self.counts)

    def select(self, t: int, rng: np.random.Generator) -> int:
        num_arms = self.counts.size
        if t <= num_arms:
            return t - 1
        return argmax_random(self.index(t), rng)


class EpsilonGreedy(Policy):
    """Epsilon-greedy: a uniformly random arm with probability epsilon, otherwise the arm with
    the highest empirical mean. Each arm is pulled once first so every mean is defined."""

    name = "Epsilon-greedy"

    def __init__(self, epsilon: float) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        self.epsilon = epsilon

    def select(self, t: int, rng: np.random.Generator) -> int:
        num_arms = self.counts.size
        if t <= num_arms:
            return t - 1
        if rng.random() < self.epsilon:
            return int(rng.integers(num_arms))
        return argmax_random(self.empirical_means, rng)


def run(
    env: BanditEnvironment, policy: Policy, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Play n rounds. Returns the arms pulled and the rewards observed, each of length n.

    The reward table is drawn from `rng` first and the policy's own randomness afterwards, so
    policies run with equal seeds see identical rewards.
    """
    rewards = env.sample_rewards(n, rng)
    policy.reset(env.num_arms)
    arms = np.empty(n, dtype=int)
    for t in range(1, n + 1):
        arm = policy.select(t, rng)
        policy.update(arm, rewards[t - 1, arm])
        arms[t - 1] = arm
    return arms, rewards[np.arange(n), arms]


def simulate(
    env: BanditEnvironment, policy: Policy, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Per-round pseudo-regret Delta_{A_t} of one run, t = 1, ..., n.

    Its cumulative sum has expectation n mu* - E[sum_t mu_{A_t}] = sum_i Delta_i E[T_i(n)], the
    report's pseudo-regret.
    """
    arms, _ = run(env, policy, n, rng)
    return env.gaps[arms]


def regret_runs(
    env: BanditEnvironment, policy: Policy, n: int, seeds: Iterable[int]
) -> np.ndarray:
    """Cumulative pseudo-regret curves, one row per seed, shape (len(seeds), n)."""
    return np.stack(
        [np.cumsum(simulate(env, policy, n, np.random.default_rng(seed))) for seed in seeds]
    )


def average_regret(
    env: BanditEnvironment, policy: Policy, n: int, seeds: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error over seeds of the cumulative pseudo-regret after each round."""
    curves = regret_runs(env, policy, n, seeds)
    return curves.mean(axis=0), curves.std(axis=0, ddof=1) / math.sqrt(curves.shape[0])


def etc_regret_bound(
    means: Sequence[float], m: int, n: int | np.ndarray
) -> float | np.ndarray:
    """The report's ETC bound for 1-subgaussian arms and horizon n >= mK:

    R_n <= m sum_i Delta_i + (n - mK) sum_i Delta_i exp(-m Delta_i^2 / 4).
    """
    means = np.asarray(means, dtype=float)
    gaps = means.max() - means
    explore_rounds = m * means.size
    n = np.asarray(n, dtype=float)
    if np.any(n < explore_rounds):
        raise ValueError("the bound needs n >= mK")
    bound = m * gaps.sum() + (n - explore_rounds) * np.sum(gaps * np.exp(-m * gaps**2 / 4))
    return float(bound) if bound.ndim == 0 else bound


def etc_tuned_m(gap: float, n: int) -> int:
    """Exploration length that minimises the two-arm bound (slide "ETC Regret"):
    m = max(1, ceil(4 / Delta^2 log(n Delta^2 / 4)))."""
    return max(1, math.ceil(4 / gap**2 * math.log(n * gap**2 / 4)))
