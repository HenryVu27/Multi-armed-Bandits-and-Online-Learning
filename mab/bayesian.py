"""Bayesian bandits with Bernoulli arms and Beta priors: Thompson sampling, the Bayesian greedy
method and the Gittins index.

Sources: Henry Vu's Final Report, section 3 (Algorithm 5, Thompson sampling; Algorithm 6, the
greedy method; the Beta posterior update) and section 4 (the calibration recursion for
V(p; alpha, beta, gamma)); the slide decks "Bayesian Bandits, Thompson Sampling and Gittins
Index" and "Markov Process and Gittins Index"; W. R. Thompson, Biometrika 25 (1933);
J. C. Gittins, "Bandit processes and dynamic allocation indices", JRSS B 41 (1979);
J. Gittins, K. Glazebrook and R. Weber, Multi-armed Bandit Allocation Indices (2nd ed., 2011);
T. Lattimore and C. Szepesvari, Bandit Algorithms (2020), chapters 35 and 36.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from mab.stochastic import Policy, argmax_random


class BetaBernoulliPolicy(Policy):
    """Keeps a Beta(alpha_i, beta_i) posterior over each arm's success probability.

    After observing x in {0, 1} from arm i the posterior is Beta(alpha_i + x, beta_i + 1 - x),
    because theta^x (1 - theta)^(1 - x) times the Beta(alpha_i, beta_i) density is proportional
    to the Beta(alpha_i + x, beta_i + 1 - x) density (report, section 3).
    """

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def reset(self, num_arms: int) -> None:
        super().reset(num_arms)
        self.alpha = np.full(num_arms, float(self.prior_alpha))
        self.beta = np.full(num_arms, float(self.prior_beta))

    def update(self, arm: int, reward: float) -> None:
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    @property
    def posterior_means(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)


class ThompsonSampling(BetaBernoulliPolicy):
    """Algorithm 5: sample theta_i ~ Beta(alpha_i, beta_i) for every arm, pull the argmax."""

    name = "Thompson sampling"

    def select(self, t: int, rng: np.random.Generator) -> int:
        return argmax_random(rng.beta(self.alpha, self.beta), rng)


class BayesianGreedy(BetaBernoulliPolicy):
    """Algorithm 6: pull the arm with the largest posterior mean alpha_i / (alpha_i + beta_i)."""

    name = "Bayesian greedy"

    def select(self, t: int, rng: np.random.Generator) -> int:
        return argmax_random(self.posterior_means, rng)


def truncation_depth(gamma: float, tol: float) -> int:
    """Smallest depth N with gamma^N / (1 - gamma) <= tol.

    Rewards lie in [0, 1], so everything after N more pulls is worth at most gamma^N / (1 - gamma)
    and cutting the recursion there changes V by no more than tol.
    """
    return max(1, math.ceil(math.log(tol * (1 - gamma)) / math.log(gamma)))


def _continuation_value(p: float, alpha: float, beta: float, gamma: float, depth: int) -> float:
    """Second term of the report's max: pull the Beta(alpha, beta) arm once, then act optimally.

    V(p; a, b) = max{ p / (1 - gamma),
                      a/(a+b) (1 + gamma V(p; a+1, b)) + b/(a+b) gamma V(p; a, b+1) }.
    Every state reachable in d more pulls is (alpha + s, beta + d - s) for s successes, so the
    recursion is evaluated bottom-up one level of the (alpha, beta) lattice at a time; each state's
    value is computed once and stored in the level array, which is the memoised recursion without
    Python call overhead. At the truncation depth the arm can no longer be learned about, so its
    value is max(p, posterior mean) / (1 - gamma).
    """
    if depth < 1:
        raise ValueError("depth must be at least 1")
    retire = p / (1.0 - gamma)
    successes = np.arange(depth + 1)
    a = alpha + successes
    b = beta + depth - successes
    values = np.maximum(retire, a / (a + b) / (1.0 - gamma))
    for level in range(depth - 1, -1, -1):
        successes = np.arange(level + 1)
        a = alpha + successes
        b = beta + level - successes
        mean = a / (a + b)
        continuation = mean * (1.0 + gamma * values[1:]) + (1.0 - mean) * gamma * values[:-1]
        values = np.maximum(retire, continuation)
    return float(continuation[0])


def calibration_value(
    p: float, alpha: float, beta: float, gamma: float, depth: int | None = None, tol: float = 1e-7
) -> float:
    """The report's V(p; alpha, beta, gamma): the best expected discounted reward from a
    Beta(alpha, beta) arm when a known arm paying p every step is also available."""
    depth = truncation_depth(gamma, tol) if depth is None else depth
    return max(p / (1.0 - gamma), _continuation_value(p, alpha, beta, gamma, depth))


@lru_cache(maxsize=None)
def gittins_index(alpha: float, beta: float, gamma: float, tol: float = 1e-6) -> float:
    """Gittins index of a Beta(alpha, beta) Bernoulli arm under discount gamma, in reward-per-step
    units: the known payoff p at which retiring forever (p / (1 - gamma)) and pulling the arm are
    equally good.

    Found by bisection on p in [posterior mean, 1]: if pulling beats retiring, the index is above
    p. The recursion is truncated at a depth where gamma^depth / (1 - gamma) <= tol / 10, and the
    bisection stops once the bracket is narrower than tol. Results are cached per state.
    """
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1)")
    depth = truncation_depth(gamma, tol / 10)
    low, high = alpha / (alpha + beta), 1.0
    while high - low > tol:
        p = 0.5 * (low + high)
        if _continuation_value(p, alpha, beta, gamma, depth) > p / (1.0 - gamma):
            low = p
        else:
            high = p
    return 0.5 * (low + high)


class GittinsPolicy(BetaBernoulliPolicy):
    """Pull the arm with the largest Gittins index; Bayes-optimal for the discounted Bernoulli
    bandit with independent Beta priors (Gittins index theorem)."""

    name = "Gittins index"

    def __init__(
        self, gamma: float, prior_alpha: float = 1.0, prior_beta: float = 1.0, tol: float = 1e-6
    ) -> None:
        super().__init__(prior_alpha, prior_beta)
        self.gamma = gamma
        self.tol = tol

    def indices(self) -> np.ndarray:
        return np.array(
            [gittins_index(a, b, self.gamma, self.tol) for a, b in zip(self.alpha, self.beta)]
        )

    def select(self, t: int, rng: np.random.Generator) -> int:
        return argmax_random(self.indices(), rng)
