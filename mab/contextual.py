"""Contextual bandits: per-context Exp3, Exp4 with expert advice, and LinUCB with replay evaluation.

Sources: Henry Vu, Final Report, Section 2 (the |C| construction, Section 2.1 and Algorithm 4
Exp4); slides "Contextual Bandits"; Lattimore and Szepesvari, Bandit Algorithms (2020),
Chapter 18; Li, Chu, Langford and Schapire, "A contextual-bandit approach to personalized news
article recommendation" (WWW 2010) for LinUCB; Li, Chu, Langford and Wang, "Unbiased offline
evaluation of contextual-bandit-based news article recommendation algorithms" (WSDM 2011) for
replay. LinUCB and the replay loop are ported from ``project/MABs.ipynb``.

Per-context Exp3 and Exp4 run ``n_runs`` copies side by side like the learners in
``mab.adversarial``. The replay policies run one copy and see one logged event at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mab.adversarial import sample_categorical, softmax

# ---------------------------------------------------------------------------------------------
# Finite contexts and expert advice
# ---------------------------------------------------------------------------------------------


class PerContextExp3:
    """One Exp3 instance per context (report Section 2), each with the anytime rate
    eta_s = sqrt(log K / (s K)) on its s-th use, since T_c(n) is not known in advance."""

    def __init__(self, n_contexts: int, n_arms: int, n_runs: int = 1):
        self.n_arms = n_arms
        self.S = np.zeros((n_runs, n_contexts, n_arms))
        self.visits = np.zeros((n_runs, n_contexts), dtype=int)

    def probabilities(self, contexts: np.ndarray, advice: np.ndarray | None = None) -> np.ndarray:
        rows = np.arange(len(contexts))
        s = self.visits[rows, contexts] + 1
        eta = np.sqrt(np.log(self.n_arms) / (s * self.n_arms))
        return softmax(eta[:, None] * self.S[rows, contexts])

    def update(
        self,
        contexts: np.ndarray,
        advice: np.ndarray | None,
        arms: np.ndarray,
        rewards: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        rows = np.arange(len(contexts))
        self.visits[rows, contexts] += 1
        self.S[rows, contexts] += 1.0
        self.S[rows, contexts, arms] -= (1.0 - rewards) / probs[rows, arms]


class Exp4:
    """Report Algorithm 4. Advice E(t) has shape (M, K); Q_t (1 x M) are expert weights,
    P_t = Q_t E(t), and the experts' reward estimates are X~_t = E(t) X^_t.

    The default eta = sqrt(2 log M / (n min(M, K))) uses the report's bound E*_n <= n min(M, K),
    so the regret is at most sqrt(2 n min(M, K) log M). With the K unit vectors as experts this
    is Exp3 with eta = sqrt(2 log K / (nK)).
    """

    def __init__(
        self, n_experts: int, n_arms: int, horizon: int, eta: float | None = None, n_runs: int = 1
    ):
        effective = min(n_experts, n_arms)
        self.eta = np.sqrt(2 * np.log(n_experts) / (horizon * effective)) if eta is None else eta
        self.S = np.zeros((n_runs, n_experts))  # S^_t,m, so Q_1 is uniform

    def expert_weights(self) -> np.ndarray:
        return softmax(self.eta * self.S)

    def probabilities(self, contexts: np.ndarray | None, advice: np.ndarray) -> np.ndarray:
        return np.einsum("rm,rmk->rk", self.expert_weights(), advice)

    def update(
        self,
        contexts: np.ndarray | None,
        advice: np.ndarray,
        arms: np.ndarray,
        rewards: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        rows = np.arange(len(arms))
        x_hat = np.ones(probs.shape)
        x_hat[rows, arms] -= (1.0 - rewards) / probs[rows, arms]
        self.S += np.einsum("rmk,rk->rm", advice, x_hat)


@dataclass
class AdviceProblem:
    """Contexts drawn i.i.d. from ``context_probs``, Bernoulli rewards with ``means[c, k]``, and
    M experts whose advice in context c is the distribution ``advice[m, c]`` over arms."""

    context_probs: np.ndarray  # (C,)
    means: np.ndarray  # (C, K)
    advice: np.ndarray  # (M, C, K)


def make_advice_problem(
    n_contexts: int,
    n_arms: int,
    n_experts: int,
    rng: np.random.Generator,
    best_mean: float = 0.75,
    other_means: tuple[float, float] = (0.2, 0.5),
    good_expert_mass: float = 0.85,
    concentration: float = 0.5,
) -> AdviceProblem:
    """Each context has one best arm. Expert 0 puts ``good_expert_mass`` on it and spreads the
    rest evenly; the other experts give Dirichlet(``concentration``) advice unrelated to rewards."""
    means = rng.uniform(*other_means, size=(n_contexts, n_arms))
    best_arm = rng.integers(n_arms, size=n_contexts)
    means[np.arange(n_contexts), best_arm] = best_mean
    advice = rng.dirichlet(np.full(n_arms, concentration), size=(n_experts, n_contexts))
    spread = (1.0 - good_expert_mass) / (n_arms - 1)
    advice[0] = spread
    advice[0, np.arange(n_contexts), best_arm] = good_expert_mass
    return AdviceProblem(np.full(n_contexts, 1.0 / n_contexts), means, advice)


@dataclass
class ContextualRun:
    regret_to_best_expert: np.ndarray  # (n_runs, n): best expert on rounds 1..t minus learner
    regret_to_best_mapping: np.ndarray  # (n_runs,): S_n - sum_t X_t, S_n from report Section 2
    expert_totals: np.ndarray  # (n_runs, M): sum_t E(t)_m x_t
    learner_total: np.ndarray  # (n_runs,)


def play_advice_problem(
    learner, problem: AdviceProblem, horizon: int, rng: np.random.Generator
) -> ContextualRun:
    """Run a learner on the advice problem. Contexts and rewards come from one child generator
    and arm draws from another, so learners given the same ``rng`` seed face the same table."""
    env_rng, learner_rng = rng.spawn(2)
    n_runs = learner.S.shape[0]
    n_experts, n_contexts, n_arms = problem.advice.shape
    rows = np.arange(n_runs)
    expert_totals = np.zeros((n_runs, n_experts))
    context_arm_totals = np.zeros((n_runs, n_contexts, n_arms))
    learner_total = np.zeros(n_runs)
    regret = np.zeros((n_runs, horizon))
    for t in range(horizon):
        contexts = env_rng.choice(n_contexts, size=n_runs, p=problem.context_probs)
        x = (env_rng.random((n_runs, n_arms)) < problem.means[contexts]).astype(float)
        advice = problem.advice[:, contexts].transpose(1, 0, 2)  # (n_runs, M, K)
        probs = learner.probabilities(contexts, advice)
        arms = sample_categorical(probs, learner_rng)
        rewards = x[rows, arms]
        learner.update(contexts, advice, arms, rewards, probs)
        expert_totals += np.einsum("rmk,rk->rm", advice, x)
        context_arm_totals[rows, contexts] += x
        learner_total += rewards
        regret[:, t] = expert_totals.max(axis=1) - learner_total
    best_mapping = context_arm_totals.max(axis=2).sum(axis=1)
    return ContextualRun(regret, best_mapping - learner_total, expert_totals, learner_total)


# ---------------------------------------------------------------------------------------------
# Logged news data: LinUCB, context-free baselines and replay
# ---------------------------------------------------------------------------------------------


def _random_argmax(scores: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(np.flatnonzero(scores == scores.max())))


class LinUCB:
    """Disjoint LinUCB: a ridge regression per arm, A_a = ridge * I + sum x x^T, b_a = sum r x.

    Score: theta_a . x_a + alpha * sqrt(x_a^T A_a^-1 x_a), with theta_a = A_a^-1 b_a. A_a^-1 is
    kept up to date with the Sherman-Morrison formula instead of a fresh inverse each round.
    """

    def __init__(self, n_arms: int, dim: int, alpha: float, ridge: float = 1.0):
        self.alpha = alpha
        self.A_inv = np.tile(np.eye(dim) / ridge, (n_arms, 1, 1))
        self.b = np.zeros((n_arms, dim))

    def scores(self, context: np.ndarray) -> np.ndarray:
        theta = np.einsum("kij,kj->ki", self.A_inv, self.b)
        mean = np.einsum("ki,ki->k", theta, context)
        width = np.sqrt(np.einsum("ki,kij,kj->k", context, self.A_inv, context))
        return mean + self.alpha * width

    def select(self, context: np.ndarray, rng: np.random.Generator) -> int:
        return _random_argmax(self.scores(context), rng)

    def update(self, arm: int, reward: float, context: np.ndarray) -> None:
        x = context[arm]
        Ax = self.A_inv[arm] @ x
        self.A_inv[arm] -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[arm] += reward * x


class EpsilonGreedy:
    """Context-free epsilon-greedy from the notebook; untried arms count as best."""

    def __init__(self, n_arms: int, epsilon: float):
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def select(self, context: np.ndarray | None, rng: np.random.Generator) -> int:
        if rng.random() < self.epsilon:
            return int(rng.integers(len(self.counts)))
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(self.counts > 0, self.sums / self.counts, np.inf)
        return _random_argmax(means, rng)

    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


class ContextFreeUCB:
    """Context-free UCB from the notebook: mean + sqrt(rho log t / T_a), untried arms first.

    t counts the policy's own rounds (matched events under replay), not every logged event.
    """

    def __init__(self, n_arms: int, rho: float = 1.0):
        self.rho = rho
        self.counts = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)

    def select(self, context: np.ndarray | None, rng: np.random.Generator) -> int:
        t = self.counts.sum() + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            index = self.sums / self.counts + np.sqrt(self.rho * np.log(t) / self.counts)
        index[self.counts == 0] = np.inf
        return _random_argmax(index, rng)

    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward


@dataclass
class NewsLog:
    arms: np.ndarray  # (N,) article shown by the uniformly random logging policy
    rewards: np.ndarray  # (N,) click (1) or not (0)
    contexts: np.ndarray  # (N, K, d) one feature vector per article


def load_news_log(path: str | Path) -> NewsLog:
    """Read ``project/dataset.txt``: arm, reward, then K * d context features per line."""
    data = np.loadtxt(path)
    arms = data[:, 0].astype(int)
    n_arms = int(arms.max()) + 1
    contexts = data[:, 2:].reshape(len(data), n_arms, -1)
    return NewsLog(arms, data[:, 1].astype(float), contexts)


def replay_evaluate(
    policy,
    log: NewsLog,
    rng: np.random.Generator,
    order: np.ndarray | None = None,
    max_matches: int | None = None,
) -> np.ndarray:
    """Replay estimator (Li et al. 2011): walk the log, keep an event only when the policy picks
    the logged arm, and update the policy only on kept events. Returns the kept rewards.

    Unbiased when the log came from a uniformly random policy and events are i.i.d., which also
    makes any reordering ``order`` of the log an equally valid replay.
    """
    order = np.arange(len(log.arms)) if order is None else order
    matched: list[float] = []
    for i in order:
        context = log.contexts[i]
        if policy.select(context, rng) == log.arms[i]:
            matched.append(log.rewards[i])
            policy.update(int(log.arms[i]), float(log.rewards[i]), context)
            if max_matches is not None and len(matched) == max_matches:
                break
    return np.asarray(matched)
