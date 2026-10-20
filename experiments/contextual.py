"""Contextual experiments: Exp4 on a synthetic advice problem and LinUCB replay on logged news data.

Sources: Henry Vu, Final Report, Section 2 and slides "Contextual Bandits"; Lattimore and
Szepesvari, Bandit Algorithms (2020), Chapter 18; Li et al. (WWW 2010) for LinUCB; Li et al.
(WSDM 2011) for replay; ``project/MABs.ipynb`` for the data and the baselines.

Writes figures/exp4.png and figures/linucb_replay.png.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mab.contextual import (
    ContextFreeUCB,
    EpsilonGreedy,
    Exp4,
    LinUCB,
    PerContextExp3,
    load_news_log,
    make_advice_problem,
    play_advice_problem,
    replay_evaluate,
)

FIGURES = ROOT / "figures"
GOLD, BLUE, PINK, GRAY = "#a87a12", "#2f6ea8", "#c0648e", "#8a8a8a"
STYLE = {
    "font.size": 13,
    "axes.labelsize": 13,
    "legend.fontsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}

# Advice problem: 50 contexts, 5 arms, 32 experts (one informative), 10,000 rounds.
HORIZON, CONTEXTS, ARMS, EXPERTS = 10_000, 50, 5, 32
# Replay: parameters fixed before looking at replay scores. alpha = 1 is the LinUCB default
# the notebook started from; the notebook's tuned alpha was chosen on the scored events.
ALPHA, EPSILON, RHO, ORDERINGS = 1.0, 0.1, 1.0, 20


def band(ax, x, values, color, label, linestyle="-"):
    low, high = np.quantile(values, [0.1, 0.9], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, values.mean(axis=0), color=color, linewidth=2, linestyle=linestyle, label=label)


def exp4_experiment(n_runs: int = 100) -> None:
    problem = make_advice_problem(CONTEXTS, ARMS, EXPERTS, np.random.default_rng(0))
    runs = {
        "Exp4": play_advice_problem(Exp4(EXPERTS, ARMS, HORIZON, n_runs=n_runs), problem, HORIZON,
                                    np.random.default_rng(1)),
        "Per-context Exp3": play_advice_problem(PerContextExp3(CONTEXTS, ARMS, n_runs=n_runs), problem,
                                                HORIZON, np.random.default_rng(1)),
    }
    exp4_bound = np.sqrt(2 * HORIZON * min(EXPERTS, ARMS) * np.log(EXPERTS))
    per_context_bound = np.sqrt(2 * CONTEXTS * HORIZON * ARMS * np.log(ARMS))
    best_expert = runs["Exp4"].expert_totals.max(axis=1).mean() / HORIZON
    print(f"exp4       best expert in hindsight earns {best_expert:.3f} per round")
    for name, run in runs.items():
        print(f"exp4       {name:17s} regret vs best expert {run.regret_to_best_expert[:, -1].mean():7.1f}  "
              f"vs best context-to-arm map {run.regret_to_best_mapping.mean():7.1f}  "
              f"reward per round {run.learner_total.mean() / HORIZON:.3f}")
    print(f"exp4       bounds: Exp4 sqrt(2nK log M) = {exp4_bound:.0f}, "
          f"per-context Exp3 sqrt(2|C|nK log K) = {per_context_bound:.0f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    t = np.arange(1, HORIZON + 1)
    ax.axhline(0, color=GRAY, linewidth=1.2, label="Best expert in hindsight")
    ax.axhline(exp4_bound, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(HORIZON * 0.02, exp4_bound + 25, r"Exp4 bound $\sqrt{2nK\log M}$", color="#555555", fontsize=11)
    band(ax, t, runs["Per-context Exp3"].regret_to_best_expert, PINK, "Per-context Exp3", "--")
    band(ax, t, runs["Exp4"].regret_to_best_expert, GOLD, "Exp4")
    ax.set_xlabel("Round t")
    ax.set_ylabel("Regret vs best expert")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9))
    fig.savefig(FIGURES / "exp4.png")
    plt.close(fig)


def replay_experiment() -> None:
    log = load_news_log(ROOT / "project" / "dataset.txt")
    policies = {  # name: (factory, legend label, color, line style)
        f"LinUCB alpha={ALPHA:g}": (lambda: LinUCB(10, 10, ALPHA), rf"LinUCB, $\alpha$ = {ALPHA:g}", GOLD, "-"),
        f"UCB rho={RHO:g}": (lambda: ContextFreeUCB(10, RHO), rf"UCB, $\rho$ = {RHO:g}", BLUE, "--"),
        f"eps-greedy eps={EPSILON:g}": (lambda: EpsilonGreedy(10, EPSILON),
                                        rf"$\epsilon$-greedy, $\epsilon$ = {EPSILON:g}", PINK, ":"),
    }
    orderings = [np.arange(len(log.arms))]
    order_rng = np.random.default_rng(0)
    orderings += [order_rng.permutation(len(log.arms)) for _ in range(ORDERINGS - 1)]
    curves = {}
    for name, (make, *_) in policies.items():
        matched = [replay_evaluate(make(), log, np.random.default_rng(i), order=order)
                   for i, order in enumerate(orderings)]
        length = min(len(m) for m in matched)
        rewards = np.stack([m[:length] for m in matched])
        curves[name] = np.cumsum(rewards, axis=1) / np.arange(1, length + 1)
        means = np.array([m.mean() for m in matched])
        print(f"replay     {name:20s} click rate {means.mean():.3f} (sd {means.std():.3f} over "
              f"{ORDERINGS} orderings, {np.mean([len(m) for m in matched]):.0f} matched events each); "
              f"log order, first 800 matches: {matched[0][:800].mean():.3f}")
    logged = log.rewards.mean()
    best_article = max(log.rewards[log.arms == k].mean() for k in range(10))
    print(f"replay     logged uniform policy {logged:.3f}; best single article {best_article:.3f}")

    length = min(c.shape[1] for c in curves.values())
    x = np.arange(1, length + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for value, offset, text in [(best_article, 0.012, "Best single article"),
                                (logged, -0.045, "Logging policy (uniform)")]:
        ax.axhline(value, color=GRAY, linewidth=1.2, linestyle="--")
        ax.text(length, value + offset, text, color="#555555", fontsize=11, ha="right")
    for name, (_, label, color, style) in policies.items():
        band(ax, x, curves[name][:, :length], color, label, style)
    ax.set_xlim(0, length)
    ax.set_ylim(0, 0.8)
    ax.set_xlabel("Matched events")
    ax.set_ylabel("Running click rate")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES / "linucb_replay.png")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    FIGURES.mkdir(exist_ok=True)
    exp4_experiment()
    replay_experiment()
