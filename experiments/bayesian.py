"""Bayesian bandit experiments on Bernoulli arms.

(a) Thompson sampling against Bayesian greedy and UCB1: cumulative pseudo-regret over seeds.
(b) Gittins indices of Beta(alpha, beta) arms for gamma = 0.9.
(c) The Gittins policy against Thompson sampling and Bayesian greedy on a discounted Bernoulli
    bandit whose arm means are drawn from the uniform prior.

Sources: Henry Vu's Final Report, sections 3 and 4 (Algorithms 5 and 6, the calibration
recursion), and the slide decks "Bayesian Bandits, Thompson Sampling and Gittins Index" and
"Markov Process and Gittins Index"; Gittins, Glazebrook and Weber, Multi-armed Bandit Allocation
Indices (2nd ed., 2011); Lattimore and Szepesvari, Bandit Algorithms (2020), chapters 35 and 36.

Run from the repository root: python experiments/bayesian.py
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from mab.bayesian import BayesianGreedy, GittinsPolicy, ThompsonSampling, gittins_index  # noqa: E402
from mab.stochastic import UCB1, BernoulliBandit, Policy, average_regret, run  # noqa: E402

# (a) the three-arm example from the Thompson sampling slides
REGRET_MEANS = (0.9, 0.5, 0.2)
REGRET_HORIZON = 5_000
REGRET_SEEDS = range(500)

# (b) and (c)
GAMMA = 0.9
TABLE_SIZE = 8
DISCOUNTED_ARMS = 5
DISCOUNTED_HORIZON = 120  # gamma^120 < 4e-6, so later rounds are negligible
DISCOUNTED_SEEDS = range(2_000)
PLOTTED_ROUNDS = 60

FIGURES = ROOT / "figures"
COLORS = {
    "UCB1": "#2f6ea8",
    "Gittins index": "#2f6ea8",
    "Thompson sampling": "#23876a",
    "Bayesian greedy": "#c0648e",
    "reference": "#8c8c8c",
}
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.edgecolor": "#4d4d4d",
    "axes.labelcolor": "#1f1f1f",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#4d4d4d",
    "ytick.color": "#4d4d4d",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}


def regret_comparison() -> None:
    env = BernoulliBandit(REGRET_MEANS)
    rounds = np.arange(1, REGRET_HORIZON + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    print(f"(a) Bernoulli arms {REGRET_MEANS}, n = {REGRET_HORIZON}, {len(REGRET_SEEDS)} seeds")
    for policy in (ThompsonSampling(), BayesianGreedy(), UCB1()):
        mean, stderr = average_regret(env, policy, REGRET_HORIZON, REGRET_SEEDS)
        color = COLORS[policy.name]
        ax.fill_between(rounds, mean - stderr, mean + stderr, color=color, alpha=0.2, lw=0)
        ax.plot(rounds, mean, color=color, lw=1.6, label=policy.name)
        print(f"    {policy.name:18s} R_n = {mean[-1]:6.1f} +/- {stderr[-1]:4.1f} (standard error)")
    ax.set_xlim(0, REGRET_HORIZON)
    ax.set_ylim(0, None)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative pseudo-regret")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(loc="upper left", handlelength=2.2)
    save(fig, "bayesian_regret.png")


def gittins_table() -> None:
    size = TABLE_SIZE
    values = np.array(
        [[gittins_index(float(a), float(b), GAMMA) for a in range(1, size + 1)]
         for b in range(1, size + 1)]
    )
    print(f"(b) Gittins index, gamma = {GAMMA}: Beta(1, 1) -> {values[0, 0]:.4f}, "
          f"Beta(2, 3) -> {values[2, 1]:.4f}, Beta({size}, {size}) -> {values[-1, -1]:.4f}")
    cmap = LinearSegmentedColormap.from_list("gittins", ["#f5f8fc", "#2f6ea8"])
    fig, ax = plt.subplots(figsize=(4.1, 3.6))
    image = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, origin="upper")
    for b in range(size):
        for a in range(size):
            value = values[b, a]
            ax.text(a, b, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > 0.84 else "#1f1f1f")
    ticks = np.arange(size)
    ax.set_xticks(ticks, [str(t + 1) for t in ticks])
    ax.set_yticks(ticks, [str(t + 1) for t in ticks])
    ax.tick_params(length=0)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    for side in ("top", "bottom", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("α (prior plus successes)")
    ax.set_ylabel("β (prior plus failures)")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label(f"Gittins index, γ = {GAMMA}")
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, width=0.6)
    save(fig, "gittins_indices.png")


def discounted_comparison() -> None:
    discount = GAMMA ** np.arange(DISCOUNTED_HORIZON)
    policies: list[Policy] = [GittinsPolicy(GAMMA), BayesianGreedy(), ThompsonSampling()]
    per_round = {}
    best_mean = np.empty(len(DISCOUNTED_SEEDS))
    for policy in policies:
        expected = np.empty((len(DISCOUNTED_SEEDS), DISCOUNTED_HORIZON))
        for row, seed in enumerate(DISCOUNTED_SEEDS):
            rng = np.random.default_rng(seed)
            env = BernoulliBandit(rng.beta(1.0, 1.0, size=DISCOUNTED_ARMS))
            arms, _ = run(env, policy, DISCOUNTED_HORIZON, rng)
            expected[row] = env.means[arms]
            best_mean[row] = env.means.max()
        per_round[policy.name] = expected

    num_seeds = len(DISCOUNTED_SEEDS)
    totals = {name: values @ discount for name, values in per_round.items()}
    oracle = best_mean * discount.sum()
    print(f"(c) {DISCOUNTED_ARMS} arms with means from Beta(1, 1), gamma = {GAMMA}, "
          f"{num_seeds} seeds; discounted total reward (mean +/- standard error):")
    print(f"    best arm known in advance  {oracle.mean():.3f} +/- {oracle.std(ddof=1) / math.sqrt(num_seeds):.3f}")
    for name, total in totals.items():
        line = f"    {name:26s} {total.mean():.3f} +/- {total.std(ddof=1) / math.sqrt(num_seeds):.3f}"
        if name != "Gittins index":
            gap = totals["Gittins index"] - total
            line += (f"; Gittins minus this policy, paired: {gap.mean():.3f} "
                     f"+/- {gap.std(ddof=1) / math.sqrt(num_seeds):.3f}")
        print(line)

    rounds = np.arange(1, PLOTTED_ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    ax.axhline(best_mean.mean(), color=COLORS["reference"], lw=1.2, ls=(0, (4, 2.5)),
               label=f"Best arm known in advance ({oracle.mean():.2f})")
    for name, values in per_round.items():
        mean = values[:, :PLOTTED_ROUNDS].mean(axis=0)
        stderr = values[:, :PLOTTED_ROUNDS].std(axis=0, ddof=1) / math.sqrt(num_seeds)
        ax.fill_between(rounds, mean - stderr, mean + stderr, color=COLORS[name], alpha=0.2, lw=0)
        ax.plot(rounds, mean, color=COLORS[name], lw=1.6,
                label=f"{name} ({totals[name].mean():.2f})")
    ax.set_xlim(1, PLOTTED_ROUNDS)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Expected reward in round t")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower right",
              title="Discounted total reward", title_fontsize=8.5, alignment="left",
              handlelength=2.2)
    save(fig, "gittins_discounted.png")


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"    saved figures/{name}")


def main() -> None:
    plt.rcParams.update(STYLE)
    regret_comparison()
    gittins_table()
    discounted_comparison()


if __name__ == "__main__":
    main()
