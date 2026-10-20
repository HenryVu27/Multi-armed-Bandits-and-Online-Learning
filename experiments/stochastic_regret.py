"""Cumulative pseudo-regret of explore-then-commit, UCB1 and epsilon-greedy on unit-variance
Gaussian arms, averaged over seeds, with the report's ETC bound drawn for comparison.

Sources: Henry Vu's Final Report (ETC rule and bound, UCB index) and the slide deck "Optimisim,
UCB and ETC"; Lattimore and Szepesvari, Bandit Algorithms (2020), chapters 6 to 8.

Run from the repository root: python experiments/stochastic_regret.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from mab.stochastic import (  # noqa: E402
    UCB1,
    EpsilonGreedy,
    ExploreThenCommit,
    GaussianBandit,
    average_regret,
    etc_regret_bound,
)

MEANS = (1.0, 0.8, 0.6, 0.4, 0.2)
HORIZON = 10_000
ETC_M = 200
EPSILON = 0.1
SEEDS = range(200)
FIGURE = ROOT / "figures" / "stochastic_regret.png"

COLORS = {
    "UCB1": "#2f6ea8",
    "Explore-then-commit": "#b84a32",
    "Epsilon-greedy": "#a87a12",
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


def main() -> None:
    env = GaussianBandit(MEANS)
    policies = [
        (UCB1(), "UCB1", COLORS["UCB1"]),
        (ExploreThenCommit(ETC_M), f"Explore-then-commit, m = {ETC_M}", COLORS["Explore-then-commit"]),
        (EpsilonGreedy(EPSILON), f"ε-greedy, ε = {EPSILON}", COLORS["Epsilon-greedy"]),
    ]
    rounds = np.arange(1, HORIZON + 1)
    explore_end = ETC_M * env.num_arms
    bound_rounds = np.arange(explore_end, HORIZON + 1)
    bound = etc_regret_bound(MEANS, ETC_M, bound_rounds)

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    print(f"Gaussian arms {MEANS}, n = {HORIZON}, {len(SEEDS)} seeds")
    for policy, label, color in policies:
        mean, stderr = average_regret(env, policy, HORIZON, SEEDS)
        ax.fill_between(rounds, mean - stderr, mean + stderr, color=color, alpha=0.2, lw=0)
        ax.plot(rounds, mean, color=color, lw=1.6, label=label)
        print(f"  {label:34s} R_n = {mean[-1]:7.1f} +/- {stderr[-1]:5.1f} (standard error)")
    ax.plot(bound_rounds, bound, color=COLORS["reference"], lw=1.2, ls=(0, (4, 2.5)),
            label="ETC regret bound")
    print(f"  ETC bound at n: {bound[-1]:.1f}; exploration cost m * sum(gaps) = {bound[0]:.1f}")

    ax.set_xlim(0, HORIZON)
    ax.set_ylim(0, None)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative pseudo-regret")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    handles, labels = ax.get_legend_handles_labels()
    order = [1, 3, 2, 0]  # first column ETC and its bound, second column the other two
    ax.legend([handles[i] for i in order], [labels[i] for i in order], ncol=2,
              loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=2.2, columnspacing=1.4,
              borderaxespad=0.3)
    FIGURE.parent.mkdir(exist_ok=True)
    fig.savefig(FIGURE)
    print(f"saved {FIGURE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
