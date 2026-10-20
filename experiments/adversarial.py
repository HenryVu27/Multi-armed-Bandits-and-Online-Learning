"""Adversarial experiments: UCB1, Exp3 and Exp3-IX against a switching and an adaptive adversary.

Sources: Henry Vu, Final Report, Section 1 and slides "Adversarial Bandits, Exp3 Algorithm";
Lattimore and Szepesvari, Bandit Algorithms (2020), Chapters 11 and 12.

Writes figures/adversarial_regret.png, figures/exp3_vs_exp3ix.png and figures/adaptive_regret.png.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mab.adversarial import (
    Exp3,
    Exp3IX,
    LeastLikelyAdversary,
    ObliviousAdversary,
    UCB1,
    exp3_bound,
    exp3ix_bound,
    play,
    switching_table,
)

FIGURES = ROOT / "figures"
COLORS = {"UCB1": "#2f6ea8", "Exp3": "#b84a32", "Exp3-IX": "#23876a"}
GRAY = "#8a8a8a"
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

HORIZON, ARMS = 10_000, 5
PHASE_ENDS = (0.2, 0.5, 1.0)


def learners(n_runs: int) -> dict:
    return {
        "UCB1": UCB1(ARMS, n_runs=n_runs),
        "Exp3": Exp3(ARMS, HORIZON, n_runs=n_runs),
        "Exp3-IX": Exp3IX(ARMS, HORIZON, n_runs=n_runs),
    }


def plot_curves(ax, curves: dict[str, np.ndarray]) -> None:
    t = np.arange(1, HORIZON + 1)
    for name, regret in curves.items():
        low, high = np.quantile(regret, [0.1, 0.9], axis=0)
        ax.fill_between(t, low, high, color=COLORS[name], alpha=0.18, linewidth=0)
        ax.plot(t, regret.mean(axis=0), color=COLORS[name], linewidth=2, label=name)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Regret vs best fixed arm")
    ax.ticklabel_format(axis="x", style="plain")


def switching_experiment(n_runs: int = 200) -> None:
    table = switching_table(HORIZON, ARMS, np.random.default_rng(0), n_runs=n_runs, phase_ends=PHASE_ENDS)
    curves = {}
    for name, learner in learners(n_runs).items():
        run = play(learner, ObliviousAdversary(table), HORIZON, np.random.default_rng(1))
        curves[name] = run.regret
        final = run.regret[:, -1]
        print(f"switching  {name:8s} mean final regret {final.mean():8.1f}  (10-90%: "
              f"{np.quantile(final, 0.1):.0f} to {np.quantile(final, 0.9):.0f})")
    bound = exp3_bound(HORIZON, ARMS)
    print(f"switching  Exp3 bound sqrt(2nK log K) = {bound:.0f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for end in PHASE_ENDS[:-1]:
        ax.axvline(end * HORIZON, color=GRAY, linewidth=0.8, linestyle=":")
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axhline(bound, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(HORIZON * 0.02, bound + 60, r"Exp3 bound $\sqrt{2nK\log K}$", color="#555555", fontsize=11)
    plot_curves(ax, curves)
    ax.legend(loc="lower left")
    fig.savefig(FIGURES / "adversarial_regret.png")
    plt.close(fig)


def spread_experiment(n_runs: int = 1000) -> None:
    table = switching_table(HORIZON, ARMS, np.random.default_rng(2), n_runs=n_runs, phase_ends=PHASE_ENDS)
    finals = {}
    for name in ("Exp3", "Exp3-IX"):
        learner = learners(n_runs)[name]
        finals[name] = play(learner, ObliviousAdversary(table), HORIZON, np.random.default_rng(3)).regret[:, -1]
    bound = exp3_bound(HORIZON, ARMS)
    ix_bound = exp3ix_bound(HORIZON, ARMS, delta=0.01)
    for name, final in finals.items():
        print(f"spread     {name:8s} median {np.median(final):7.0f}  std {final.std():6.0f}  "
              f"99th pct {np.quantile(final, 0.99):6.0f}  max {final.max():6.0f}  "
              f"runs above sqrt(2nK log K): {np.mean(final > bound):.1%}  "
              f"above Exp3-IX bound (delta = 0.01): {np.mean(final > ix_bound):.1%}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    edges = np.linspace(min(f.min() for f in finals.values()), max(f.max() for f in finals.values()), 60)
    for name, final in finals.items():
        ax.hist(final, bins=edges, color=COLORS[name], alpha=0.35, linewidth=0)
        ax.hist(final, bins=edges, color=COLORS[name], histtype="step", linewidth=1.6, label=name)
    ax.axvline(bound, color=GRAY, linewidth=1.2, linestyle="--")
    ax.axvline(ix_bound, color=GRAY, linewidth=1.2, linestyle=":")
    top = ax.get_ylim()[1]
    ax.text(bound + 30, top * 0.93, r"$\sqrt{2nK\log K}$", color="#555555", fontsize=11)
    ax.text(ix_bound + 30, top * 0.93, "Exp3-IX bound\n" + r"($\delta = 0.01$)", color="#555555",
            fontsize=11, va="top")
    ax.set_xlabel(f"Final regret after n = {HORIZON:,} rounds")
    ax.set_ylabel(f"Runs (of {n_runs:,})")
    ax.legend(loc="center right")
    fig.savefig(FIGURES / "exp3_vs_exp3ix.png")
    plt.close(fig)


def adaptive_experiment(n_runs: int = 100) -> None:
    curves = {}
    for name, learner in learners(n_runs).items():
        run = play(learner, LeastLikelyAdversary(), HORIZON, np.random.default_rng(4))
        curves[name] = run.regret
        print(f"adaptive   {name:8s} mean final regret {run.regret[:, -1].mean():8.1f}")
    bound = exp3_bound(HORIZON, ARMS)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(bound, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(HORIZON * 0.22, bound + 180, r"Exp3 bound $\sqrt{2nK\log K}$", color="#555555", fontsize=11)
    plot_curves(ax, curves)
    ax.text(HORIZON * 0.98, curves["Exp3"][:, -1].mean() + 900, "Exp3 and Exp3-IX\n(curves overlap)",
            color="#333333", fontsize=11, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93))
    fig.savefig(FIGURES / "adaptive_regret.png")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    FIGURES.mkdir(exist_ok=True)
    switching_experiment()
    spread_experiment()
    adaptive_experiment()
