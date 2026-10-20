"""Restless experiment: Whittle index, myopic and random policies on random two-state arms.

Sources: Henry Vu, Final Report, Section 4.2 (the relaxed constraint and its Lagrangian) and
slides "Whittle Index and Restless Bandits"; Whittle (1988).

Each arm set has N = 10 arms, and a policy must pull exactly m of them every round. Rewards are
reported as a fraction of Whittle's relaxation bound, which no policy can beat on average.
Writes figures/restless_whittle.png.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mab.restless import (
    average_reward,
    indexable,
    myopic_policy,
    random_fatigue_arm,
    random_policy,
    random_two_state_arm,
    relaxation_upper_bound,
    simulate,
    whittle_policy,
)

FIGURES = ROOT / "figures"
BLUE, PINK, GRAY = "#2f6ea8", "#c0648e", "#8a8a8a"
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

N_ARMS, ARM_SETS, HORIZON, BURN_IN, RUNS = 10, 30, 2000, 200, 4
GAMMA = 0.99  # discount for the index; close to 1 because the score is long-run average reward
POLICY_NAMES = ("Whittle index", "Myopic", "Random")


def draw_arm_sets(draw, seed: int) -> list:
    rng = np.random.default_rng(seed)
    arm_sets, rejected = [], 0
    while len(arm_sets) < ARM_SETS:
        arms = [draw(rng) for _ in range(N_ARMS)]
        if indexable(arms, GAMMA).all():
            arm_sets.append(arms)
        else:
            rejected += 1
    print(f"drew {ARM_SETS} indexable arm sets ({rejected} rejected)")
    return arm_sets


def fraction_of_bound(arm_sets: list, budgets: list[int], seed: int) -> dict[str, np.ndarray]:
    """Average reward over relaxation bound, shape (arm sets, budgets) per policy."""
    out = {name: np.zeros((len(arm_sets), len(budgets))) for name in POLICY_NAMES}
    for i, arms in enumerate(arm_sets):
        policies = dict(zip(POLICY_NAMES, (whittle_policy(arms, GAMMA), myopic_policy(arms), random_policy())))
        for j, m in enumerate(budgets):
            bound = relaxation_upper_bound(arms, m)
            for name, policy in policies.items():
                totals = simulate(arms, policy, m, HORIZON, np.random.default_rng([seed, i, m]), n_runs=RUNS)
                out[name][i, j] = average_reward(totals, BURN_IN).mean() / bound
    return out


def report(label: str, results: dict[str, np.ndarray], budgets: list[int]) -> None:
    for j, m in enumerate(budgets):
        cells = "  ".join(f"{name} {results[name][:, j].mean():.3f}" for name in POLICY_NAMES)
        print(f"{label:12s} m = {m}: {cells}")


def main() -> None:
    budgets = list(range(1, N_ARMS))
    fatigue = fraction_of_bound(draw_arm_sets(random_fatigue_arm, seed=0), budgets, seed=1)
    report("fatigue", fatigue, budgets)
    unstructured = fraction_of_bound(draw_arm_sets(random_two_state_arm, seed=2), [3, 5], seed=3)
    report("unstructured", unstructured, [3, 5])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhline(1.0, color=GRAY, linewidth=1.2, linestyle="--")
    ax.text(2.5, 1.015, "Relaxation bound", color="#555555", fontsize=11)
    styles = {"Whittle index": (BLUE, "-", "o"), "Myopic": (PINK, "--", "s"), "Random": (PINK, ":", "^")}
    for name in POLICY_NAMES:
        color, linestyle, marker = styles[name]
        values = fatigue[name]
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        ax.fill_between(budgets, low, high, color=color, alpha=0.15, linewidth=0)
        ax.plot(budgets, values.mean(axis=0), color=color, linestyle=linestyle, marker=marker,
                markersize=7, linewidth=2, label=name)
    ax.set_xticks(budgets)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(f"Arms pulled per round, m (of N = {N_ARMS})")
    ax.set_ylabel("Average reward / bound")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "restless_whittle.png")
    plt.close(fig)


if __name__ == "__main__":
    plt.rcParams.update(STYLE)
    FIGURES.mkdir(exist_ok=True)
    main()
