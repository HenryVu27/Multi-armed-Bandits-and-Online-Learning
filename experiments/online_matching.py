"""Water-filling against greedy for online bipartite matching, on the hard instance and on random graphs.

Top: the upper-triangular instance of size n (online vertex v is adjacent to offline vertices
v..n-1). Water-filling falls to 1 - 1/e; greedy that sends each arrival to the highest free index
falls to 1/2; greedy with random tie-breaking is averaged over 30 seeds.
Bottom: random graphs with 100 + 100 vertices, a planted perfect matching (OPT = 100) and extra
edges with average degree d per online vertex; lines are means over 40 seeds and bands span the
minimum to the maximum.

Ratios are ALG / OPT. Run: python experiments/online_matching.py  (writes figures/online_matching.png)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

from online import plot_style as style  # noqa: E402
from online.matching import (  # noqa: E402
    GREEDY_RATIO,
    WATER_FILLING_RATIO,
    greedy,
    offline_maximum_matching,
    random_instance,
    upper_triangular_instance,
    water_filling,
)

SIZES = np.unique(np.geomspace(2, 1000, 30).astype(int))
TIE_SEEDS = 30
N_RANDOM = 100
DEGREES = np.geomspace(0.25, 30, 15)
GRAPH_SEEDS = 40
RANDOM_GREEDY = "#d9a3bd"  # lighter step of the baseline color, for the random tie-break variant


def hard_instance_curves(rng):
    wf, adversarial, random_tie = [], [], []
    for n in SIZES:
        adjacency = upper_triangular_instance(n)
        wf.append(water_filling(adjacency).value / n)
        adversarial.append(greedy(adjacency, "last").sum() / n)
        random_tie.append(np.mean([greedy(adjacency, "random", rng).sum() / n for _ in range(TIE_SEEDS)]))
    return np.array(wf), np.array(adversarial), np.array(random_tie)


def random_graph_curves(rng):
    wf = np.zeros((DEGREES.size, GRAPH_SEEDS))
    gr = np.zeros((DEGREES.size, GRAPH_SEEDS))
    for i, d in enumerate(DEGREES):
        for s in range(GRAPH_SEEDS):
            adjacency = random_instance(rng, N_RANDOM, d / N_RANDOM)
            opt = offline_maximum_matching(adjacency)
            wf[i, s] = water_filling(adjacency).value / opt
            gr[i, s] = greedy(adjacency, "random", rng).sum() / opt
    return wf, gr


def main() -> None:
    style.apply()
    rng = np.random.default_rng(11)
    wf_hard, greedy_hard, greedy_random_tie = hard_instance_curves(rng)
    wf_rand, greedy_rand = random_graph_curves(rng)

    print(f"upper-triangular n={SIZES[-1]}: water-filling {wf_hard[-1]:.4f} (1 - 1/e = {WATER_FILLING_RATIO:.4f}), "
          f"greedy adversarial {greedy_hard[-1]:.4f}, greedy random tie-break {greedy_random_tie[-1]:.4f}")
    print(f"random graphs: water-filling min {wf_rand.min():.3f}, mean {wf_rand.mean():.3f}; "
          f"greedy min {greedy_rand.min():.3f}, mean {greedy_rand.mean():.3f}; "
          f"largest mean gap {np.max(wf_rand.mean(1) - greedy_rand.mean(1)):.3f} at d = "
          f"{DEGREES[np.argmax(wf_rand.mean(1) - greedy_rand.mean(1))]:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(style.WIDTH, 7.6))
    ax = axes[0]
    style.bound_line(ax, WATER_FILLING_RATIO)
    style.bound_line(ax, GREEDY_RATIO)
    ax.plot(SIZES, wf_hard, color=style.CLASSICAL, marker="o", markevery=4, zorder=3)
    ax.plot(SIZES, greedy_hard, color=style.BASELINE, marker="s", markevery=(2, 4), zorder=3)
    ax.plot(SIZES, greedy_random_tie, color=RANDOM_GREEDY, marker="s", markevery=(1, 4), zorder=2)
    ax.set_xscale("log")
    ax.set_xlim(SIZES[0], SIZES[-1])
    ax.xaxis.set_major_locator(FixedLocator([2, 10, 100, 1000]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["2", "10", "100", "1000"])
    ax.set_xlabel("instance size n")
    style.panel_label(ax, "Upper-triangular hard instance")

    ax = axes[1]
    style.bound_line(ax, WATER_FILLING_RATIO)
    style.bound_line(ax, GREEDY_RATIO)
    for ratios, color, marker, offset in [(wf_rand, style.CLASSICAL, "o", 0), (greedy_rand, RANDOM_GREEDY, "s", 2)]:
        ax.fill_between(DEGREES, ratios.min(1), ratios.max(1), color=color, alpha=0.12, linewidth=0, zorder=2)
        ax.plot(DEGREES, ratios.mean(1), color=color, marker=marker, markevery=(offset, 3), zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(DEGREES[0], DEGREES[-1])
    ax.xaxis.set_major_locator(FixedLocator([0.25, 1, 3, 10, 30]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["0.25", "1", "3", "10", "30"])
    ax.set_xlabel("average extra degree d of an online vertex")
    style.panel_label(ax, "Random graphs, 100 + 100 vertices")

    for ax in axes:
        ax.set_ylim(0.45, 1.02)
        ax.set_ylabel("ALG / OPT")

    handles = [
        Line2D([], [], color=style.CLASSICAL, marker="o", label="water-filling"),
        Line2D([], [], color=style.BASELINE, marker="s", label="greedy, adversarial ties"),
        Line2D([], [], color=RANDOM_GREEDY, marker="s", label="greedy, random ties (mean)"),
        Line2D([], [], color=style.BOUND, linestyle=(0, (4, 3)), linewidth=1.1, label="bounds 1 − 1/e and 1/2"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.2)
    print("saved", style.save(fig, "online_matching.png"))


if __name__ == "__main__":
    main()
