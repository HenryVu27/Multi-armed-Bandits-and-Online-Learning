"""Empirical competitive ratio of the Zhou-Chakrabarty-Lukose threshold rule across theta = U / L.

For each theta (L = 1, U = theta) the threshold rule and greedy run on two adversarial families
(the density ramp and the low-then-high flood from online/knapsack.py) and on 300 random
instances. The worst OPT / ALG over the adversarial families is compared with the bound
1 + ln(theta); the median over random instances shows typical behavior.

Run: python experiments/online_knapsack.py  (writes figures/online_knapsack.png)
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
from online.knapsack import (  # noqa: E402
    flood_instance,
    greedy_knapsack,
    offline_fractional_knapsack,
    ramp_instance,
    random_instance,
    threshold_knapsack,
    zcl_competitive_ratio,
)

THETAS = np.geomspace(1.0, 1000.0, 31)[1:]  # 10^(k/10), so 10, 100 and 1000 are on the grid
N_RANDOM = 300


def adversarial_instances(L: float, U: float):
    for p in np.geomspace(L, U, 12):
        yield ramp_instance(p, L, n_steps=1000)
    yield flood_instance(L, U)


def main() -> None:
    style.apply()
    rng = np.random.default_rng(7)
    rows = []
    for theta in THETAS:
        L, U = 1.0, float(theta)
        worst = {"threshold": 0.0, "greedy": 0.0}
        for values, weights in adversarial_instances(L, U):
            opt = offline_fractional_knapsack(values, weights).value
            worst["threshold"] = max(worst["threshold"], opt / threshold_knapsack(values, weights, L, U).value)
            worst["greedy"] = max(worst["greedy"], opt / greedy_knapsack(values, weights).value)
        random_ratios = {"threshold": [], "greedy": []}
        for _ in range(N_RANDOM):
            values, weights = random_instance(rng, L, U, n_items=int(rng.integers(10, 80)),
                                              total_weight=rng.uniform(1.0, 5.0))
            opt = offline_fractional_knapsack(values, weights).value
            random_ratios["threshold"].append(opt / threshold_knapsack(values, weights, L, U).value)
            random_ratios["greedy"].append(opt / greedy_knapsack(values, weights).value)
        rows.append((theta, worst["threshold"], worst["greedy"], np.median(random_ratios["threshold"]),
                     np.median(random_ratios["greedy"]), max(random_ratios["threshold"])))
    rows = np.array(rows)
    theta, zcl_worst, greedy_worst, zcl_median, greedy_median, zcl_random_max = rows.T
    bound = np.array([zcl_competitive_ratio(1.0, t) for t in theta])
    assert np.all(zcl_worst <= bound * (1 + 1e-9)) and np.all(zcl_random_max <= bound * (1 + 1e-9))
    for t in [10.0, 100.0, 1000.0]:
        i = np.argmin(np.abs(theta - t))
        print(f"theta={theta[i]:7.1f}: bound {bound[i]:.3f}, threshold worst {zcl_worst[i]:.3f}, "
              f"random max {zcl_random_max[i]:.3f}, median {zcl_median[i]:.3f}; "
              f"greedy worst {greedy_worst[i]:.1f}, median {greedy_median[i]:.3f}")

    fig, ax = plt.subplots(figsize=(style.WIDTH, 4.3))
    ax.plot(theta, bound, color=style.BOUND, linestyle=(0, (4, 3)), linewidth=1.3, zorder=4)
    ax.plot(theta, theta, color=style.BOUND, linestyle=(0, (1, 2.5)), linewidth=1.3, zorder=4)
    ax.plot(theta, zcl_worst, color=style.CLASSICAL, marker="o", markevery=3, zorder=3)
    ax.plot(theta, zcl_median, color=style.CLASSICAL, linewidth=1.4, alpha=0.55, zorder=3)
    ax.plot(theta, greedy_worst, color=style.BASELINE, marker="s", markevery=(1, 3), zorder=2)
    ax.plot(theta, greedy_median, color=style.BASELINE, linewidth=1.4, alpha=0.55, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(theta[0], theta[-1])
    ax.set_ylim(0.9, 1300)
    ticks = [1, 2, 5, 10, 100, 1000]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel("price range   θ = U / L")
    ax.set_ylabel("OPT / ALG")

    handles = [
        Line2D([], [], color=style.CLASSICAL, marker="o", label="threshold, worst case"),
        Line2D([], [], color=style.CLASSICAL, linewidth=1.4, alpha=0.55, label="threshold, random median"),
        Line2D([], [], color=style.BOUND, linestyle=(0, (4, 3)), linewidth=1.3, label="bound 1 + ln θ"),
        Line2D([], [], color=style.BASELINE, marker="s", label="greedy, worst case"),
        Line2D([], [], color=style.BASELINE, linewidth=1.4, alpha=0.55, label="greedy, random median"),
        Line2D([], [], color=style.BOUND, linestyle=(0, (1, 2.5)), linewidth=1.3, label="θ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    print("saved", style.save(fig, "online_knapsack.png"))


if __name__ == "__main__":
    main()
