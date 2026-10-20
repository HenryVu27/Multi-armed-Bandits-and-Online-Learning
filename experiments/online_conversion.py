"""Empirical competitive ratio of the learning-augmented conversion algorithms against prediction error.

Setting from Sun et al. (NeurIPS 2021) Figure 1: prices in [L, U] = [2, 10], so theta = 5. For each
prediction P and true maximum price V on a geometric grid, the algorithm runs on the p-instance of
Definition 4.3 (prices rise from L to V, then drop to L), the worst instance with maximum V for
any threshold algorithm. The plotted value is the worst OPT / ALG over all pairs with the same
ratio P / V. At P / V = 1 it is the consistency; its maximum is the robustness.

Run: python experiments/online_conversion.py  (writes figures/online_conversion.png)
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
from online.conversion import (  # noqa: E402
    classical_one_way_threshold,
    classical_reservation_price,
    naive_reservation_price,
    offline_optimum,
    one_max_search,
    one_max_search_bounds,
    one_way_trading,
    one_way_trading_bounds,
    p_instance,
    pareto_one_way_threshold,
    pareto_reservation_price,
)

L, U = 2.0, 10.0
LAMBDAS = [0.2, 0.5, 0.8]
NAIVE_LAMBDA = 0.5
GRID_POINTS = 161  # geometric grid, so P / V takes the values q^k
MAX_STEPS = 70  # plot P / V within q^-70 .. q^70, about 0.5 .. 2
N_STEPS = 800


def worst_by_signed_error(run, policy_for_prediction, grid):
    """Worst OPT / ALG for each P / V = q^k, over all V on the grid with P = V q^k also on the grid."""
    prices = p_instance(grid, L, N_STEPS)
    opt = offline_optimum(prices)
    R = np.array([opt / run(prices, policy_for_prediction(P)) for P in grid])  # R[index of P, index of V]
    steps = np.arange(-MAX_STEPS, MAX_STEPS + 1)
    worst = np.array([np.diagonal(R, offset=-k).max() for k in steps])  # entries with index(P) - index(V) = k
    q = grid[1] / grid[0]
    return q**steps, worst


def main() -> None:
    style.apply()
    theta = U / L
    grid = np.geomspace(L, U, GRID_POINTS)
    colors = style.lambda_colors(len(LAMBDAS))
    problems = [
        ("1-max-search", one_max_search, lambda lam: (lambda P: pareto_reservation_price(P, L, U, lam)),
         lambda P: classical_reservation_price(L, U), one_max_search_bounds),
        ("One-way trading", one_way_trading, lambda lam: (lambda P: pareto_one_way_threshold(P, L, U, lam)),
         lambda P: classical_one_way_threshold(L, U), one_way_trading_bounds),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(style.WIDTH, 7.8), sharex=True)
    for ax, (name, run, pareto, classical, bounds) in zip(axes, problems, strict=True):
        for k, (lam, color) in enumerate(zip(LAMBDAS, colors, strict=True)):
            eta, gamma = bounds(lam, theta)
            style.bound_line(ax, eta)
            style.bound_line(ax, gamma)
            x, worst = worst_by_signed_error(run, pareto(lam), grid)
            ax.plot(x, worst, color=color, marker=style.LAMBDA_MARKERS[k], markevery=(k * 5, 20), zorder=3)
            at_one = worst[MAX_STEPS]
            under, over = MAX_STEPS - 29, MAX_STEPS + 7
            print(f"{name:16s} lambda={lam}: OPT/ALG {at_one:.3f} at P=V (eta {eta:.3f}), "
                  f"max {worst.max():.3f} (gamma {gamma:.3f}), "
                  f"{worst[under]:.3f} at P/V={x[under]:.2f}, {worst[over]:.3f} at P/V={x[over]:.2f}")
        best = bounds(1.0, theta).robustness
        x, worst = worst_by_signed_error(run, classical, grid)
        ax.plot(x, worst, color=style.CLASSICAL, zorder=3)
        print(f"{name:16s} classical: max OPT/ALG {worst.max():.3f} (theory {best:.3f})")
        if run is one_max_search:
            x, worst = worst_by_signed_error(run, lambda P: naive_reservation_price(P, L, U, NAIVE_LAMBDA), grid)
            ax.plot(x, worst, color=style.BASELINE, linestyle=(0, (6, 2, 1.5, 2)), zorder=2)
            print(f"{name:16s} naive lambda={NAIVE_LAMBDA}: {worst[MAX_STEPS]:.3f} at P=V, max {worst.max():.3f}")
        style.panel_label(ax, name)
        ax.set_ylabel("worst OPT / ALG")
        ax.set_ylim(0.9, 4.6)
        ax.set_xscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_locator(FixedLocator([0.5, 0.67, 1.0, 1.5, 2.0]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticklabels(["0.5", "0.67", "1", "1.5", "2"])
    axes[-1].set_xlabel("prediction / true maximum price   P / V")

    handles = [Line2D([], [], color=c, marker=m, label=f"λ = {lam}")
               for lam, c, m in zip(LAMBDAS, colors, style.LAMBDA_MARKERS, strict=False)]
    handles += [
        Line2D([], [], color=style.CLASSICAL, label="classical (λ = 1)"),
        Line2D([], [], color=style.BASELINE, linestyle=(0, (6, 2, 1.5, 2)), label=f"naive mix (λ = {NAIVE_LAMBDA})"),
        Line2D([], [], color=style.BOUND, linestyle=(0, (4, 3)), linewidth=1.1, label="theory: η and γ per λ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.2)
    print("saved", style.save(fig, "online_conversion.png"))


if __name__ == "__main__":
    main()
