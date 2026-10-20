"""Empirical competitive ratio of PDLA for ski rental against prediction error.

Bamas, Maggiori and Svensson (NeurIPS 2020), Algorithm 4 and Theorem 2, with buying cost B = 100
(lambda B and B / lambda are integers for every lambda below, as the theorem assumes). For each
ratio r = N_pred / N, the plotted value is the worst expected cost / OPT over seasons
N = 1..10B, where the expected cost of the randomized algorithm equals its fractional cost.
At r = 1 it is the consistency; its maximum is the robustness.

Run: python experiments/online_ski_rental.py  (writes figures/online_ski_rental.png)
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
from online.ski_rental import (  # noqa: E402
    break_even_cost,
    break_even_ratio,
    pdla,
    pdla_bounds,
    pdla_cost_curve,
    primal_dual_ratio,
    round_online,
)

B = 100
LAMBDAS = [0.2, 0.5, 0.8]
N_MAX = 10 * B
RATIOS = np.geomspace(1 / 8, 8, 401)


def worst_by_signed_error(lam: float) -> np.ndarray:
    """PDLA only checks whether N_pred >= B, so the cost for each N follows one of two curves."""
    N = np.arange(1, N_MAX + 1)
    opt = np.minimum(N, B)
    fast = pdla_cost_curve(B, B, lam, N_MAX)  # prediction says buy
    slow = pdla_cost_curve(B, B - 1, lam, N_MAX)  # prediction says rent
    return np.array([np.max(np.where(r * N >= B, fast, slow) / opt) for r in RATIOS])


def follow_prediction_worst() -> np.ndarray:
    N = np.arange(1, N_MAX + 1)
    opt = np.minimum(N, B)
    return np.array([np.max(np.where(r * N >= B, B, N) / opt) for r in RATIOS])


def main() -> None:
    style.apply()
    colors = style.lambda_colors(len(LAMBDAS))
    fig, ax = plt.subplots(figsize=(style.WIDTH, 4.3))
    at_one = np.argmin(np.abs(RATIOS - 1.0))

    for k, (lam, color) in enumerate(zip(LAMBDAS, colors, strict=True)):
        consistency, robustness = pdla_bounds(lam, B)
        style.bound_line(ax, consistency)
        style.bound_line(ax, robustness)
        worst = worst_by_signed_error(lam)
        ax.plot(RATIOS, worst, color=color, marker=style.LAMBDA_MARKERS[k], markevery=(k * 12, 50), zorder=3)
        print(f"lambda={lam}: cost/OPT {worst[at_one]:.3f} at N_pred=N (consistency {consistency:.3f}), "
              f"max {worst.max():.3f} (robustness {robustness:.3f})")

    pd = worst_by_signed_error(1.0)
    ax.plot(RATIOS, pd, color=style.CLASSICAL, zorder=3)
    be = max(break_even_cost(n, B) / min(n, B) for n in range(1, N_MAX + 1))
    ax.plot(RATIOS, np.full(RATIOS.size, be), color=style.CLASSICAL, linestyle=(0, (1.2, 2.2)), linewidth=2.2, zorder=3)
    follow = follow_prediction_worst()
    ax.plot(RATIOS, follow, color=style.BASELINE, linestyle=(0, (6, 2, 1.5, 2)), zorder=2)
    print(f"primal-dual (lambda = 1): max {pd.max():.4f} (theory {primal_dual_ratio(B):.4f}, "
          f"e/(e-1) = {np.e / (np.e - 1):.4f})")
    print(f"break-even: max {be:.3f} (theory {break_even_ratio(B):.3f})")
    at_two = np.argmin(np.abs(RATIOS - 2.0))
    print(f"follow prediction: {follow[at_one]:.3f} at N_pred = N, {follow[at_two]:.3f} at N_pred = 2N")

    rng = np.random.default_rng(2024)
    for N, N_pred, lam in [(150, 400, 0.5), (20, 400, 0.5), (300, 50, 0.5)]:
        run = pdla(N, B, N_pred, lam)
        draws = [round_online(run, rng) for _ in range(20000)]
        print(f"rounding check N={N}, N_pred={N_pred}, lambda={lam}: fractional {run.cost:.2f}, "
              f"mean of 20,000 randomized runs {np.mean(draws):.2f}")

    ax.set_xscale("log")
    ax.set_xlim(RATIOS[0], RATIOS[-1])
    ax.xaxis.set_major_locator(FixedLocator([1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    ax.set_ylim(0.9, 6.0)
    ax.set_xlabel("predicted / actual skiing days   N_pred / N")
    ax.set_ylabel("worst expected cost / OPT")

    handles = [Line2D([], [], color=c, marker=m, label=f"PDLA, λ = {lam}")
               for lam, c, m in zip(LAMBDAS, colors, style.LAMBDA_MARKERS, strict=False)]
    handles += [
        Line2D([], [], color=style.CLASSICAL, label="primal-dual (λ = 1)"),
        Line2D([], [], color=style.CLASSICAL, linestyle=(0, (1.2, 2.2)), linewidth=2.2, label="break-even"),
        Line2D([], [], color=style.BASELINE, linestyle=(0, (6, 2, 1.5, 2)), label="follow prediction"),
        Line2D([], [], color=style.BOUND, linestyle=(0, (4, 3)), linewidth=1.1, label="theory: C and R per λ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.4,
               columnspacing=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    print("saved", style.save(fig, "online_ski_rental.png"))


if __name__ == "__main__":
    main()
