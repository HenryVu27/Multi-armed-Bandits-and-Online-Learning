"""Shared matplotlib style for the online-algorithm experiment figures.

Figures are 5.5 inches wide so that 12-point text stays readable when a figure is shown at half
the width of a web page. Two-panel figures stack vertically for the same reason.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402

LEARNED_LOW = "#2f6ea8"  # learning-augmented, low lambda
LEARNED_HIGH = "#23876a"  # learning-augmented, high lambda
CLASSICAL = "#b84a32"
BASELINE = "#c0648e"
BOUND = "#8a8a8a"
INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e6e6e6"

LAMBDA_MARKERS = ["o", "s", "^", "D"]
WIDTH = 5.5
FIGURES = Path(__file__).resolve().parents[1] / "figures"


def lambda_colors(n: int) -> list[str]:
    """n shades from LEARNED_LOW (low lambda) to LEARNED_HIGH (high lambda)."""
    a, b = np.array(to_rgb(LEARNED_LOW)), np.array(to_rgb(LEARNED_HIGH))
    if n == 1:
        return [LEARNED_LOW]
    return [to_hex(a + (b - a) * t) for t in np.linspace(0.0, 1.0, n)]


def apply() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#bdbdbd",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 10.5,
            "legend.labelcolor": INK,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "lines.markersize": 6,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def bound_line(ax, y: float, label: str | None = None, x_text: float = 1.0) -> None:
    """Gray dashed horizontal line for a theoretical bound, with an optional small label at its right end."""
    ax.axhline(y, color=BOUND, linestyle=(0, (4, 3)), linewidth=1.1, zorder=1)
    if label:
        ax.annotate(
            label,
            xy=(x_text, y),
            xycoords=("axes fraction", "data"),
            xytext=(3, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9.5,
            color=MUTED,
            annotation_clip=False,
        )


def panel_label(ax, text: str) -> None:
    ax.text(0.0, 1.03, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=12, color=INK, weight="bold")


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    return path
