"""Online fractional knapsack with capacity 1 and value densities v_t / w_t in [L, U].

Sources: Henry Vu's deck "Online Algorithm and Online Optimization" (online_learning/), slides
23-33, which derive the threshold with the online primal-dual method, and Zhou, Chakrabarty and
Lukose, "Budget constrained bidding in keyword auctions and online knapsack problems", WINE 2008.

The threshold at utilization z is Psi(z) = L for z in [0, beta] and (U e / L)^z (L / e) for z in
[beta, 1], with beta = 1 / (1 + ln(U / L)). It satisfies Psi(z) = (1 + ln(U / L)) * integral of Psi
from 0 to z on [beta, 1], so the algorithm is (1 + ln(U / L))-competitive, and no online algorithm
does better (slide 33). Zhou et al. accept whole items and need every weight to be small; here an
item is accepted fractionally, up to the utilization where Psi reaches its density, which gives
the same ratio for items of any size.

Ratios are OPT / ALG >= 1 (knapsack maximizes value).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "KnapsackResult",
    "zcl_beta",
    "zcl_threshold",
    "zcl_utilization_at",
    "zcl_competitive_ratio",
    "threshold_knapsack",
    "greedy_knapsack",
    "offline_fractional_knapsack",
    "ramp_instance",
    "flood_instance",
    "random_instance",
]


@dataclass(frozen=True)
class KnapsackResult:
    """Fraction of each item's weight taken (in [0, 1]) and the total value collected."""

    fractions: np.ndarray
    value: float


def zcl_beta(L: float, U: float) -> float:
    """beta = 1 / (1 + ln(U / L)): below this utilization every item is accepted."""
    return 1.0 / (1.0 + np.log(U / L))


def zcl_competitive_ratio(L: float, U: float) -> float:
    """1 + ln(U / L), the ratio OPT / ALG of the threshold algorithm."""
    return 1.0 + np.log(U / L)


def zcl_threshold(z: np.ndarray | float, L: float, U: float) -> np.ndarray:
    """Psi(z) = L on [0, beta] and (U e / L)^z (L / e) on [beta, 1]."""
    z = np.asarray(z, dtype=float)
    return np.where(z <= zcl_beta(L, U), L, (U * np.e / L) ** z * (L / np.e))


def zcl_utilization_at(density: np.ndarray | float, L: float, U: float) -> np.ndarray:
    """sup {z : Psi(z) <= density} = (1 + ln(density / L)) / (1 + ln(U / L)), clipped to [0, 1]."""
    density = np.asarray(density, dtype=float)
    z = (1.0 + np.log(np.maximum(density, 1e-300) / L)) / (1.0 + np.log(U / L))
    return np.where(density < L, 0.0, np.clip(z, 0.0, 1.0))


def threshold_knapsack(values: np.ndarray, weights: np.ndarray, L: float, U: float) -> KnapsackResult:
    """Zhou-Chakrabarty-Lukose threshold rule, fractional version.

    Item t raises the utilization z to min(z + w_t, sup {z' : Psi(z') <= v_t / w_t}), so every unit
    of weight taken has density at least the threshold where it lands (slide 30's x_t rule).
    """
    values, weights = _as_items(values, weights)
    target = zcl_utilization_at(values / weights, L, U)
    fractions = np.zeros(values.size)
    z = 0.0
    for t in range(values.size):
        take = min(weights[t], max(0.0, target[t] - z), 1.0 - z)
        fractions[t] = take / weights[t]
        z += take
    return KnapsackResult(fractions, float(np.dot(fractions, values)))


def greedy_knapsack(values: np.ndarray, weights: np.ndarray) -> KnapsackResult:
    """Baseline: take as much of every item as still fits (a threshold of L everywhere)."""
    values, weights = _as_items(values, weights)
    fractions = np.zeros(values.size)
    z = 0.0
    for t in range(values.size):
        take = min(weights[t], 1.0 - z)
        fractions[t] = take / weights[t]
        z += take
    return KnapsackResult(fractions, float(np.dot(fractions, values)))


def offline_fractional_knapsack(values: np.ndarray, weights: np.ndarray) -> KnapsackResult:
    """Offline optimum: fill the knapsack in decreasing order of density (exact for the fractional problem)."""
    values, weights = _as_items(values, weights)
    order = np.argsort(-values / weights, kind="stable")
    fractions = np.zeros(values.size)
    room = 1.0
    for t in order:
        take = min(weights[t], room)
        fractions[t] = take / weights[t]
        room -= take
        if room <= 0.0:
            break
    return KnapsackResult(fractions, float(np.dot(fractions, values)))


def ramp_instance(p: float, L: float, n_steps: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """Worst case for the threshold rule: items of weight 1 whose densities rise from L to p.

    Each item could fill the knapsack alone, so the threshold rule fills exactly up to
    sup {z : Psi(z) <= density} at each step and ends with about p / (1 + ln(U / L)), while OPT
    packs the last item at density p.
    """
    densities = np.linspace(L, p, n_steps)
    weights = np.ones(n_steps)
    return densities * weights, weights


def flood_instance(L: float, U: float) -> tuple[np.ndarray, np.ndarray]:
    """Worst case for greedy: weight 1 at density L fills the knapsack before weight 1 at density U."""
    weights = np.array([1.0, 1.0])
    return np.array([L, U]) * weights, weights


def random_instance(
    rng: np.random.Generator,
    L: float,
    U: float,
    n_items: int = 100,
    total_weight: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Items with log-uniform densities in [L, U] and Dirichlet weights summing to `total_weight`."""
    densities = np.exp(rng.uniform(np.log(L), np.log(U), n_items))
    weights = total_weight * rng.dirichlet(np.full(n_items, 0.5))
    weights = np.maximum(weights, 1e-12)
    return densities * weights, weights


def _as_items(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape != weights.shape or values.ndim != 1:
        raise ValueError("values and weights must be 1-D arrays of the same length")
    if np.any(weights <= 0):
        raise ValueError("weights must be positive")
    return values, weights
