"""Ski rental: renting costs 1 per day, buying costs B, and the number of skiing days N is unknown.

Sources:
- Henry Vu's deck "Online Algorithm and Online Optimization" (online_learning/), slides 2-11: the
  offline cost min(N, B), the deterministic break-even rule with ratio (2B - 1) / B, and the
  randomized buy-day distribution p_i = ((B - 1) / B)^(B - i) alpha / B with
  alpha = 1 / (1 - (1 - 1/B)^B), which tends to e / (e - 1).
- Buchbinder, Jain and Naor, ESA 2007, as restated in Bamas, Maggiori and Svensson, "The
  primal-dual method for learning augmented algorithms", NeurIPS 2020 (arXiv:2010.11632),
  Algorithm 3: the online primal-dual algorithm, which keeps a fractional solution with ratio
  c / (c - 1), c = (1 + 1/B)^B, tending to e / (e - 1). Rounding it online with one uniform draw
  gives a randomized algorithm whose expected cost equals the fractional cost.
- Bamas et al., Algorithm 4 (PDLA) and Theorem 2: with a predicted number of days N_pred and
  lambda in (0, 1], cost <= min{lambda / (1 - e(-lambda)) * S, 1 / (1 - e(-lambda)) * OPT}, where
  e(z) = (1 + 1/B)^(zB) and S is the cost of blindly following the prediction.

Every competitive ratio here is ALG / OPT >= 1 (ski rental minimizes cost).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = [
    "Bounds",
    "FractionalRun",
    "offline_cost",
    "break_even_cost",
    "break_even_ratio",
    "buy_day_distribution",
    "randomized_buy_day_expected_cost",
    "primal_dual",
    "pdla",
    "pdla_cost_curve",
    "pdla_bounds",
    "primal_dual_ratio",
    "follow_prediction_cost",
    "round_online",
]


class Bounds(NamedTuple):
    """Consistency (ratio to the cost of following the prediction) and robustness (ratio to OPT)."""

    consistency: float
    robustness: float


@dataclass(frozen=True)
class FractionalRun:
    """The primal-dual trajectory over N days.

    x[j] is the buy variable after day j + 1, f[j] the fraction rented on day j + 1 and y[j] the
    dual variable of that day's covering constraint x + f_j >= 1.
    """

    B: int
    x: np.ndarray
    f: np.ndarray
    y: np.ndarray

    @property
    def cost(self) -> float:
        """Fractional cost B min(x_N, 1) + sum_j f_j, which is also the expected cost of `round_online`."""
        return float(self.B * min(self.x[-1], 1.0) + self.f.sum())

    @property
    def dual_value(self) -> float:
        return float(self.y.sum())


def offline_cost(N: int, B: int) -> int:
    """OPT = min(N, B): buy on day 1 if N >= B, otherwise rent every day."""
    return min(N, B)


def break_even_cost(N: int, B: int, buy_day: int | None = None) -> int:
    """Rent for buy_day - 1 days and buy on day buy_day (default B, the break-even day).

    Cost is buy_day - 1 + B if skiing lasts until buy_day, otherwise N.
    """
    d = B if buy_day is None else buy_day
    return d - 1 + B if N >= d else N


def break_even_ratio(B: int) -> float:
    """Worst ALG / OPT of the break-even rule, (2B - 1) / B, reached when N = B."""
    return (2 * B - 1) / B


def buy_day_distribution(B: int) -> np.ndarray:
    """Henry's slide 11: buy on day i in 1..B with probability ((B - 1) / B)^(B - i) alpha / B."""
    alpha = 1.0 / (1.0 - (1.0 - 1.0 / B) ** B)
    i = np.arange(1, B + 1)
    return ((B - 1) / B) ** (B - i) * alpha / B


def randomized_buy_day_expected_cost(N: int, B: int) -> float:
    """Expected cost of buying on a random day drawn from `buy_day_distribution` (slide 8's sum)."""
    p = buy_day_distribution(B)
    days = np.arange(1, B + 1)
    cost = np.where(days <= N, days - 1 + B, N)
    return float(np.dot(p, cost))


def _fractional(B: int, c: float, dual_step: float, n_days: int) -> FractionalRun:
    """Shared loop of Algorithms 3 and 4: on each day with x < 1 rent 1 - x and grow x."""
    x = np.empty(n_days)
    f = np.zeros(n_days)
    y = np.zeros(n_days)
    current = 0.0
    for j in range(n_days):
        if current < 1.0 - 1e-12:  # the day's constraint x + f_j >= 1 is not yet covered by x alone
            f[j] = 1.0 - current
            current = (1.0 + 1.0 / B) * current + 1.0 / ((c - 1.0) * B)
            y[j] = dual_step
        x[j] = current
    return FractionalRun(B, x, f, y)


def _e(z: float, B: int) -> float:
    """e(z) = (1 + 1/B)^(zB), which tends to exp(z) as B grows."""
    return (1.0 + 1.0 / B) ** (z * B)


def primal_dual(N: int, B: int) -> FractionalRun:
    """Algorithm 3 (Buchbinder, Jain and Naor): c = e(1), every dual variable set to 1."""
    return _fractional(B, _e(1.0, B), 1.0, N)


def pdla(N: int, B: int, N_pred: int, lam: float) -> FractionalRun:
    """Algorithm 4 (PDLA): grow x fast (c = e(lam)) if N_pred >= B, slowly (c = e(1/lam)) otherwise.

    In the slow case each dual variable is lam instead of 1. lam = 1 is exactly Algorithm 3.
    """
    _check_lambda(lam)
    if N_pred >= B:
        return _fractional(B, _e(lam, B), 1.0, N)
    return _fractional(B, _e(1.0 / lam, B), lam, N)


def pdla_cost_curve(B: int, N_pred: int, lam: float, n_max: int) -> np.ndarray:
    """PDLA's fractional cost for every horizon N = 1..n_max (the trajectory does not depend on N)."""
    run = pdla(n_max, B, N_pred, lam)
    return B * np.minimum(run.x, 1.0) + np.cumsum(run.f)


def follow_prediction_cost(N: int, B: int, N_pred: int) -> int:
    """S: buy on day 1 if N_pred >= B, otherwise rent every day.

    Bamas et al. write S with N_pred > B; using >= matches Algorithm 4's own branch, and the two
    only differ when N_pred = B exactly.
    """
    return B if N_pred >= B else N


def pdla_bounds(lam: float, B: int | None = None) -> Bounds:
    """Theorem 2: consistency lam / (1 - e(-lam)) and robustness 1 / (1 - e(-lam)).

    With finite B this uses e(z) = (1 + 1/B)^(zB) and assumes lam B and B / lam are integers, as the
    theorem does; B = None gives the limit with e(z) = exp(z). lam = 1 gives the primal-dual
    ratio c / (c - 1), which tends to e / (e - 1).
    """
    _check_lambda(lam)
    e_minus = np.exp(-lam) if B is None else _e(-lam, B)
    return Bounds(float(lam / (1.0 - e_minus)), float(1.0 / (1.0 - e_minus)))


def primal_dual_ratio(B: int | None = None) -> float:
    """Ratio of Algorithm 3: 1 / (1 - e(-1)), equal to e / (e - 1) when B = None."""
    return pdla_bounds(1.0, B).robustness


def round_online(run: FractionalRun, rng: np.random.Generator) -> int:
    """Randomized ski rental from a fractional run: draw u ~ U(0, 1] once, rent each day until x >= u, then buy.

    On day j the skier rents if x_{j-1} < u and buys at the end of the day once x_j >= u, so the
    expected cost is B min(x_N, 1) + sum_j f_j, the fractional cost.
    """
    u = 1.0 - rng.random()  # in (0, 1]
    before = np.concatenate([[0.0], run.x[:-1]])
    rented = int(np.sum(before < u))
    bought = bool(run.x[-1] >= u)
    return rented + run.B * bought


def _check_lambda(lam: float) -> None:
    if not 0.0 < lam <= 1.0:
        raise ValueError(f"lambda must be in (0, 1], got {lam}")
