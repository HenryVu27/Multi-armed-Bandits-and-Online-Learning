"""Online conversion: 1-max-search and one-way trading with prices in [L, U].

Sources: Henry Vu's deck "Pareto-Optimal Learning-Augmented Algorithms for Online Conversion
Problems" (online_learning/) and Sun, Lee, Hajiesmaili, Wierman and Tsang, "Pareto-optimal
learning-augmented algorithms for online conversion problems", NeurIPS 2021, arXiv:2109.01556.

- Algorithm 1, the online threshold-based algorithm (OTA): trade only while the price is at least
  the threshold phi(w) at the current utilization w (the fraction of the asset already traded).
- Classical thresholds (Section 2): the reservation price sqrt(LU) for 1-max-search, ratio
  sqrt(theta), and phi(w) = L + (alpha* L - L) exp(alpha* w) for one-way trading, ratio
  alpha* = 1 + W((theta - 1) / e), where W is the Lambert W function and theta = U / L.
- Learning-augmented thresholds that take a prediction P of the maximum price and a robustness
  parameter lambda in [0, 1]: equation (6) for 1-max-search (Theorem 3.5) and equations (8)-(10)
  for one-way trading (Theorem 3.6), with consistency eta(lambda) and robustness gamma(lambda)
  from equations (4) and (7). Both trade-offs are Pareto-optimal (Theorems 4.1 and 4.2).
- Proposition 3.1's naive reservation price lambda sqrt(LU) + (1 - lambda) P, as a baseline.

Every competitive ratio here is OPT / ALG >= 1, as in the paper (conversion maximizes return).
Each instance ends with a compulsory conversion: whatever is left after step N - 1 is traded at
the last price v_N.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import lambertw

__all__ = [
    "Bounds",
    "Piece",
    "Threshold",
    "alpha_star",
    "classical_reservation_price",
    "classical_one_way_threshold",
    "one_max_search_bounds",
    "one_way_trading_bounds",
    "naive_one_max_search_bounds",
    "pareto_reservation_price",
    "naive_reservation_price",
    "pareto_one_way_threshold",
    "one_max_search",
    "one_way_trades",
    "one_way_trading",
    "offline_optimum",
    "p_instance",
    "random_instances",
    "reservation_price_ratio",
    "threshold_ratio",
]


class Bounds(NamedTuple):
    """Consistency (ratio with a perfect prediction) and robustness (worst ratio over all predictions)."""

    consistency: float
    robustness: float


# ---------------------------------------------------------------------------------------------
# Theoretical ratios
# ---------------------------------------------------------------------------------------------


def alpha_star(theta: float) -> float:
    """Optimal one-way trading ratio alpha* = 1 + W((theta - 1) / e) (Section 2)."""
    return float(1.0 + lambertw((theta - 1.0) / np.e).real)


def one_max_search_bounds(lam: float, theta: float) -> Bounds:
    """Equation (4): gamma = [sqrt((1 - lam)^2 + 4 lam theta) - (1 - lam)] / (2 lam), eta = theta / gamma.

    eta and gamma solve eta = theta / gamma and eta = lam gamma + 1 - lam (equation (5)), so lam = 0
    gives (1, theta) and lam = 1 gives (sqrt(theta), sqrt(theta)).
    """
    _check_lambda(lam)
    if lam == 0.0:
        return Bounds(1.0, float(theta))
    gamma = (np.sqrt((1.0 - lam) ** 2 + 4.0 * lam * theta) - (1.0 - lam)) / (2.0 * lam)
    return Bounds(float(theta / gamma), float(gamma))


def one_way_trading_bounds(lam: float, theta: float) -> Bounds:
    """Equation (7): gamma = alpha* + (1 - lam)(theta - alpha*) and the matching Pareto consistency.

    eta = theta / [theta / gamma + (theta - 1)(1 - ln((theta - 1) / (gamma - 1)) / gamma)], the
    lower bound of Theorem 4.2. lam = 0 gives (1, theta) and lam = 1 gives (alpha*, alpha*).
    """
    _check_lambda(lam)
    a = alpha_star(theta)
    gamma = a + (1.0 - lam) * (theta - a)
    eta = theta / (theta / gamma + (theta - 1.0) * (1.0 - np.log((theta - 1.0) / (gamma - 1.0)) / gamma))
    return Bounds(float(eta), float(gamma))


def naive_one_max_search_bounds(lam: float, theta: float) -> Bounds:
    """Proposition 3.1: the naive price is sqrt(theta)-consistent (1 if lam = 0) and
    (lam sqrt(theta) + (1 - lam) theta)-robust."""
    _check_lambda(lam)
    consistency = 1.0 if lam == 0.0 else float(np.sqrt(theta))
    return Bounds(consistency, float(lam * np.sqrt(theta) + (1.0 - lam) * theta))


# ---------------------------------------------------------------------------------------------
# Threshold functions
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Piece:
    """phi(w) = L + (value_at_start - L) exp(rate (w - start)) on [start, end); rate 0 is a flat piece."""

    start: float
    end: float
    value_at_start: float
    rate: float

    def value(self, w: np.ndarray, L: float) -> np.ndarray:
        return L + (self.value_at_start - L) * np.exp(self.rate * (w - self.start))

    def integral(self, a: np.ndarray, b: np.ndarray, L: float) -> np.ndarray:
        """Integral of phi over [a, b] inside this piece."""
        if self.rate == 0.0:
            return self.value_at_start * (b - a)
        scale = (self.value_at_start - L) / self.rate
        return L * (b - a) + scale * (np.exp(self.rate * (b - self.start)) - np.exp(self.rate * (a - self.start)))


@dataclass(frozen=True)
class Threshold:
    """A non-decreasing, piecewise threshold phi: [0, 1] -> [L, U] built from `Piece`s in order."""

    L: float
    pieces: tuple[Piece, ...]

    def __call__(self, w: np.ndarray | float) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        out = np.full(w.shape, np.nan)
        for piece in self.pieces:  # pieces are ordered, so later pieces take over (right-continuous)
            if piece.end > piece.start:
                out = np.where(w >= piece.start, piece.value(w, self.L), out)
        return out

    def utilization_at(self, price: np.ndarray | float) -> np.ndarray:
        """sup {u in [0, 1] : phi(u) <= price}, and 0 if phi(0) > price.

        Line 3 of Algorithm 1 trades from the current utilization up to this point, since the
        objective v x - integral(phi) is concave in x with derivative v - phi(w + x).
        """
        v = np.asarray(price, dtype=float)
        u = np.zeros(v.shape)
        for piece in self.pieces:
            if piece.end <= piece.start:
                continue
            reached = v >= piece.value_at_start
            if piece.rate == 0.0:
                inside = np.full(v.shape, piece.end)
            else:
                ratio = np.maximum(v - self.L, 1e-300) / (piece.value_at_start - self.L)
                inside = np.clip(piece.start + np.log(ratio) / piece.rate, piece.start, piece.end)
            u = np.where(reached, inside, u)
        return u

    def integral(self, w: np.ndarray | float) -> np.ndarray:
        """Integral of phi from 0 to w."""
        w = np.asarray(w, dtype=float)
        total = np.zeros(w.shape)
        for piece in self.pieces:
            if piece.end <= piece.start:
                continue
            b = np.clip(w, piece.start, piece.end)
            total = total + piece.integral(np.full(w.shape, piece.start), b, self.L)
        return total


def classical_reservation_price(L: float, U: float) -> float:
    """Reservation price sqrt(LU) of El-Yaniv et al.; OTA with it is sqrt(theta)-competitive."""
    return float(np.sqrt(L * U))


def classical_one_way_threshold(L: float, U: float) -> Threshold:
    """phi(w) = L + (alpha* L - L) exp(alpha* w) for w in [0, 1]; it ends at phi(1) = U."""
    a = alpha_star(U / L)
    return Threshold(L, (Piece(0.0, 1.0, a * L, a),))


def pareto_reservation_price(P: float, L: float, U: float, lam: float) -> float:
    """Equation (6), the reservation price given a prediction P of the maximum price.

    P in [L, L eta)       -> L eta
    P in [L eta, L gamma) -> lam L gamma + (1 - lam) P / eta
    P in [L gamma, U]     -> L gamma
    """
    _check_prediction(P, L, U)
    eta, gamma = one_max_search_bounds(lam, U / L)
    if P < L * eta:
        return float(L * eta)
    if P < L * gamma:
        return float(lam * L * gamma + (1.0 - lam) * P / eta)
    return float(L * gamma)


def naive_reservation_price(P: float, L: float, U: float, lam: float) -> float:
    """Proposition 3.1's linear combination lam sqrt(LU) + (1 - lam) P."""
    _check_prediction(P, L, U)
    _check_lambda(lam)
    return float(lam * np.sqrt(L * U) + (1.0 - lam) * P)


def pareto_one_way_threshold(P: float, L: float, U: float, lam: float) -> Threshold:
    """Equations (8)-(10), the one-way trading threshold given a prediction P of the maximum price.

    With eta, gamma from equation (7), beta and M solve equation (9). For P < M the threshold is
    (8a): an eta-rate exponential on [0, beta) and a gamma-rate exponential up to U on [beta, 1].
    For P >= M it is (8b): a gamma-rate exponential up to M1, a flat piece at M1, an eta-rate
    exponential from M1 up to P, and a gamma-rate exponential up to U, with M1, beta1, beta1' and
    beta2 solving equation (10).

    lam = 1 returns the classical threshold (with eta = gamma = alpha* both pieces of (8a) are the
    classical curve) and lam = 0 returns a flat threshold at P (convert everything at the first
    price that reaches the prediction), the limits of (8)-(10) at those values.
    """
    _check_prediction(P, L, U)
    _check_lambda(lam)
    theta = U / L
    if lam == 1.0:
        return classical_one_way_threshold(L, U)
    if lam == 0.0:
        return Threshold(L, (Piece(0.0, 1.0, float(P), 0.0),))

    eta, gamma = one_way_trading_bounds(lam, theta)

    # Equation (9): M = L + (eta L - L) exp(eta beta) and M gamma / eta = L + (U - L) exp(gamma (beta - 1)).
    def eq9(beta: float) -> float:
        M = L + (eta - 1.0) * L * np.exp(eta * beta)
        return M * gamma / eta - L - (U - L) * np.exp(gamma * (beta - 1.0))

    beta = brentq(eq9, 0.0, 1.0, xtol=1e-15, rtol=1e-15)
    M = L + (eta - 1.0) * L * np.exp(eta * beta)
    if P < M:
        return Threshold(
            L,
            (
                Piece(0.0, beta, eta * L, eta),
                Piece(beta, 1.0, M * gamma / eta, gamma),
            ),
        )

    # Equation (10), fourth line: beta2 is explicit in P.
    top = min(P * gamma / eta, U)
    beta2 = 1.0 + np.log((top - L) / (U - L)) / gamma

    def beta1_of(M1: float) -> float:  # first line of (10)
        return np.log((max(M1 / L, gamma) - 1.0) / (gamma - 1.0)) / gamma

    def beta1_prime_from_flat(M1: float) -> float:  # second line of (10), solved for beta1'
        b1 = beta1_of(M1)
        head = Piece(0.0, b1, gamma * L, gamma).integral(0.0, b1, L)
        return (M1 / eta - head + b1 * M1 - L) / (M1 - L)

    def beta1_prime_from_rise(M1: float) -> float:  # third line of (10), solved for beta1'
        return beta2 - np.log((P - L) / (M1 - L)) / eta

    def residual(M1: float) -> float:
        return beta1_prime_from_flat(M1) - beta1_prime_from_rise(M1)

    # The root with beta1' >= beta1 >= 0 lies in [eta L, P]: at P = M it is M1 = eta L, where (8b)
    # reduces to the eta-curve of (8a), and it rises to M1 = U at P = U. A second root below eta L
    # has beta1' < 0 and is not a valid threshold.
    lo, hi = eta * L, float(P)
    tol = 1e-12
    if abs(residual(hi)) <= tol:
        M1 = hi
    elif residual(lo) <= tol:
        M1 = lo
    else:
        M1 = brentq(residual, lo, hi, xtol=1e-15, rtol=1e-15)

    b1 = beta1_of(M1)
    b1p = float(np.clip(beta1_prime_from_rise(M1), b1, beta2))
    return Threshold(
        L,
        (
            Piece(0.0, b1, gamma * L, gamma),
            Piece(b1, b1p, M1, 0.0),
            Piece(b1p, beta2, M1, eta),
            Piece(beta2, 1.0, top, gamma),
        ),
    )


# ---------------------------------------------------------------------------------------------
# Running the algorithms
# ---------------------------------------------------------------------------------------------


def one_max_search(prices: np.ndarray, reservation_price: float) -> np.ndarray:
    """1-max-search with OTA: sell the whole unit at the first price >= the reservation price.

    `prices` has shape (..., N); each row is one instance. If no price reaches the reservation
    price, the unit is sold at the last price v_N. Returns the return of each instance.
    """
    prices = np.asarray(prices, dtype=float)
    hit = prices >= reservation_price
    first = np.argmax(hit, axis=-1)
    first = np.where(hit.any(axis=-1), first, prices.shape[-1] - 1)
    return np.take_along_axis(prices, first[..., None], axis=-1)[..., 0]


def one_way_trades(prices: np.ndarray, threshold: Threshold) -> np.ndarray:
    """Amount traded at each step by Algorithm 1 for one-way trading, shape (..., N).

    Step n trades up to utilization sup{u : phi(u) <= v_n}, so the utilization is the running
    maximum of that quantity. The last step converts whatever is left.
    """
    prices = np.asarray(prices, dtype=float)
    target = threshold.utilization_at(prices[..., :-1])
    w = np.maximum.accumulate(target, axis=-1)
    before = np.concatenate([np.zeros(w.shape[:-1] + (1,)), w], axis=-1)
    trades = np.diff(before, axis=-1)
    remainder = 1.0 - before[..., -1:]
    return np.concatenate([trades, remainder], axis=-1)


def one_way_trading(prices: np.ndarray, threshold: Threshold) -> np.ndarray:
    """Return of one-way trading with OTA_phi on each instance (rows of `prices`)."""
    prices = np.asarray(prices, dtype=float)
    return np.sum(prices * one_way_trades(prices, threshold), axis=-1)


def offline_optimum(prices: np.ndarray) -> np.ndarray:
    """The offline optimum of both problems: trade everything at the maximum price."""
    return np.max(np.asarray(prices, dtype=float), axis=-1)


# ---------------------------------------------------------------------------------------------
# Instances and continuous-limit ratios
# ---------------------------------------------------------------------------------------------


def p_instance(V: np.ndarray | float, L: float, n_steps: int = 1000) -> np.ndarray:
    """Definition 4.3: prices rise linearly from L to V over n_steps - 1 steps, then drop to L.

    For any threshold algorithm this is the worst instance with maximum price V: it trades at
    (nearly) the threshold itself and converts the rest at L. Vector V gives one row per value.
    """
    V = np.atleast_1d(np.asarray(V, dtype=float))
    ramp = L + (V[:, None] - L) * np.linspace(0.0, 1.0, n_steps - 1)[None, :]
    prices = np.concatenate([ramp, np.full((V.size, 1), L)], axis=1)
    return prices


def random_instances(
    rng: np.random.Generator,
    L: float,
    U: float,
    n_instances: int,
    n_steps: int = 200,
    volatility: float = 0.08,
    crash_probability: float = 0.3,
) -> np.ndarray:
    """Geometric random walks clipped to [L, U]; with some probability the last price crashes to L."""
    start = np.exp(rng.uniform(np.log(L), np.log(U), size=(n_instances, 1)))
    steps = rng.normal(0.0, volatility, size=(n_instances, n_steps - 1))
    log_path = np.log(start) + np.concatenate([np.zeros((n_instances, 1)), np.cumsum(steps, axis=1)], axis=1)
    prices = np.clip(np.exp(log_path), L, U)
    crash = rng.random(n_instances) < crash_probability
    prices[crash, -1] = L
    return prices


def reservation_price_ratio(V: np.ndarray | float, reservation_price: float, L: float) -> np.ndarray:
    """Worst OPT / ALG over instances with maximum price V for a reservation price (continuous limit)."""
    V = np.asarray(V, dtype=float)
    return np.where(V >= reservation_price, V / reservation_price, V / L)


def threshold_ratio(V: np.ndarray | float, threshold: Threshold) -> np.ndarray:
    """Worst OPT / ALG over instances with maximum price V for OTA_phi (continuous limit).

    From the proof of Theorem 3.4: ALG >= integral of phi up to w + (1 - w) L, where w is the
    utilization reached at price V, with equality on the p-instance as its steps shrink.
    """
    V = np.asarray(V, dtype=float)
    w = threshold.utilization_at(V)
    return V / (threshold.integral(w) + (1.0 - w) * threshold.L)


def _check_lambda(lam: float) -> None:
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lambda must be in [0, 1], got {lam}")


def _check_prediction(P: float, L: float, U: float) -> None:
    if not 0.0 < L < U:
        raise ValueError(f"need 0 < L < U, got L={L}, U={U}")
    if not L <= P <= U:
        raise ValueError(f"prediction must be in [L, U] = [{L}, {U}], got {P}")
