"""The value of each action for an applicant with fraud probability p.

Every action's expected value is a straight line in p:

    EV(p) = (1 − p)·G − p·B − k

where G is what a genuine applicant is worth to you after that action, B is what a
fraudster still costs you after it, and k is the fixed cost of taking it:

    action    G              B              k
    approve   V              L              0
    verify    (1 − a_g)·V    (1 − a_f)·L    s
    review    (1 − f)·V      (1 − r)·L      c
    decline   0              0              0

The best action at p is the highest line. Because the lines are straight, each action
wins over one unbroken stretch of p, so decisions fall into bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTIONS: tuple[str, ...] = ("approve", "verify", "review", "decline")
APPROVE, VERIFY, REVIEW, DECLINE = range(4)
NON_REVIEW = np.array([APPROVE, VERIFY, DECLINE])


@dataclass(frozen=True)
class Economics:
    """The eight numbers that price every decision (see config/assumptions.yaml)."""

    V: float  # value of a genuine customer onboarded
    L: float  # cost of a fraudster onboarded
    c: float  # cost of a human review
    s: float  # cost of a verification check
    r: float  # share of fraudsters a reviewer stops
    f: float  # share of genuine applicants a reviewer wrongly rejects
    a_g: float  # share of genuine applicants who abandon at verification
    a_f: float  # share of fraudsters who abandon or fail at verification

    def keep_good(self) -> np.ndarray:
        """Share of genuine applicants who end up onboarded, by action."""
        return np.array([1.0, 1.0 - self.a_g, 1.0 - self.f, 0.0])

    def keep_fraud(self) -> np.ndarray:
        """Share of fraudsters who end up onboarded, by action."""
        return np.array([1.0, 1.0 - self.a_f, 1.0 - self.r, 0.0])

    def fixed_cost(self) -> np.ndarray:
        return np.array([0.0, self.s, self.c, 0.0])

    def value_if_good(self) -> np.ndarray:
        """Value of each action when the applicant turns out to be genuine (G − k)."""
        return self.keep_good() * self.V - self.fixed_cost()

    def value_if_fraud(self) -> np.ndarray:
        """Value of each action when the applicant turns out to be a fraudster (−B − k)."""
        return -self.keep_fraud() * self.L - self.fixed_cost()

    def intercepts(self) -> np.ndarray:
        """EV at p = 0."""
        return self.value_if_good()

    def slopes(self) -> np.ndarray:
        """d EV / dp = −(G + B). Always ≤ 0: more risk never makes an action more valuable."""
        return self.value_if_fraud() - self.value_if_good()


def expected_values(p: np.ndarray | float, econ: Economics) -> np.ndarray:
    """Expected £ value of each action. Shape (len(p), 4), columns in ACTIONS order."""
    p = np.atleast_1d(np.asarray(p, dtype=float))
    return np.outer(1.0 - p, econ.value_if_good()) + np.outer(p, econ.value_if_fraud())


def best_action(p: np.ndarray | float, econ: Economics) -> np.ndarray:
    """Index of the highest line at each p, ignoring review capacity."""
    return np.argmax(expected_values(p, econ), axis=1)


def best_non_review(p: np.ndarray | float, econ: Economics) -> np.ndarray:
    """What to do with a case when there is no review slot for it."""
    ev = expected_values(p, econ)[:, NON_REVIEW]
    return NON_REVIEW[np.argmax(ev, axis=1)]


def review_gain(p: np.ndarray | float, econ: Economics) -> np.ndarray:
    """£ a review adds over the best alternative. Concave in p, so its peak sits mid-band."""
    ev = expected_values(p, econ)
    return ev[:, REVIEW] - ev[:, NON_REVIEW].max(axis=1)


def approve_decline_breakeven(econ: Economics) -> float:
    """The p where approving and declining are worth the same: V / (V + L)."""
    return econ.V / (econ.V + econ.L)


@dataclass(frozen=True)
class Band:
    action: int
    start: float
    end: float

    @property
    def name(self) -> str:
        return ACTIONS[self.action]


def envelope(econ: Economics, lo: float = 0.0, hi: float = 1.0) -> list[Band]:
    """The bands of p in which each action is best, found exactly by walking the upper envelope.

    Starting from the best line at `lo`, repeatedly jump to the first line with a gentler slope
    that overtakes the current one. Slopes only ever get gentler as p rises, which is why each
    action owns at most one band.
    """
    a, b = econ.intercepts(), econ.slopes()
    p = lo
    # Best line at `lo`; on a tie take the gentler slope, because it wins just to the right.
    current = max(range(4), key=lambda j: (a[j] + b[j] * p, b[j], -j))
    bands: list[Band] = []
    while p < hi:
        next_p, nxt = hi, None
        for j in range(4):
            if b[j] <= b[current]:
                continue
            cross = (a[current] - a[j]) / (b[j] - b[current])
            if cross <= p + 1e-15:
                continue
            if cross < next_p - 1e-15 or (
                nxt is not None and abs(cross - next_p) <= 1e-15 and b[j] > b[nxt]
            ):
                next_p, nxt = cross, j
        bands.append(Band(current, p, min(next_p, hi)))
        if nxt is None or next_p >= hi:
            break
        p, current = next_p, nxt
    return bands
