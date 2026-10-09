"""How many applicants an A/B test of the verify step needs (see docs/experiment_plan.md)."""

from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist


def sample_size_per_arm(
    p_control: float, lift: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Applicants per arm to detect an absolute change `lift` in a rate, two-sided test.

    Standard two-proportion formula with a pooled variance under the null.
    """
    p_treat = p_control + lift
    if not (0 < p_control < 1 and 0 < p_treat < 1) or lift == 0:
        raise ValueError("rates must lie strictly between 0 and 1, and lift must be non-zero")
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    p_bar = (p_control + p_treat) / 2
    num = z_a * sqrt(2 * p_bar * (1 - p_bar)) + z_b * sqrt(
        p_control * (1 - p_control) + p_treat * (1 - p_treat)
    )
    return ceil((num / lift) ** 2)


def weeks_needed(per_arm: int, eligible_per_week: float, arms: int = 2) -> float:
    """How long the test runs if every eligible applicant is randomised."""
    return arms * per_arm / eligible_per_week
