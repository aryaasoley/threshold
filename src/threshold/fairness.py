"""Who bears the extra checks? Genuine applicants by age group, under every policy.

Age is a protected characteristic under the Equality Act 2010, so the fix is not separate
cut-offs by age. Threshold shows the gap on every policy and flags it, the way you would
escalate to Compliance. A bonus check retrains without age to see whether the gap closes;
it often does not, because other features carry the same information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threshold.config import CALIBRATION_MONTH, TEST_MONTHS, Assumptions, load_config
from threshold.histogram import load_hist
from threshold.policy import POLICY_LABELS, compare

GROUPS = ("under 50", "50+")
METRICS = {
    "genuine_checked_rate": "Any extra check or decline",
    "genuine_verify_rate": "Asked to verify",
    "genuine_review_rate": "Sent to review",
    "genuine_decline_rate": "Declined",
}


def group_rates(alloc: np.ndarray, n_good: np.ndarray) -> dict[str, float]:
    """Share of genuine applicants sent to each action, given a bucket allocation."""
    per_action = alloc.T @ n_good
    total = n_good.sum()
    out = {
        f"genuine_{name}_rate": float(per_action[i] / total) if total else np.nan
        for i, name in enumerate(("approve", "verify", "review", "decline"))
    }
    out["genuine_checked_rate"] = 1 - out["genuine_approve_rate"]
    return out


def fairness_table(hist_table: pd.DataFrame, a: Assumptions, variant: str = "full") -> pd.DataFrame:
    """One row per policy x metric: the rate for each age group and the 50+ / under-50 ratio."""
    val = load_hist(hist_table, variant, (CALIBRATION_MONTH,))
    test = load_hist(hist_table, variant, TEST_MONTHS)
    comp = compare(val, test, a)
    by_group = {g: load_hist(hist_table, variant, TEST_MONTHS, age_group=g) for g in GROUPS}
    warn = load_config()["fairness"]["warn_ratio"]
    rows = []
    for policy in ("paper_cutoff", "best_cutoff", "threshold"):
        rates = {g: group_rates(comp.allocs[policy], by_group[g].n_good) for g in GROUPS}
        for metric, label in METRICS.items():
            young, old = rates["under 50"][metric], rates["50+"][metric]
            ratio = old / young if young else np.nan
            rows.append(
                {
                    "variant": variant,
                    "policy": policy,
                    "policy_label": POLICY_LABELS[policy],
                    "metric": metric,
                    "metric_label": label,
                    "under_50": young,
                    "over_50": old,
                    "ratio": ratio,
                    "flag": bool(np.isfinite(ratio) and (ratio > warn or ratio < 1 / warn)),
                }
            )
    return pd.DataFrame(rows)
