"""How sure are we? Bootstrap ranges and a sensitivity (tornado) chart.

Fraud is about 1% of applicants, so every headline number rests on a few thousand fraud
cases. The bootstrap resamples both the calibration month (so the cut-offs move) and the
test months (so the outcome moves), and re-runs the whole fit-then-test procedure each time.
The tornado swings each assumption across its plausible range, one at a time, and shows
which one moves the answer most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threshold.config import Assumptions
from threshold.histogram import ScoreHist
from threshold.policy import compare, matched_uplift, threshold_curve

HEADLINE = {
    "threshold_profit_per_1k": "Threshold profit per 1,000 applications (£)",
    "best_cutoff_profit_per_1k": "Best single cut-off profit per 1,000 applications (£)",
    "profit_uplift_per_1k": "Extra profit vs best single cut-off, per 1,000 applications (£)",
    "more_genuine_approved": "More genuine customers onboarded at the same fraud loss",
    "fewer_turned_away": "Fewer genuine customers turned away at the same fraud loss",
    "fraud_stopped_share": "Share of fraud stopped (Threshold)",
    "genuine_onboarded_share": "Share of genuine applicants onboarded (Threshold)",
}


def headline(
    val: ScoreHist, test: ScoreHist, a: Assumptions, curve_points: int = 40
) -> dict[str, float]:
    """The numbers the README and memo lead with, for one sample."""
    comp = compare(val, test, a)
    t, best = comp.table.loc["threshold"], comp.table.loc["best_cutoff"]
    up = matched_uplift(threshold_curve(val, test, a, curve_points), best)
    return {
        "threshold_profit_per_1k": t["profit_per_1k"],
        "best_cutoff_profit_per_1k": best["profit_per_1k"],
        "profit_uplift_per_1k": t["profit_per_1k"] - best["profit_per_1k"],
        "more_genuine_approved": up["more_genuine_approved"],
        "fewer_turned_away": up["fewer_turned_away"],
        "fraud_stopped_share": t["fraud_stopped_share"],
        "genuine_onboarded_share": t["genuine_onboarded_share"],
    }


def bootstrap(
    val: ScoreHist, test: ScoreHist, a: Assumptions, reps: int = 300, seed: int = 7
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        [headline(val.resample(rng), test.resample(rng), a, curve_points=30) for _ in range(reps)]
    )


def summarise(point: dict[str, float], boot: pd.DataFrame) -> pd.DataFrame:
    """Point estimate with a 95% bootstrap range for each headline number."""
    return pd.DataFrame(
        {
            "metric": list(HEADLINE),
            "label": list(HEADLINE.values()),
            "estimate": [point[k] for k in HEADLINE],
            "low": [float(boot[k].quantile(0.025)) for k in HEADLINE],
            "high": [float(boot[k].quantile(0.975)) for k in HEADLINE],
        }
    )


def tornado(
    val: ScoreHist,
    test: ScoreHist,
    a: Assumptions,
    ranges: dict[str, tuple[float, float]],
    metric: str = "profit_uplift_per_1k",
) -> pd.DataFrame:
    """Swing each assumption to its low and high value; record the metric each time."""
    base = headline(val, test, a, curve_points=20)[metric]
    rows = []
    for name, (lo, hi) in ranges.items():
        at_lo = headline(val, test, a.with_(**{name: lo}), curve_points=20)[metric]
        at_hi = headline(val, test, a.with_(**{name: hi}), curve_points=20)[metric]
        rows.append(
            {
                "assumption": name,
                "low_value": lo,
                "high_value": hi,
                "metric_at_low": at_lo,
                "metric_at_high": at_hi,
                "base": base,
                "swing": abs(at_hi - at_lo),
            }
        )
    return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


def quick_ranges(
    val: ScoreHist, test: ScoreHist, a: Assumptions, reps: int = 120, seed: int = 7
) -> pd.DataFrame:
    """A lighter bootstrap for the app: profit per 1,000 for each policy, no trade-off curve."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        table = compare(val.resample(rng), test.resample(rng), a).table
        rows.append(
            {
                **{f"{k}_profit_per_1k": table.loc[k, "profit_per_1k"] for k in table.index},
                "uplift_per_1k": table.loc["threshold", "profit_per_1k"]
                - table.loc["best_cutoff", "profit_per_1k"],
                "fraud_stopped_share": table.loc["threshold", "fraud_stopped_share"],
                "genuine_onboarded_share": table.loc["threshold", "genuine_onboarded_share"],
            }
        )
    boot = pd.DataFrame(rows)
    return boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "low", 0.975: "high"})
