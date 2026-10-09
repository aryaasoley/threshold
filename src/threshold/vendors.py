"""What each data source is worth, and so the most you should pay for it.

Retrain the model without one supplier's features, recalibrate, re-fit the policy and
score it on the test months. The drop in profit per application is that source's value:
the break-even price per check, and your ceiling in a negotiation. Sources overlap in what
they reveal, so the values do not add up; dropping all external vendors at once shows by
how much.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threshold.config import CALIBRATION_MONTH, TEST_MONTHS, Assumptions
from threshold.histogram import ScoreHist, build_hist_table, load_hist
from threshold.model import EXTERNAL_GROUPS, FEATURE_GROUPS, FEATURES, GROUP_LABELS
from threshold.policy import apply, evaluate, optimise

VARIANT_LABELS = {
    "full": "All data",
    **{f"minus_{g}": f"Without {GROUP_LABELS[g].lower()}" for g in FEATURE_GROUPS},
    "minus_external": "Without any external vendor",
    "no_age": "All data except age",
}


def variant_features() -> dict[str, list[str]]:
    """The feature list for every model variant the pipeline trains."""
    out = {"full": FEATURES}
    for g, cols in FEATURE_GROUPS.items():
        out[f"minus_{g}"] = [f for f in FEATURES if f not in cols]
    external = {f for g in EXTERNAL_GROUPS for f in FEATURE_GROUPS[g]}
    out["minus_external"] = [f for f in FEATURES if f not in external]
    out["no_age"] = [f for f in FEATURES if f != "customer_age"]
    return out


def _threshold_outcome(val: ScoreHist, test: ScoreHist, a: Assumptions) -> dict[str, float]:
    econ = a.economics()
    pol = optimise(val, econ, a.review_capacity(val.months))
    return evaluate(apply(pol, test.p, test.n, a.review_capacity(test.months)), test, econ)


def vendor_values(hist_table: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Point estimate of each source's value per application under assumptions `a`."""
    variants = [v for v in hist_table["variant"].unique() if v.startswith("minus_")]
    base = _threshold_outcome(
        load_hist(hist_table, "full", (CALIBRATION_MONTH,)),
        load_hist(hist_table, "full", TEST_MONTHS),
        a,
    )
    rows = []
    for v in variants:
        out = _threshold_outcome(
            load_hist(hist_table, v, (CALIBRATION_MONTH,)),
            load_hist(hist_table, v, TEST_MONTHS),
            a,
        )
        group = v.removeprefix("minus_")
        rows.append(
            {
                "variant": v,
                "source": GROUP_LABELS.get(group, "All external vendors together"),
                "external": group in EXTERNAL_GROUPS or group == "external",
                "value_per_application": (base["profit"] - out["profit"]) / base["applications"],
                "fraud_stopped_with": base["fraud_stopped_share"],
                "fraud_stopped_without": out["fraud_stopped_share"],
            }
        )
    return pd.DataFrame(rows).sort_values("value_per_application", ascending=False)


def applicant_values(scored: pd.DataFrame, a: Assumptions) -> pd.Series:
    """Realised £ value of the Threshold decision for each test applicant (one model variant)."""
    variant = scored["variant"].iloc[0]
    val = load_hist(build_hist_table(scored), variant, (CALIBRATION_MONTH,))
    econ = a.economics()
    pol = optimise(val, econ, a.review_capacity(val.months))
    test = scored[scored["month"].isin(TEST_MONTHS)]
    alloc = apply(pol, test["p"].to_numpy(), None, a.review_capacity(len(TEST_MONTHS)))
    y = test["is_fraud"].to_numpy()
    value = np.where(y == 1, alloc @ econ.value_if_fraud(), alloc @ econ.value_if_good())
    return pd.Series(value, index=test["application_id"].to_numpy())


def paired_bootstrap(
    full: pd.Series, without: pd.Series, reps: int = 300, seed: int = 7
) -> tuple[float, float]:
    """95% range for a source's value per application, resampling the same applicants."""
    d = (full - without.reindex(full.index)).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(reps)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
