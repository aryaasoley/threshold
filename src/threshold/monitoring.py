"""Replay the test months as if they were live and watch for trouble.

Confirmed fraud arrives weeks late in real life, so outcome-based checks (fraud caught,
actual vs expected fraud) lag. The population stability index (PSI) needs no labels: it only
asks whether this month's scores and features look like the reference month's. That makes
it the early warning. Rule of thumb: under 0.10 stable, 0.10–0.25 watch, over 0.25 act.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from threshold.config import CALIBRATION_MONTH, TEST_MONTHS, TRAIN_MONTHS, Assumptions, load_config
from threshold.economics import ACTIONS
from threshold.histogram import ScoreHist, load_hist
from threshold.model import CATEGORIES
from threshold.policy import apply, evaluate, optimise, reviews_wanted


def psi(reference: np.ndarray, current: np.ndarray, eps: float = 1e-4) -> float:
    """Σ (cur − ref) · ln(cur / ref) over buckets, on shares. 0 means identical."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = np.clip(ref / max(ref.sum(), 1e-12), eps, None)
    cur = np.clip(cur / max(cur.sum(), 1e-12), eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI on reference-decile buckets, with missing values as their own bucket."""
    ref = reference.to_numpy(dtype=float)
    cur = current.to_numpy(dtype=float)
    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))

    def counts(x: np.ndarray) -> np.ndarray:
        finite = x[~np.isnan(x)]
        c = np.bincount(np.searchsorted(edges, finite, side="right"), minlength=len(edges) + 1)
        return np.append(c, np.isnan(x).sum())

    return psi(counts(ref), counts(cur))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    levels = pd.Index(reference.astype("object").fillna("missing").unique()).union(
        current.astype("object").fillna("missing").unique()
    )
    ref = reference.astype("object").fillna("missing").value_counts().reindex(levels, fill_value=0)
    cur = current.astype("object").fillna("missing").value_counts().reindex(levels, fill_value=0)
    return psi(ref.to_numpy(), cur.to_numpy())


def feature_psi(
    df: pd.DataFrame,
    features: list[str],
    reference_months: tuple[int, ...] = TRAIN_MONTHS,
    months: tuple[int, ...] = (CALIBRATION_MONTH, *TEST_MONTHS),
) -> pd.DataFrame:
    """PSI of each feature in each month against the months the model learned from."""
    ref = df[df["month"].isin(reference_months)]
    rows = []
    for m in months:
        cur = df[df["month"] == m]
        for f in features:
            value = (
                psi_categorical(ref[f], cur[f]) if f in CATEGORIES else psi_numeric(ref[f], cur[f])
            )
            rows.append({"month": m, "feature": f, "psi": value})
    return pd.DataFrame(rows)


def score_psi(reference: ScoreHist, current: ScoreHist, buckets: int = 10) -> float:
    """PSI of the fraud score, on deciles of the reference month's score."""
    cum = np.cumsum(reference.n) / max(reference.n.sum(), 1)
    group = np.minimum((cum * buckets).astype(int), buckets - 1)
    ref = np.bincount(group, weights=reference.n, minlength=buckets)
    cur = np.bincount(group, weights=current.n, minlength=buckets)
    return psi(ref, cur)


def rag(value: float, watch: float, act: float) -> str:
    if not np.isfinite(value):
        return "grey"
    return "green" if value < watch else ("amber" if value < act else "red")


def monthly_monitor(
    hist_table: pd.DataFrame,
    a: Assumptions,
    variant: str = "full",
    months: tuple[int, ...] = (CALIBRATION_MONTH, *TEST_MONTHS),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-by-month KPIs for a policy fitted on the calibration month.

    Returns (summary, bands): one row per month, and actual vs expected fraud per action band.
    """
    cfg = load_config()["monitoring"]
    econ = a.economics()
    ref = load_hist(hist_table, variant, (CALIBRATION_MONTH,))
    pol = optimise(ref, econ, a.review_capacity(1))
    rows, band_rows = [], []
    for m in months:
        h = load_hist(hist_table, variant, (m,))
        alloc = apply(pol, h.p, h.n, a.review_capacity(1))
        out = evaluate(alloc, h, econ)
        expected = float(h.sum_p.sum())
        actual = float(h.n_fraud.sum())
        calib_gap = actual / expected - 1 if expected else np.nan
        s_psi = score_psi(ref, h)
        rows.append(
            {
                "month": m,
                "role": "reference" if m == CALIBRATION_MONTH else "live replay",
                "applications": out["applications"],
                "approve_rate": out["approve_rate"],
                "verify_rate": out["verify_rate"],
                "review_rate": out["review_rate"],
                "decline_rate": out["decline_rate"],
                "fraud_rate": out["fraud"] / out["applications"],
                "fraud_stopped_share": out["fraud_stopped_share"],
                "profit_per_1k": out["profit_per_1k"],
                "reviews_wanted": reviews_wanted(pol, h),
                "review_capacity": a.review_capacity(1),
                "expected_fraud": expected,
                "actual_fraud": actual,
                "calibration_gap": calib_gap,
                "score_psi": s_psi,
                "score_status": rag(s_psi, cfg["psi_watch"], cfg["psi_act"]),
                "calibration_status": rag(
                    abs(calib_gap), cfg["calibration_watch"], cfg["calibration_act"]
                ),
            }
        )
        for i, action in enumerate(ACTIONS):
            band_rows.append(
                {
                    "month": m,
                    "action": action,
                    "applications": float(alloc[:, i] @ h.n),
                    "expected_fraud": float(alloc[:, i] @ h.sum_p),
                    "actual_fraud": float(alloc[:, i] @ h.n_fraud),
                }
            )
    summary = pd.DataFrame(rows)
    summary["capacity_status"] = np.where(
        summary["reviews_wanted"] <= summary["review_capacity"],
        "green",
        np.where(summary["reviews_wanted"] <= 1.2 * summary["review_capacity"], "amber", "red"),
    )
    return summary, pd.DataFrame(band_rows)
