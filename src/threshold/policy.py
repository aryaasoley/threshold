"""Choose the best action for every applicant, within the review team's capacity.

Without a capacity limit the answer is simply the highest line (economics.best_action).
With one, the review band may hold more cases than the team can handle. Every review takes
one slot, so the best use of K slots is the K cases where a review adds the most over the
next-best action. That gain is concave in p, so the cases kept still form one unbroken
band, just narrower. The gain of the best case left without a slot is the shadow price of
capacity: what one more review slot is worth.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from threshold.config import Assumptions
from threshold.economics import (
    ACTIONS,
    APPROVE,
    DECLINE,
    REVIEW,
    Band,
    Economics,
    best_non_review,
    review_gain,
)
from threshold.histogram import ScoreHist


@dataclass(frozen=True)
class Policy:
    """A decision rule fitted on one sample: review where gain > shadow_price, else best line."""

    econ: Economics
    shadow_price: float  # £ the marginal review slot would add (0 when capacity is slack)
    capacity: float | None  # review slots available on the sample it was fitted on
    alloc: np.ndarray  # (bins, 4): share of each bucket sent to each action on that sample
    p: np.ndarray  # the bucket probabilities it was fitted on

    @property
    def capacity_binds(self) -> bool:
        return self.shadow_price > 0


def _one_hot(actions: np.ndarray) -> np.ndarray:
    out = np.zeros((len(actions), 4))
    out[np.arange(len(actions)), actions] = 1.0
    return out


def _fill_reviews(
    gain: np.ndarray, n: np.ndarray, capacity: float, max_share: np.ndarray
) -> tuple[np.ndarray, float]:
    """Give slots to cases in order of gain, up to `max_share` of each bucket.

    Returns (share of each bucket reviewed, shadow price). The shadow price is the gain of
    the best case left without a slot, or 0 if every candidate fits.
    """
    share = np.zeros(len(gain))
    idx = np.flatnonzero((max_share > 0) & (n > 0))
    if len(idx) == 0:
        return share, 0.0
    order = idx[np.argsort(-gain[idx], kind="stable")]
    wanted = n[order] * max_share[order]
    cum = np.cumsum(wanted)
    if cum[-1] <= capacity:
        share[order] = max_share[order]
        return share, 0.0
    fits = cum <= capacity
    share[order[fits]] = max_share[order[fits]]
    k = int(np.argmin(fits))  # first bucket that does not fit whole
    used = cum[k - 1] if k > 0 else 0.0
    share[order[k]] = (capacity - used) / n[order[k]]
    return share, float(gain[order[k]])


def _with_reviews(p: np.ndarray, econ: Economics, share: np.ndarray) -> np.ndarray:
    alloc = _one_hot(best_non_review(p, econ)) * (1.0 - share)[:, None]
    alloc[:, REVIEW] += share
    return alloc


def optimise(hist: ScoreHist, econ: Economics, capacity: float | None = None) -> Policy:
    """Best action per bucket, using at most `capacity` reviews on this sample."""
    gain = review_gain(hist.p, econ)
    cap = np.inf if capacity is None else max(capacity, 0.0)
    share, price = _fill_reviews(gain, hist.n, cap, (gain > 0).astype(float))
    return Policy(
        econ=econ,
        shadow_price=price,
        capacity=None if np.isinf(cap) else float(cap),
        alloc=_with_reviews(hist.p, econ, share),
        p=hist.p,
    )


def apply(
    policy: Policy,
    p: np.ndarray,
    n: np.ndarray | None = None,
    capacity: float | None = None,
) -> np.ndarray:
    """Use a fitted policy on new applicants (e.g. the test months).

    The cut-offs are fixed: review where the gain beats the shadow price found on the fitting
    sample. On the same bucket grid, the bucket that was only partly reviewed keeps the same
    share, so a policy fitted for K slots asks for about K slots' worth of cases. If more cases
    qualify than `capacity` allows, analysts work the most valuable first and the rest get
    their next-best action.
    """
    econ = policy.econ
    gain = review_gain(p, econ)
    if len(p) == len(policy.p) and np.allclose(p, policy.p):
        max_share = policy.alloc[:, REVIEW]
    else:
        max_share = _in_review_band(gain, policy.shadow_price).astype(float)
    if n is None:
        n = np.ones(len(p))
    cap = np.inf if capacity is None else max(capacity, 0.0)
    share, _ = _fill_reviews(gain, n, cap, max_share)
    return _with_reviews(p, econ, share)


def _in_review_band(gain: np.ndarray, shadow_price: float) -> np.ndarray:
    # >= so the bucket that was only part-filled on the fitting sample stays a candidate
    return (gain > 0) & (gain >= shadow_price)


def reviews_wanted(policy: Policy, hist: ScoreHist) -> float:
    """Cases the fixed cut-offs send to review, before any capacity limit."""
    return float(apply(policy, hist.p, hist.n)[:, REVIEW] @ hist.n)


def evaluate(alloc: np.ndarray, hist: ScoreHist, econ: Economics) -> dict[str, float]:
    """Realised outcome of an allocation on labelled data, in counts and £."""
    good = alloc.T @ hist.n_good
    fraud = alloc.T @ hist.n_fraud
    n_good, n_fraud = hist.n_good.sum(), hist.n_fraud.sum()
    apps = n_good + n_fraud
    profit = float(good @ econ.value_if_good() + fraud @ econ.value_if_fraud())
    goods_on = float(good @ econ.keep_good())
    frauds_on = float(fraud @ econ.keep_fraud())
    out = {
        "applications": float(apps),
        "genuine": float(n_good),
        "fraud": float(n_fraud),
        "profit": profit,
        "profit_per_1k": 1000 * profit / apps if apps else 0.0,
        "genuine_onboarded": goods_on,
        "genuine_onboarded_share": goods_on / n_good if n_good else 0.0,
        "genuine_turned_away": float(n_good - goods_on),
        "fraud_onboarded": frauds_on,
        "fraud_stopped_share": 1 - frauds_on / n_fraud if n_fraud else 0.0,
        "fraud_loss": frauds_on * econ.L,
        "genuine_checked_share": float(good[1:].sum() / n_good) if n_good else 0.0,
    }
    for i, name in enumerate(ACTIONS):
        out[f"{name}_count"] = float(good[i] + fraud[i])
        out[f"{name}_rate"] = float((good[i] + fraud[i]) / apps) if apps else 0.0
        out[f"{name}_genuine"] = float(good[i])
        out[f"{name}_fraud"] = float(fraud[i])
    out["reviews"] = out["review_count"]
    return out


# --- simple alternatives -----------------------------------------------------------------


def approve_all(hist: ScoreHist) -> np.ndarray:
    return _one_hot(np.full(len(hist.p), APPROVE))


def cutoff_alloc(p: np.ndarray, cutoff: float) -> np.ndarray:
    """Two actions only: decline at or above the cut-off, approve below it."""
    return _one_hot(np.where(p >= cutoff, DECLINE, APPROVE))


def cutoff_at_fpr(hist: ScoreHist, fpr: float = 0.05) -> float:
    """The cut-off that flags `fpr` of genuine applicants: how the dataset's paper sets it."""
    order = np.argsort(-hist.p)
    flagged = np.cumsum(hist.n_good[order]) / max(hist.n_good.sum(), 1)
    k = int(np.searchsorted(flagged, fpr, side="right"))
    k = min(max(k, 1), len(order))
    return float(hist.p[order[k - 1]])


def best_cutoff(hist: ScoreHist, econ: Economics) -> float:
    """The single approve/decline cut-off with the highest profit on this sample."""
    order = np.argsort(hist.p)
    per_bucket = hist.n_good[order] * econ.V - hist.n_fraud[order] * econ.L
    # profit when declining everything from position k upwards
    profit = np.concatenate([[0.0], np.cumsum(per_bucket)])
    k = int(np.argmax(profit))
    return float(hist.p[order[k]]) if k < len(order) else 1.0 + 1e-9


# --- the whole comparison ----------------------------------------------------------------


@dataclass(frozen=True)
class Comparison:
    table: pd.DataFrame  # one row per policy, columns = evaluate() keys
    allocs: dict[str, np.ndarray]  # each policy's allocation on the test buckets
    policy: Policy  # Threshold's policy as fitted on the calibration month
    paper_cutoff: float
    best_cutoff: float
    reviews_wanted: float  # on the test months, before the capacity limit
    review_capacity: float  # on the test months


POLICY_LABELS = {
    "approve_all": "Approve everyone",
    "paper_cutoff": "One cut-off at 5% FPR (paper)",
    "best_cutoff": "Best single cut-off",
    "threshold": "Threshold (4 actions, capacity-aware)",
}


def compare(
    val: ScoreHist, test: ScoreHist, a: Assumptions, team_size: float | None = None
) -> Comparison:
    """Fit every policy on the calibration month, then score them all on the test months."""
    econ = a.economics()
    team = a.team_size if team_size is None else team_size
    pol = optimise(val, econ, a.review_capacity(val.months, team))
    test_cap = a.review_capacity(test.months, team)
    t_paper = cutoff_at_fpr(val, 0.05)
    t_best = best_cutoff(val, econ)
    allocs = {
        "approve_all": approve_all(test),
        "paper_cutoff": cutoff_alloc(test.p, t_paper),
        "best_cutoff": cutoff_alloc(test.p, t_best),
        "threshold": apply(pol, test.p, test.n, test_cap),
    }
    table = pd.DataFrame.from_dict(
        {k: evaluate(v, test, econ) for k, v in allocs.items()}, orient="index"
    )
    table.index.name = "policy"
    return Comparison(
        table=table,
        allocs=allocs,
        policy=pol,
        paper_cutoff=t_paper,
        best_cutoff=t_best,
        reviews_wanted=reviews_wanted(pol, test),
        review_capacity=test_cap,
    )


def cutoff_curve(test: ScoreHist, econ: Economics) -> pd.DataFrame:
    """Every single approve/decline cut-off at once: decline the top-k buckets, k = 0..all."""
    order = np.argsort(-test.p)
    good_declined = np.concatenate([[0.0], np.cumsum(test.n_good[order])])
    fraud_declined = np.concatenate([[0.0], np.cumsum(test.n_fraud[order])])
    goods_on = test.n_good.sum() - good_declined
    frauds_on = test.n_fraud.sum() - fraud_declined
    return pd.DataFrame(
        {
            "family": "single cut-off",
            "knob": np.concatenate([[1.0 + 1e-9], test.p[order]]),
            "genuine_onboarded_share": goods_on / max(test.n_good.sum(), 1),
            "genuine_turned_away": good_declined,
            "fraud_onboarded": frauds_on,
            "fraud_loss": frauds_on * econ.L,
            "profit": goods_on * econ.V - frauds_on * econ.L,
        }
    )


def threshold_curve(
    val: ScoreHist, test: ScoreHist, a: Assumptions, points: int = 60
) -> pd.DataFrame:
    """Threshold's policy as the weight on fraud is dialled from 30x lower to 30x higher.

    Each point is a policy fitted on the calibration month and scored on the test months
    with the true assumptions, so the curve is the menu of trade-offs the engine offers.
    """
    econ = a.economics()
    cap_val, cap_test = a.review_capacity(val.months), a.review_capacity(test.months)
    rows = []
    for weight in np.geomspace(econ.L / 30, econ.L * 30, points):
        pol = optimise(val, replace(econ, L=weight), cap_val)
        out = evaluate(apply(pol, test.p, test.n, cap_test), test, econ)
        rows.append(
            {
                "family": "Threshold",
                "knob": weight,
                "genuine_onboarded_share": out["genuine_onboarded_share"],
                "genuine_turned_away": out["genuine_turned_away"],
                "fraud_onboarded": out["fraud_onboarded"],
                "fraud_loss": out["fraud_loss"],
                "profit": out["profit"],
            }
        )
    return pd.DataFrame(rows)


def tradeoff_curves(
    val: ScoreHist, test: ScoreHist, a: Assumptions, points: int = 60
) -> pd.DataFrame:
    """Genuine customers onboarded vs fraud losses, single cut-off vs Threshold."""
    return pd.concat(
        [cutoff_curve(test, a.economics()), threshold_curve(val, test, a, points)],
        ignore_index=True,
    )


def matched_uplift(
    curves: pd.DataFrame, reference: dict[str, float] | pd.Series
) -> dict[str, float]:
    """At the same fraud loss as `reference`, how many more genuine customers Threshold onboards."""
    t = curves[curves["family"] == "Threshold"].sort_values("fraud_onboarded")
    x, y = t["fraud_onboarded"].to_numpy(), t["genuine_onboarded_share"].to_numpy()
    y = np.maximum.accumulate(y)  # the frontier: never accept fewer customers for more fraud
    share = float(np.interp(reference["fraud_onboarded"], x, y))
    ref_share = float(reference["genuine_onboarded_share"])
    ref_away = 1 - ref_share
    return {
        "genuine_onboarded_share": share,
        "reference_share": ref_share,
        "more_genuine_approved": share / ref_share - 1 if ref_share else 0.0,
        "fewer_turned_away": 1 - (1 - share) / ref_away if ref_away else 0.0,
    }


# --- capacity: the value of one more analyst ---------------------------------------------


def team_curve(
    val: ScoreHist, test: ScoreHist, a: Assumptions, sizes: range | None = None
) -> pd.DataFrame:
    """Profit by review-team size, and what each extra analyst adds per working day.

    Planned on the calibration month, like every other choice, at calibrated expected values:
    each analyst adds only a few hundred reviews, so a realised step in profit would rest on a
    handful of fraud cases and a hiring answer would flip on noise. In expectation the curve is
    concave by construction: each analyst works the most valuable cases left.

    The policy charges each review its full cost c, which includes the analyst's time. Salaries
    are paid whether or not a slot is used, so to value a whole analyst we add the time cost of
    the reviews done back on ("gross") and compare the step up with a day's salary. The realised
    profit of each team size on the test months is kept alongside as a check.
    """
    econ = a.economics()
    if sizes is None:
        wanted = reviews_wanted(optimise(val, econ), val)
        per_analyst = a.review_capacity(val.months, 1)
        top = int(np.ceil(wanted / per_analyst)) + 3 if per_analyst else 10
        sizes = range(0, max(top, a.team_size + 4, 6) + 1)
    expected = replace(val, n_good=val.n * (1 - val.p), n_fraud=val.n * val.p)
    days = val.months * a.working_days_per_month
    rows = []
    for n in sizes:
        pol = optimise(val, econ, a.review_capacity(val.months, n))
        out = evaluate(pol.alloc, expected, econ)
        gross = out["profit"] + a.review_time_cost * out["reviews"]
        realised = evaluate(
            apply(pol, test.p, test.n, a.review_capacity(test.months, n)), test, econ
        )
        rows.append(
            {
                "analysts": n,
                "capacity_per_day": a.cases_per_analyst_day * n,
                "reviews_per_day": out["reviews"] / days,
                "shadow_price": pol.shadow_price,
                "slot_value": pol.shadow_price + a.review_time_cost if pol.capacity_binds else 0.0,
                "gross_per_day": gross / days,
                "net_per_day": gross / days - n * a.analyst_daily_cost,
                "profit_per_1k": out["profit_per_1k"],
                "realised_profit_per_1k": realised["profit_per_1k"],
            }
        )
    df = pd.DataFrame(rows)
    df["marginal_value_per_day"] = df["gross_per_day"].diff()
    df["analyst_daily_cost"] = a.analyst_daily_cost
    df["worth_hiring"] = df["marginal_value_per_day"] > a.analyst_daily_cost
    return df


def recommended_team(curve: pd.DataFrame) -> int:
    return int(curve.loc[curve["net_per_day"].idxmax(), "analysts"])


# --- describing a policy -----------------------------------------------------------------


def decision_bands(policy: Policy, p_max: float = 1.0, points: int = 20_001) -> list[Band]:
    """The bands the fitted policy actually uses, capacity included, as (action, start, end)."""
    p = np.linspace(0, p_max, points)
    actions = apply(policy, p).argmax(axis=1)
    edges = np.flatnonzero(np.diff(actions)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(p)]])
    return [
        Band(int(actions[s]), float(p[s]), float(p[min(e, len(p) - 1)]))
        for s, e in zip(starts, ends, strict=True)
    ]


def describe_bands(bands: list[Band]) -> str:
    parts = []
    for i, b in enumerate(bands):
        if i == 0:
            parts.append(f"{b.name} below {b.end:.1%}")
        elif i == len(bands) - 1:
            parts.append(f"{b.name} above {b.start:.1%}")
        else:
            parts.append(f"{b.name} {b.start:.1%}–{b.end:.1%}")
    return " · ".join(parts)
