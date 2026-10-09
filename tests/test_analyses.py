"""Monitoring, histograms, capacity and the experiment calculator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from threshold.config import load_assumptions
from threshold.economics import REVIEW
from threshold.experiment import sample_size_per_arm
from threshold.fairness import group_rates
from threshold.histogram import assign_bins, bin_centres, bin_edges, build_hist_table, load_hist
from threshold.monitoring import psi, psi_numeric, rag
from threshold.policy import (
    best_cutoff,
    cutoff_alloc,
    cutoff_at_fpr,
    evaluate,
    optimise,
    team_curve,
)


def test_psi_is_zero_for_identical_distributions():
    assert psi(np.array([10, 20, 30]), np.array([1, 2, 3])) == pytest.approx(0.0)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 50_000))
    small = psi_numeric(ref, pd.Series(rng.normal(0.1, 1, 50_000)))
    large = psi_numeric(ref, pd.Series(rng.normal(1.0, 1, 50_000)))
    assert 0 < small < 0.1 < 0.25 < large


def test_rag_thresholds():
    assert rag(0.05, 0.1, 0.25) == "green"
    assert rag(0.15, 0.1, 0.25) == "amber"
    assert rag(0.30, 0.1, 0.25) == "red"


def test_bins_cover_zero_to_one():
    e = bin_edges()
    assert e[0] == 0 and e[-1] == 1
    assert np.all(np.diff(e) > 0)
    assert assign_bins(np.array([0.0, 1.0]))[[0, 1]].tolist() == [0, len(e) - 2]
    c = bin_centres()
    assert np.all((c >= e[:-1]) & (c <= e[1:]))


def test_histogram_table_round_trips_counts():
    rng = np.random.default_rng(3)
    p = rng.beta(0.5, 40, 5_000)
    scored = pd.DataFrame(
        {
            "variant": "full",
            "month": rng.choice([5, 6, 7], 5_000),
            "age_group": rng.choice(["under 50", "50+"], 5_000),
            "p": p,
            "is_fraud": rng.binomial(1, p),
        }
    )
    hist = load_hist(build_hist_table(scored), "full", (5, 6, 7))
    assert hist.n_fraud.sum() == scored["is_fraud"].sum()
    assert hist.n.sum() == len(scored)
    assert hist.sum_p.sum() == pytest.approx(p.sum())


def test_capacity_binds_and_prices_the_marginal_slot(econ, toy_hist):
    free = optimise(toy_hist, econ)
    wanted = free.alloc[:, REVIEW] @ toy_hist.n
    tight = optimise(toy_hist, econ, wanted / 2)
    assert free.shadow_price == 0
    assert tight.shadow_price > 0
    assert tight.alloc[:, REVIEW] @ toy_hist.n == pytest.approx(wanted / 2)


def test_threshold_beats_best_single_cutoff_in_expectation(econ, toy_hist):
    """With calibrated p and spare capacity, more options can only help."""
    four = evaluate(optimise(toy_hist, econ).alloc, toy_hist, econ)["profit"]
    t = best_cutoff(toy_hist, econ)
    two = evaluate(cutoff_alloc(toy_hist.p, t), toy_hist, econ)["profit"]
    assert four >= two


def test_paper_cutoff_flags_about_five_percent_of_genuine(toy_hist):
    t = cutoff_at_fpr(toy_hist, 0.05)
    flagged = toy_hist.n_good[toy_hist.p >= t].sum() / toy_hist.n_good.sum()
    assert flagged <= 0.05 + 1e-9
    assert flagged > 0.04


def test_team_curve_marginal_value_falls(toy_hist):
    a = load_assumptions()
    curve = team_curve(toy_hist, toy_hist, a, range(0, 12))
    gross = curve["gross_per_day"].to_numpy()
    assert np.all(np.diff(gross) >= -1e-6), "an extra analyst never destroys value"
    marginal = curve["marginal_value_per_day"].dropna().to_numpy()
    assert np.all(np.diff(marginal) <= 1e-6), "each extra analyst is worth no more than the last"


def test_group_rates_by_hand():
    alloc = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    rates = group_rates(alloc, np.array([80.0, 15.0, 5.0]))
    assert rates["genuine_approve_rate"] == pytest.approx(0.80)
    assert rates["genuine_verify_rate"] == pytest.approx(0.15)
    assert rates["genuine_decline_rate"] == pytest.approx(0.05)
    assert rates["genuine_checked_rate"] == pytest.approx(0.20)


def test_sample_size_matches_textbook_value():
    # 10% → 12% at 5% significance and 80% power: about 3,841 per arm
    assert sample_size_per_arm(0.10, 0.02) == pytest.approx(3841, abs=5)
