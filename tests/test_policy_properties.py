"""Random-input tests: properties that must hold for thousands of random assumptions."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from threshold.config import load_assumptions
from threshold.economics import APPROVE, REVIEW, Economics, best_action, envelope, expected_values
from threshold.histogram import ScoreHist, bin_centres
from threshold.policy import apply, optimise, team_curve

share = st.floats(0.0, 0.99)
economics = st.builds(
    Economics,
    V=st.floats(1, 500),
    L=st.floats(10, 20_000),
    c=st.floats(0.01, 100),
    s=st.floats(0.01, 50),
    r=share,
    f=share,
    a_g=share,
    a_f=share,
)


@st.composite
def histograms(draw) -> ScoreHist:
    p = bin_centres(200)
    seed = draw(st.integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    n = rng.poisson(rng.uniform(0, 300, size=len(p)))
    fraud = rng.binomial(n, p)
    return ScoreHist(p=p, n_good=(n - fraud).astype(float), n_fraud=fraud.astype(float))


def expected_profit(policy_alloc: np.ndarray, hist: ScoreHist, econ: Economics) -> float:
    return float((policy_alloc * expected_values(hist.p, econ)).sum(axis=1) @ hist.n)


@settings(max_examples=500)
@given(econ=economics, p=st.floats(0, 1))
def test_chosen_action_has_the_highest_value(econ, p):
    ev = expected_values(p, econ)[0]
    assert ev[best_action(p, econ)[0]] >= ev.max() - 1e-9


@settings(max_examples=500)
@given(econ=economics, p=st.floats(0, 1))
def test_envelope_agrees_with_brute_force(econ, p):
    band = next(b for b in envelope(econ) if b.start <= p <= b.end)
    ev = expected_values(p, econ)[0]
    assert ev[band.action] >= ev.max() - 1e-6 * max(1.0, abs(ev).max())


@settings(max_examples=500)
@given(econ=economics)
def test_each_action_owns_one_unbroken_band_in_slope_order(econ):
    actions = best_action(np.linspace(0, 1, 4001), econ)
    runs = [actions[0], *actions[1:][actions[1:] != actions[:-1]]]
    assert len(runs) == len(set(runs)), f"an action came back: {runs}"
    slopes = econ.slopes()[runs]
    assert np.all(np.diff(slopes) >= -1e-9), "bands must run from steepest line to flattest"


@settings(max_examples=300, deadline=None)
@given(econ=economics, hist=histograms(), capacity=st.floats(0, 20_000))
def test_reviews_never_exceed_capacity(econ, hist, capacity):
    pol = optimise(hist, econ, capacity)
    assert pol.alloc[:, REVIEW] @ hist.n <= capacity + 1e-6
    applied = apply(pol, hist.p, hist.n, capacity)
    assert applied[:, REVIEW] @ hist.n <= capacity + 1e-6
    assert np.allclose(pol.alloc.sum(axis=1), 1.0)


@settings(max_examples=300, deadline=None)
@given(econ=economics, hist=histograms(), capacity=st.floats(0, 10_000), extra=st.floats(0, 10_000))
def test_more_capacity_never_lowers_profit(econ, hist, capacity, extra):
    small = expected_profit(optimise(hist, econ, capacity).alloc, hist, econ)
    large = expected_profit(optimise(hist, econ, capacity + extra).alloc, hist, econ)
    assert large >= small - 1e-6 * max(1.0, abs(small))


@settings(max_examples=300, deadline=None)
@given(econ=economics, hist=histograms(), capacity=st.floats(0, 10_000))
def test_reviewed_cases_form_one_unbroken_band(econ, hist, capacity):
    reviewed = optimise(hist, econ, capacity).alloc[:, REVIEW] > 0
    occupied = hist.n > 0
    idx = np.flatnonzero(reviewed & occupied)
    if len(idx):
        between = occupied[idx[0] : idx[-1] + 1]
        assert reviewed[idx[0] : idx[-1] + 1][between].all()


@settings(max_examples=500)
@given(econ=economics, factor=st.floats(1.0, 20.0))
def test_higher_cost_per_fraud_never_increases_approvals(econ, factor):
    p = np.linspace(0, 1, 2001)
    approved_low = best_action(p, econ) == APPROVE
    approved_high = best_action(p, replace(econ, L=econ.L * factor)) == APPROVE
    assert not np.any(approved_high & ~approved_low)


@settings(max_examples=60, deadline=None)
@given(hist=histograms(), other=histograms(), cost=st.floats(10, 60), minutes=st.floats(4, 40))
def test_each_extra_analyst_is_worth_no_more_than_the_last(hist, other, cost, minutes):
    a = load_assumptions().with_(analyst_hourly_cost=cost, minutes_per_case=minutes)
    curve = team_curve(hist, other, a, range(0, 25))
    marginal = curve["marginal_value_per_day"].dropna().to_numpy()
    assert np.all(np.diff(marginal) <= 1e-6 * max(1.0, abs(marginal).max()))
