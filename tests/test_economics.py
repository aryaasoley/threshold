"""Known-answer tests: results you can check on a whiteboard."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from threshold.economics import (
    APPROVE,
    DECLINE,
    REVIEW,
    VERIFY,
    approve_decline_breakeven,
    best_action,
    envelope,
    expected_values,
    review_gain,
)


def test_approve_and_decline_cross_at_v_over_v_plus_l(econ):
    p_star = approve_decline_breakeven(econ)
    assert p_star == pytest.approx(50 / (50 + 950))  # 5%
    ev = expected_values(p_star, econ)[0]
    assert ev[APPROVE] == pytest.approx(ev[DECLINE])


def test_two_action_world_switches_exactly_at_breakeven(econ):
    """Price verify and review out of the market and only the approve/decline cut-off is left."""
    two_action = replace(econ, c=1e9, s=1e9)
    p_star = approve_decline_breakeven(two_action)
    assert best_action(p_star - 1e-9, two_action)[0] == APPROVE
    assert best_action(p_star + 1e-9, two_action)[0] == DECLINE
    bands = envelope(two_action)
    assert [b.name for b in bands] == ["approve", "decline"]
    assert bands[0].end == pytest.approx(p_star)


def test_zero_risk_means_approve(econ):
    assert best_action(0.0, econ)[0] == APPROVE


def test_certain_fraud_means_decline(econ):
    assert best_action(1.0, econ)[0] == DECLINE


def test_default_bands_come_in_the_expected_order(econ):
    names = [b.name for b in envelope(econ)]
    assert names == ["approve", "verify", "review", "decline"]


def test_band_edges_are_where_lines_cross(econ):
    bands = envelope(econ)
    for left, right in pairwise(bands):
        ev = expected_values(left.end, econ)[0]
        assert ev[left.action] == pytest.approx(ev[right.action])


def test_expected_value_by_hand(econ):
    p = 0.02
    ev = expected_values(p, econ)[0]
    assert ev[APPROVE] == pytest.approx(0.98 * 50 - 0.02 * 950)
    assert ev[VERIFY] == pytest.approx(0.98 * 0.85 * 50 - 0.02 * 0.3 * 950 - 1.5)
    assert ev[REVIEW] == pytest.approx(0.98 * 0.95 * 50 - 0.02 * 0.2 * 950 - 8)
    assert ev[DECLINE] == 0


def test_review_gain_peaks_inside_the_review_band(econ):
    review = next(b for b in envelope(econ) if b.name == "review")
    p = np.linspace(0, 0.5, 5001)
    peak = p[np.argmax(review_gain(p, econ))]
    assert review.start <= peak <= review.end
