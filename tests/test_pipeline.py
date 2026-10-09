"""The whole pipeline, end to end, on a small fake dataset."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from threshold import pipeline

EXPECTED = [
    "score_hist", "bins", "policy_comparison", "tradeoff", "team_curve", "vendor_values",
    "fairness", "monitoring", "monitoring_bands", "feature_psi", "headline", "tornado",
    "calibration", "kpi_monthly", "kpi_monthly_segments", "kpi_policy_segments",
]  # fmt: skip


@pytest.mark.slow
def test_pipeline_runs_on_fake_data(tmp_path):
    meta = pipeline.run(fake=True, rows=15_000, fast=True, out=tmp_path)
    assert meta["data_source"] == "synthetic"
    for name in EXPECTED:
        assert (tmp_path / f"{name}.parquet").exists(), name
    metrics = json.loads((tmp_path / "model_metrics.json").read_text())
    assert 0.5 < metrics["gbm"]["calibrated"]["roc_auc"] <= 1

    hist = pd.read_parquet(tmp_path / "score_hist.parquet")
    assert set(hist["month"]) == {5, 6, 7}, "only out-of-sample months are shipped"
    assert {"p", "is_fraud", "application_id"}.isdisjoint(hist.columns), "no row-level data"

    comp = pd.read_parquet(tmp_path / "policy_comparison.parquet").set_index("policy")
    assert comp.loc["threshold", "profit"] >= comp.loc["approve_all", "profit"]
