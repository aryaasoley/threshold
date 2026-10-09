"""Run everything end to end: data → SQL → model → decisions → analyses → artifacts.

    python -m threshold.pipeline            # real data (run `make data` first)
    python -m threshold.pipeline --fake     # generated data with the real column names (CI)

Only small summary tables are written to artifacts/. No raw rows leave data/.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from threshold import fairness, monitoring, uncertainty, vendors
from threshold.config import (
    ARTIFACTS,
    CALIBRATION_MONTH,
    FAKE_ARTIFACTS,
    TEST_MONTHS,
    assumption_ranges,
    load_assumptions,
    load_config,
)
from threshold.data import REAL_CSV, build_database, check_data, db_path, load_model_input, run_sql
from threshold.economics import ACTIONS
from threshold.fake_data import FAKE_CSV, write_fake
from threshold.histogram import bins_table, build_hist_table, load_hist
from threshold.model import (
    FEATURES,
    GBMSettings,
    ScoredVariant,
    gbm_backend,
    model_metrics,
    reliability_table,
    split_by_time,
    train_and_score,
)
from threshold.policy import apply, compare, optimise, recommended_team, team_curve, tradeoff_curves

TIMINGS: dict[str, float] = {}


@contextmanager
def stage(name: str):
    start = time.perf_counter()
    print(f"▸ {name} …", flush=True)
    yield
    TIMINGS[name] = round(time.perf_counter() - start, 2)
    print(f"  done in {TIMINGS[name]:.1f}s", flush=True)


def _write(df: pd.DataFrame, name: str, out: Path) -> None:
    df.to_parquet(out / f"{name}.parquet", index=False)


def run(fake: bool = False, rows: int = 60_000, fast: bool = False, out: Path = ARTIFACTS) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source = "synthetic" if fake else "feedzai-baf-base"
    a = load_assumptions()
    cfg = load_config()
    reps = 40 if fast else int(cfg["uncertainty"]["bootstrap_reps"])
    seed = int(cfg["uncertainty"]["seed"])
    settings = (
        GBMSettings(learning_rate=0.1, max_rounds=300, patience=30) if fast else GBMSettings()
    )
    t0 = time.perf_counter()

    with stage("data and SQL layer"):
        if fake:
            csv = write_fake(FAKE_CSV, rows)
        else:
            csv = REAL_CSV
            if not csv.exists():
                raise SystemExit(f"{csv} not found. Run `make data` first (see DATA.md).")
        con = build_database(csv, db_path("fake" if fake else "real"))
        check_data(con)
        df = load_model_input(con)
        for name in ("kpi_monthly", "kpi_monthly_segments"):
            _write(con.execute(f"SELECT * FROM {name}").df(), name, out)

    splits = split_by_time(df)
    variants: dict[str, ScoredVariant] = {}

    with stage("model: baseline and gradient boosting"):
        baseline = train_and_score("baseline", splits, kind="logistic")
        variants["full"] = train_and_score("full", splits, FEATURES, "gbm", None, settings)
        rounds = variants["full"].model.rounds

    with stage("model metrics and calibration"):
        metrics = {"backend": gbm_backend(), "rounds": rounds, "test_months": list(TEST_MONTHS)}
        reliability = []
        for label, sv in (("baseline", baseline), ("gbm", variants["full"])):
            test = sv.scored[sv.scored["month"].isin(TEST_MONTHS)]
            y = test["is_fraud"].to_numpy()
            metrics[label] = {
                "raw": model_metrics(y, test["p_raw"].to_numpy()),
                "calibrated": model_metrics(y, test["p"].to_numpy()),
                "calibrator": {"a": sv.calibrator.a, "b": sv.calibrator.b},
            }
            for kind in ("raw", "calibrated"):
                col = "p_raw" if kind == "raw" else "p"
                reliability.append(
                    reliability_table(y, test[col].to_numpy()).assign(model=label, stage=kind)
                )
        (out / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
        _write(pd.concat(reliability), "calibration", out)

    with stage("model variants for vendor value and fairness"):
        for name, feats in vendors.variant_features().items():
            if name != "full":
                variants[name] = train_and_score(name, splits, feats, "gbm", rounds, settings)

    with stage("probability histograms"):
        scored_all = pd.concat([sv.scored for sv in variants.values()], ignore_index=True)
        hist = build_hist_table(scored_all)
        _write(hist, "score_hist", out)
        _write(bins_table(), "bins", out)

    val = load_hist(hist, "full", (CALIBRATION_MONTH,))
    test = load_hist(hist, "full", TEST_MONTHS)

    with stage("decisions and segment KPIs (SQL)"):
        full = variants["full"].scored
        pol = optimise(val, a.economics(), a.review_capacity(1))
        parts = []
        for m in (CALIBRATION_MONTH, *TEST_MONTHS):
            month = full[full["month"] == m]
            alloc = apply(pol, month["p"].to_numpy(), None, a.review_capacity(1))
            parts.append(
                month[["application_id", "month", "p"]].assign(
                    action=np.array(ACTIONS)[alloc.argmax(axis=1)]
                )
            )
        decisions = pd.concat(parts, ignore_index=True)
        con.register("decisions_df", decisions)
        con.execute("CREATE OR REPLACE TABLE decisions AS SELECT * FROM decisions_df")
        run_sql(con, "04_segments")
        _write(con.execute("SELECT * FROM kpi_policy_segments").df(), "kpi_policy_segments", out)
        con.close()

    with stage("policy comparison and trade-off curves"):
        comp = compare(val, test, a)
        _write(comp.table.reset_index(), "policy_comparison", out)
        _write(tradeoff_curves(val, test, a), "tradeoff", out)

    with stage("value of an analyst"):
        curve = team_curve(val, test, a)
        _write(curve, "team_curve", out)

    with stage("value of each data source (paired bootstrap)"):
        values = vendors.vendor_values(hist, a)
        per_app = {name: vendors.applicant_values(sv.scored, a) for name, sv in variants.items()}
        ranges = [
            vendors.paired_bootstrap(per_app["full"], per_app[v], reps, seed)
            for v in values["variant"]
        ]
        values["low"] = [r[0] for r in ranges]
        values["high"] = [r[1] for r in ranges]
        _write(values, "vendor_values", out)

    with stage("fairness by age group"):
        fair = pd.concat(
            [fairness.fairness_table(hist, a, "full"), fairness.fairness_table(hist, a, "no_age")]
        )
        _write(fair, "fairness", out)

    with stage("monitoring and drift"):
        summary, bands = monitoring.monthly_monitor(hist, a)
        _write(summary, "monitoring", out)
        _write(bands, "monitoring_bands", out)
        _write(monitoring.feature_psi(df, FEATURES), "feature_psi", out)

    with stage(f"uncertainty: bootstrap ({reps} reps) and tornado"):
        point = uncertainty.headline(val, test, a)
        boot = uncertainty.bootstrap(val, test, a, reps, seed)
        _write(uncertainty.summarise(point, boot), "headline", out)
        _write(uncertainty.tornado(val, test, a, assumption_ranges()), "tornado", out)

    TIMINGS["total"] = round(time.perf_counter() - t0, 1)
    meta = {
        "data_source": source,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "applications": len(df),
        "test_applications": int(test.applications),
        "test_frauds": int(test.n_fraud.sum()),
        "model_backend": gbm_backend(),
        "fast_mode": fast,
        "assumptions": a.to_dict(),
        "recommended_team": recommended_team(curve),
        "test_reviews_wanted": comp.reviews_wanted,
        "test_review_capacity": comp.review_capacity,
        "timings_seconds": TIMINGS,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n✓ pipeline finished in {TIMINGS['total']:.0f}s → {out}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fake", action="store_true", help="use generated data (no download)")
    parser.add_argument("--rows", type=int, default=60_000, help="rows of fake data")
    parser.add_argument("--fast", action="store_true", help="fewer trees and bootstrap reps (CI)")
    parser.add_argument("--out", type=Path, help="default: artifacts/ (real) or artifacts_fake/")
    args = parser.parse_args()
    out = args.out or (FAKE_ARTIFACTS if args.fake else ARTIFACTS)
    run(fake=args.fake, rows=args.rows, fast=args.fast, out=out)


if __name__ == "__main__":
    main()
