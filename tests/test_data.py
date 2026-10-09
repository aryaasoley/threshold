"""Data checks: the SQL layer cleans what it should and the split cannot leak."""

from __future__ import annotations

import pytest

from threshold.config import CALIBRATION_MONTH, TEST_MONTHS, TRAIN_MONTHS
from threshold.data import (
    NEGATIVE_MEANS_MISSING,
    SENTINEL_COLUMNS,
    DataCheckError,
    check_data,
    load_model_input,
)
from threshold.model import FEATURES, split_by_time


def test_every_month_is_present(fake_db):
    months = fake_db.execute("SELECT DISTINCT month FROM model_input ORDER BY 1").fetchall()
    assert [m for (m,) in months] == list(range(8))


def test_check_data_passes_on_clean_data(fake_db):
    check_data(fake_db)


@pytest.mark.parametrize("col", SENTINEL_COLUMNS)
def test_minus_one_becomes_null(fake_db, col):
    raw = fake_db.execute(f"SELECT count(*) FROM raw_applications WHERE {col} = -1").fetchone()[0]
    clean = fake_db.execute(f"SELECT count(*) FROM applications WHERE {col} = -1").fetchone()[0]
    nulls = fake_db.execute(f"SELECT count(*) FROM applications WHERE {col} IS NULL").fetchone()[0]
    assert raw > 0, "fake data should contain the sentinel, or this test proves nothing"
    assert clean == 0
    assert nulls == raw


@pytest.mark.parametrize("col", NEGATIVE_MEANS_MISSING)
def test_negative_amounts_become_null(fake_db, col):
    clean = fake_db.execute(f"SELECT count(*) FROM applications WHERE {col} < 0").fetchone()[0]
    assert clean == 0


def test_check_data_catches_a_surviving_sentinel(fake_db):
    fake_db.execute("CREATE OR REPLACE TABLE model_input_backup AS SELECT * FROM model_input")
    try:
        fake_db.execute("UPDATE model_input SET bank_months_count = -1 WHERE application_id = 1")
        with pytest.raises(DataCheckError, match="bank_months_count"):
            check_data(fake_db)
    finally:
        fake_db.execute("CREATE OR REPLACE TABLE model_input AS SELECT * FROM model_input_backup")


def test_fraud_rate_is_in_a_sensible_range(fake_db):
    rate = fake_db.execute("SELECT avg(is_fraud) FROM model_input").fetchone()[0]
    assert 0.002 < rate < 0.05


def test_no_test_month_rows_used_in_training(fake_db):
    splits = split_by_time(load_model_input(fake_db))
    assert set(splits.train["month"]) <= set(TRAIN_MONTHS)
    assert set(splits.calibration["month"]) == {CALIBRATION_MONTH}
    assert set(splits.test["month"]) == set(TEST_MONTHS)
    train_ids = set(splits.train["application_id"])
    assert train_ids.isdisjoint(splits.calibration["application_id"])
    assert train_ids.isdisjoint(splits.test["application_id"])


def test_every_feature_reaches_the_model(fake_db):
    cols = set(load_model_input(fake_db).columns)
    assert set(FEATURES) <= cols
    assert len(FEATURES) == 30
