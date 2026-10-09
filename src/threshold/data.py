"""Download the data, load it into DuckDB, run the SQL layer and check the result."""

from __future__ import annotations

import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from threshold.config import (
    CALIBRATION_MONTH,
    DATA_DIR,
    RAW_DIR,
    SQL_DIR,
    TEST_MONTHS,
    TRAIN_MONTHS,
)

KAGGLE_DATASET = "sgpjesus/bank-account-fraud-dataset-neurips-2022"
REAL_CSV = RAW_DIR / "Base.csv"

SENTINEL_COLUMNS = [
    "prev_address_months_count",
    "current_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]
NEGATIVE_MEANS_MISSING = ["intended_balcon_amount", "session_length_in_minutes"]


class DataCheckError(RuntimeError):
    pass


def download(dest: Path = RAW_DIR) -> Path:
    """Fetch Base.csv from Kaggle. Needs a free Kaggle account and API token (see DATA.md)."""
    if REAL_CSV.exists():
        print(f"{REAL_CSV} already present")
        return REAL_CSV
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as err:  # pragma: no cover
        raise SystemExit("The kaggle package is missing: run `uv sync`") from err
    dest.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"downloading Base.csv from kaggle.com/datasets/{KAGGLE_DATASET} …")
    api.dataset_download_file(KAGGLE_DATASET, "Base.csv", path=str(dest))
    archive = dest / "Base.csv.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
        archive.unlink()
    if not REAL_CSV.exists():
        raise SystemExit(f"Download finished but {REAL_CSV} is missing")
    return REAL_CSV


def db_path(source: str) -> Path:
    return DATA_DIR / f"threshold_{source}.duckdb"


def run_sql(con: duckdb.DuckDBPyConnection, name: str) -> None:
    con.execute((SQL_DIR / f"{name}.sql").read_text())


def build_database(csv: Path, db: Path) -> duckdb.DuckDBPyConnection:
    """Create the DuckDB file and run the cleaning, feature and KPI SQL."""
    db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db))
    path = str(csv).replace("'", "''")
    con.execute(
        "CREATE OR REPLACE VIEW raw_applications AS "
        f"SELECT * FROM read_csv_auto('{path}', header = true)"
    )
    for name in ("01_clean", "02_features", "03_monthly_kpis"):
        run_sql(con, name)
    return con


def check_data(
    con: duckdb.DuckDBPyConnection, min_rate: float = 0.002, max_rate: float = 0.05
) -> None:
    """Fail loudly if the cleaned data is not what the rest of the pipeline assumes."""
    problems: list[str] = []

    per_month = con.execute(
        "SELECT month, count(*) AS n, avg(is_fraud) AS rate "
        "FROM model_input GROUP BY month ORDER BY month"
    ).df()
    if sorted(per_month["month"]) != list(range(8)):
        problems.append(f"expected months 0-7, found {sorted(per_month['month'])}")
    if (per_month["n"] < 100).any():
        problems.append("a month has fewer than 100 applications")
    bad = per_month[(per_month["rate"] < min_rate) | (per_month["rate"] > max_rate)]
    if len(bad):
        problems.append(
            f"fraud rate outside [{min_rate}, {max_rate}] in months {bad['month'].tolist()}"
        )

    for col in SENTINEL_COLUMNS:
        n = con.execute(f"SELECT count(*) FROM model_input WHERE {col} = -1").fetchone()[0]
        if n:
            problems.append(f"{col} still has {n} rows of -1")
    for col in NEGATIVE_MEANS_MISSING:
        n = con.execute(f"SELECT count(*) FROM model_input WHERE {col} < 0").fetchone()[0]
        if n:
            problems.append(f"{col} still has {n} negative rows")

    dupes = con.execute(
        "SELECT count(*) - count(DISTINCT application_id) FROM model_input"
    ).fetchone()[0]
    if dupes:
        problems.append(f"{dupes} duplicate application ids")

    splits = con.execute(
        "SELECT split, list(DISTINCT month ORDER BY month) FROM model_input GROUP BY split"
    ).fetchall()
    expected = {
        "train": list(TRAIN_MONTHS),
        "calibration": [CALIBRATION_MONTH],
        "test": list(TEST_MONTHS),
    }
    if {k: list(v) for k, v in splits} != expected:
        problems.append(f"split/month mapping is {dict(splits)}, expected {expected}")

    if problems:
        raise DataCheckError("Data checks failed:\n  - " + "\n  - ".join(problems))


def load_model_input(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute("SELECT * FROM model_input ORDER BY application_id").df()
