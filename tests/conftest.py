from __future__ import annotations

import duckdb
import numpy as np
import pytest

from threshold.data import build_database
from threshold.economics import Economics
from threshold.fake_data import write_fake
from threshold.histogram import ScoreHist, bin_centres


@pytest.fixture(scope="session")
def fake_csv(tmp_path_factory):
    return write_fake(tmp_path_factory.mktemp("raw") / "fake_base.csv", n=20_000, seed=1)


@pytest.fixture(scope="session")
def fake_db(fake_csv, tmp_path_factory) -> duckdb.DuckDBPyConnection:
    con = build_database(fake_csv, tmp_path_factory.mktemp("db") / "fake.duckdb")
    yield con
    con.close()


@pytest.fixture
def econ() -> Economics:
    """The worked example from the docs: V = £50, L = £950, so decline above 5%."""
    return Economics(V=50, L=950, c=8, s=1.5, r=0.8, f=0.05, a_g=0.15, a_f=0.7)


@pytest.fixture
def toy_hist() -> ScoreHist:
    """A calibrated histogram: in each bucket the fraud share equals p."""
    p = bin_centres()
    n = np.where(p < 0.5, 2_000 * np.exp(-p * 60), 1.0).round() + 1
    return ScoreHist(p=p, n_good=np.round(n * (1 - p)), n_fraud=np.round(n * p), months=1.0)
