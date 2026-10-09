"""Generate a fake Base.csv with the real column names, types and quirks.

Used by the tests and by CI, so the whole pipeline can run without the real data ever
sitting in the repo. Fraud depends on the features through a known logistic score, so a
model has something to learn, and a few features drift by month so monitoring has
something to find. None of the numbers it produces mean anything about real applicants.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from threshold.config import RAW_DIR

FAKE_CSV = RAW_DIR / "fake_base.csv"

COLUMNS = [
    "fraud_bool", "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "days_since_request",
    "intended_balcon_amount", "payment_type", "zip_count_4w", "velocity_6h", "velocity_24h",
    "velocity_4w", "bank_branch_count_8w", "date_of_birth_distinct_emails_4w",
    "employment_status", "credit_risk_score", "email_is_free", "housing_status",
    "phone_home_valid", "phone_mobile_valid", "bank_months_count", "has_other_cards",
    "proposed_credit_limit", "foreign_request", "source", "session_length_in_minutes",
    "device_os", "keep_alive_session", "device_distinct_emails_8w", "device_fraud_count",
    "month",
]  # fmt: skip


def make_fake(n: int = 60_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    month_weights = np.array([0.14, 0.13, 0.14, 0.15, 0.13, 0.12, 0.10, 0.09])
    month = rng.choice(8, size=n, p=month_weights / month_weights.sum())
    drift = month / 7  # 0 → 1 across the eight months

    def with_missing(values: np.ndarray, share: float, sentinel: float = -1) -> np.ndarray:
        values = values.astype(float)
        values[rng.random(n) < share] = sentinel
        return values

    age = rng.choice([10, 20, 30, 40, 50, 60, 70, 80, 90], size=n,
                     p=[0.02, 0.25, 0.30, 0.22, 0.13, 0.05, 0.02, 0.007, 0.003])  # fmt: skip
    df = pd.DataFrame(
        {
            "income": rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), size=n),
            "name_email_similarity": rng.beta(1.3, 1.1, size=n),
            "prev_address_months_count": with_missing(rng.integers(5, 380, size=n), 0.70),
            "current_address_months_count": with_missing(rng.gamma(1.5, 60, size=n).round(), 0.01),
            "customer_age": age,
            "days_since_request": rng.exponential(1.0, size=n),
            "intended_balcon_amount": np.where(
                rng.random(n) < 0.74, -rng.uniform(0.1, 15, size=n), rng.uniform(0, 110, size=n)
            ),
            "payment_type": rng.choice(
                ["AA", "AB", "AC", "AD", "AE"], size=n, p=[0.26, 0.37, 0.25, 0.12, 0.0]
            ),
            "zip_count_4w": rng.gamma(2.0, 700, size=n).round(),
            "velocity_6h": rng.normal(5600 - 1500 * drift, 3000, size=n),
            "velocity_24h": rng.normal(4800 - 900 * drift, 1500, size=n),
            "velocity_4w": rng.normal(4900 - 600 * drift, 900, size=n),
            "bank_branch_count_8w": rng.choice(
                [0, 1, 2, 5, 20, 200, 1500], size=n, p=[0.35, 0.25, 0.1, 0.1, 0.1, 0.07, 0.03]
            ),
            "date_of_birth_distinct_emails_4w": rng.poisson(9, size=n),
            "employment_status": rng.choice(
                ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
                size=n,
                p=[0.73, 0.14, 0.04, 0.03, 0.02, 0.04, 0.0],
            ),
            "credit_risk_score": rng.normal(130 + 20 * drift, 70, size=n).round(),
            "email_is_free": rng.binomial(1, 0.53, size=n),
            "housing_status": rng.choice(
                ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
                size=n,
                p=[0.17, 0.26, 0.37, 0.03, 0.17, 0.0, 0.0],
            ),
            "phone_home_valid": rng.binomial(1, 0.42, size=n),
            "phone_mobile_valid": rng.binomial(1, 0.89, size=n),
            "bank_months_count": with_missing(rng.integers(1, 33, size=n), 0.25),
            "has_other_cards": rng.binomial(1, 0.22, size=n),
            "proposed_credit_limit": rng.choice(
                [200, 500, 1000, 1500, 2000], size=n, p=[0.4, 0.3, 0.15, 0.1, 0.05]
            ).astype(float),
            "foreign_request": rng.binomial(1, 0.025, size=n),
            "source": np.where(rng.random(n) < 0.993, "INTERNET", "TELEAPP"),
            "session_length_in_minutes": with_missing(rng.lognormal(1.5, 0.9, size=n), 0.002),
            "device_os": rng.choice(
                ["other", "linux", "windows", "macintosh", "x11"],
                size=n,
                p=[0.34, 0.33, 0.26, 0.06, 0.01],
            ),
            "keep_alive_session": rng.binomial(1, 0.58, size=n),
            "device_distinct_emails_8w": with_missing(
                rng.choice([0, 1, 2], size=n, p=[0.03, 0.94, 0.03]), 0.0005
            ),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": month,
        }
    )

    # A known fraud score: the model's job is to rediscover it.
    z = (
        -4.8
        + 0.9 * (df["customer_age"] >= 50)
        - 1.4 * df["name_email_similarity"]
        + 0.9 * (df["housing_status"] == "BA")
        + 0.006 * (df["credit_risk_score"] - 130)
        + 0.6 * df["email_is_free"]
        - 0.7 * df["phone_home_valid"]
        - 0.6 * df["keep_alive_session"]
        + 1.0 * (df["device_os"] == "windows")
        + 1.2 * (df["device_distinct_emails_8w"] >= 2)
        + 0.0007 * (df["proposed_credit_limit"] - 500)
        - 0.5 * df["has_other_cards"]
        + 0.5 * (df["payment_type"] == "AC")
        + 0.4 * (df["prev_address_months_count"] == -1)
        + 0.15 * drift
    )
    df["fraud_bool"] = rng.binomial(1, 1 / (1 + np.exp(-z)))
    return df[COLUMNS]


def write_fake(path: Path = FAKE_CSV, n: int = 60_000, seed: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    make_fake(n, seed).to_csv(path, index=False)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--out", type=Path, default=FAKE_CSV)
    args = parser.parse_args()
    print(f"wrote {write_fake(args.out, args.rows)}")
