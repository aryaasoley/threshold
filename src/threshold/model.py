"""Estimate each applicant's probability of being a fraudster.

Learn on months 0–4, calibrate on month 5, test on months 6–7. The model is deliberately
modest: a logistic-regression baseline and one gradient-boosted tree model. The interesting
work happens after the score, in economics.py and policy.py.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from threshold.config import CALIBRATION_MONTH, TEST_MONTHS, TRAIN_MONTHS

# Who would plausibly supply each feature. Used to value data sources in vendors.py.
FEATURE_GROUPS: dict[str, list[str]] = {
    "application": [
        "income", "customer_age", "employment_status", "housing_status", "payment_type",
        "proposed_credit_limit", "prev_address_months_count", "current_address_months_count",
        "intended_balcon_amount", "days_since_request", "source",
    ],
    "internal": [
        "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w", "bank_branch_count_8w",
        "date_of_birth_distinct_emails_4w",
    ],
    "device": [
        "device_os", "device_distinct_emails_8w", "device_fraud_count", "keep_alive_session",
        "session_length_in_minutes", "foreign_request",
    ],
    "bureau": ["credit_risk_score", "has_other_cards", "bank_months_count"],
    "email_phone": [
        "name_email_similarity", "email_is_free", "phone_home_valid", "phone_mobile_valid",
    ],
}  # fmt: skip
GROUP_LABELS = {
    "application": "Application form",
    "internal": "Internal activity checks",
    "device": "Device data",
    "bureau": "Credit bureau",
    "email_phone": "Email and phone checks",
}
EXTERNAL_GROUPS = ("device", "bureau", "email_phone")  # the ones a vendor would sell you
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]

CATEGORIES: dict[str, list[str]] = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_by_time(df: pd.DataFrame) -> Splits:
    """Months 0–4 to learn, month 5 to calibrate and choose cut-offs, months 6–7 to test."""
    s = Splits(
        train=df[df["month"].isin(TRAIN_MONTHS)],
        calibration=df[df["month"] == CALIBRATION_MONTH],
        test=df[df["month"].isin(TEST_MONTHS)],
    )
    assert set(s.train["month"]).isdisjoint(set(s.test["month"]) | {CALIBRATION_MONTH})
    return s


def prepare(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Select features and give categoricals a fixed set of levels, so months line up."""
    X = df[features].copy()
    for col, levels in CATEGORIES.items():
        if col in X:
            X[col] = pd.Categorical(X[col], categories=levels)
    return X


def gbm_backend() -> str:
    """LightGBM when it loads; otherwise scikit-learn's histogram GBM (same algorithm family).

    On macOS LightGBM needs the OpenMP runtime (`brew install libomp`).
    """
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        return "sklearn-hgb"
    return "lightgbm"


@dataclass(frozen=True)
class GBMSettings:
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    colsample: float = 0.8
    l2: float = 1.0
    max_rounds: int = 2000
    patience: int = 100
    seed: int = 7


@dataclass
class FraudModel:
    """A fitted estimator plus the features it expects. Outputs raw (uncalibrated) scores."""

    estimator: Any
    features: list[str]
    kind: str
    rounds: int | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        X = (
            prepare(df, self.features)
            if self.kind != "logistic"
            else _baseline_frame(df, self.features)
        )
        return self.estimator.predict_proba(X)[:, 1]


def _make_lightgbm(s: GBMSettings, rounds: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=rounds,
        learning_rate=s.learning_rate,
        num_leaves=s.num_leaves,
        min_child_samples=s.min_child_samples,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=s.colsample,
        reg_lambda=s.l2,
        random_state=s.seed,
        verbose=-1,
    )


def _make_hgb(
    s: GBMSettings, rounds: int, warm_start: bool = False
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=s.learning_rate,
        max_iter=rounds,
        max_leaf_nodes=s.num_leaves,
        min_samples_leaf=s.min_child_samples,
        max_features=s.colsample,
        l2_regularization=s.l2,
        categorical_features="from_dtype",
        early_stopping=False,
        warm_start=warm_start,
        random_state=s.seed,
    )


def _best_rounds(
    train: pd.DataFrame, hold: pd.DataFrame, features: list[str], s: GBMSettings
) -> int:
    """Early stopping that respects time: learn on months 0–3, stop on month 4."""
    Xt, yt = prepare(train, features), train["is_fraud"].to_numpy()
    Xh, yh = prepare(hold, features), hold["is_fraud"].to_numpy()
    if gbm_backend() == "lightgbm":
        import lightgbm as lgb

        m = _make_lightgbm(s, s.max_rounds)
        m.fit(
            Xt, yt, eval_set=[(Xh, yh)], eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(s.patience, verbose=False)],
        )  # fmt: skip
        return max(int(m.best_iteration_ or s.max_rounds), 10)
    # scikit-learn: grow in steps with warm_start and watch month-4 log loss
    m = _make_hgb(s, 0, warm_start=True)
    best, best_rounds, rounds, step = np.inf, 0, 0, 25
    while rounds < s.max_rounds:
        rounds += step
        m.set_params(max_iter=rounds)
        m.fit(Xt, yt)
        loss = log_loss(yh, m.predict_proba(Xh)[:, 1], labels=[0, 1])
        if loss < best - 1e-6:
            best, best_rounds = loss, rounds
        elif rounds - best_rounds >= s.patience:
            break
    return max(best_rounds, 10)


def fit_gbm(
    train: pd.DataFrame,
    features: list[str] = FEATURES,
    rounds: int | None = None,
    settings: GBMSettings | None = None,
) -> FraudModel:
    """Fit on all training months. If `rounds` is None, choose it by early stopping on month 4."""
    settings = settings or GBMSettings()
    if rounds is None:
        inner = train[train["month"] < max(TRAIN_MONTHS)]
        hold = train[train["month"] == max(TRAIN_MONTHS)]
        rounds = _best_rounds(inner, hold, features, settings)
    X, y = prepare(train, features), train["is_fraud"].to_numpy()
    backend = gbm_backend()
    est = _make_lightgbm(settings, rounds) if backend == "lightgbm" else _make_hgb(settings, rounds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        est.fit(X, y)
    return FraudModel(est, list(features), backend, rounds)


def _baseline_frame(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for col in CATEGORIES:
        if col in X:
            X[col] = X[col].astype("object").fillna("missing").astype(str)
    return X


def fit_baseline(train: pd.DataFrame, features: list[str] = FEATURES) -> FraudModel:
    """Logistic regression with median imputation, missing flags and one-hot categoricals."""
    cats = [f for f in features if f in CATEGORIES]
    nums = [f for f in features if f not in CATEGORIES]
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                nums,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
        ]
    )
    pipe = Pipeline([("pre", pre), ("lr", LogisticRegression(max_iter=2000, C=0.5))])
    pipe.fit(_baseline_frame(train, features), train["is_fraud"].to_numpy())
    return FraudModel(pipe, list(features), "logistic")


# --- calibration ------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


@dataclass(frozen=True)
class PlattCalibrator:
    """Rescale scores so that, of applicants scored 4%, about 4% turn out to be fraud.

    Fits fraud ~ sigmoid(a·logit(raw) + b) on the calibration month. Two parameters only, so
    it cannot overfit a month that holds ~1,000 frauds, and it keeps the model's ranking.
    """

    a: float
    b: float

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw)[:, None], y)
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(self.a * _logit(raw) + self.b)))


# --- metrics ----------------------------------------------------------------------------


def tpr_at_fpr(y: np.ndarray, p: np.ndarray, fpr: float = 0.05) -> float:
    """Share of fraud caught when `fpr` of genuine applicants are flagged (the paper's metric)."""
    f, t, _ = roc_curve(y, p)
    return float(np.interp(fpr, f, t))


def model_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "tpr_at_5pct_fpr": tpr_at_fpr(y, p, 0.05),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 12) -> pd.DataFrame:
    """Equal-count buckets of predicted p: mean predicted vs observed fraud rate."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return pd.DataFrame(
        {
            "bucket": np.arange(bins),
            "mean_predicted": [float(p[c].mean()) for c in chunks],
            "observed": [float(y[c].mean()) for c in chunks],
            "n": [len(c) for c in chunks],
        }
    )


# --- one call: train, calibrate, score ---------------------------------------------------


@dataclass
class ScoredVariant:
    name: str
    model: FraudModel
    calibrator: PlattCalibrator
    scored: (
        pd.DataFrame
    )  # calibration + test months: application_id, month, age_group, is_fraud, p_raw, p


def train_and_score(
    name: str,
    splits: Splits,
    features: list[str] = FEATURES,
    kind: str = "gbm",
    rounds: int | None = None,
    settings: GBMSettings | None = None,
) -> ScoredVariant:
    if kind == "gbm":
        model = fit_gbm(splits.train, features, rounds, settings)
    else:
        model = fit_baseline(splits.train, features)
    cal_raw = model.predict_raw(splits.calibration)
    calibrator = PlattCalibrator.fit(cal_raw, splits.calibration["is_fraud"].to_numpy())
    later = pd.concat([splits.calibration, splits.test])
    raw = model.predict_raw(later)
    scored = later[["application_id", "month", "age_group", "is_fraud"]].assign(
        variant=name, p_raw=raw, p=calibrator(raw)
    )
    return ScoredVariant(name, model, calibrator, scored.reset_index(drop=True))
