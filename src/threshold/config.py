"""Paths, and the business assumptions from config/assumptions.yaml."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from threshold.economics import Economics

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "assumptions.yaml"
SQL_DIR = ROOT / "sql"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS = ROOT / "artifacts"
FAKE_ARTIFACTS = ROOT / "artifacts_fake"  # never committed: results from generated data
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
DOCS = ROOT / "docs"

TRAIN_MONTHS = (0, 1, 2, 3, 4)
CALIBRATION_MONTH = 5
TEST_MONTHS = (6, 7)

# field name -> (section, key) in assumptions.yaml
_YAML_KEYS: dict[str, tuple[str, str]] = {
    "V": ("economics", "good_customer_value"),
    "L": ("economics", "loss_per_fraud"),
    "s": ("verification", "cost_per_check"),
    "a_g": ("verification", "good_dropoff"),
    "a_f": ("verification", "fraud_dropoff"),
    "r": ("review", "fraud_catch_rate"),
    "f": ("review", "good_reject_rate"),
    "analyst_hourly_cost": ("review", "analyst_hourly_cost"),
    "minutes_per_case": ("review", "minutes_per_case"),
    "productive_share": ("review", "productive_share"),
    "hours_per_day": ("review", "hours_per_day"),
    "tooling_cost_per_case": ("review", "tooling_cost_per_case"),
    "team_size": ("review", "team_size"),
    "working_days_per_month": ("review", "working_days_per_month"),
}

LABELS: dict[str, str] = {
    "V": "Value of a good customer (£)",
    "L": "Loss per fraud (£)",
    "s": "Verification cost (£ per check)",
    "a_g": "Genuine drop-off at verification",
    "a_f": "Fraudster drop-off at verification",
    "r": "Share of fraud reviewers catch",
    "f": "Share of genuine reviewers reject",
    "analyst_hourly_cost": "Analyst cost (£ per hour, loaded)",
    "minutes_per_case": "Minutes per review",
    "productive_share": "Share of paid time on cases",
    "hours_per_day": "Paid hours per day",
    "tooling_cost_per_case": "Tooling cost (£ per review)",
    "team_size": "Review analysts",
    "working_days_per_month": "Working days per month",
}


@dataclass(frozen=True)
class Assumptions:
    """Every business number, with the review-team arithmetic derived from them."""

    V: float
    L: float
    s: float
    a_g: float
    a_f: float
    r: float
    f: float
    analyst_hourly_cost: float
    minutes_per_case: float
    productive_share: float
    hours_per_day: float
    tooling_cost_per_case: float
    team_size: int
    working_days_per_month: float

    @property
    def cases_per_analyst_day(self) -> float:
        return self.hours_per_day * 60 / self.minutes_per_case * self.productive_share

    @property
    def analyst_daily_cost(self) -> float:
        return self.analyst_hourly_cost * self.hours_per_day

    @property
    def review_time_cost(self) -> float:
        """Analyst time per case, including the unproductive share of their day."""
        return self.analyst_daily_cost / self.cases_per_analyst_day

    @property
    def c(self) -> float:
        return self.review_time_cost + self.tooling_cost_per_case

    def economics(self) -> Economics:
        return Economics(
            V=self.V, L=self.L, c=self.c, s=self.s, r=self.r, f=self.f, a_g=self.a_g, a_f=self.a_f
        )

    def review_capacity(self, months: float, team_size: float | None = None) -> float:
        """Review slots over `months` calendar months."""
        team = self.team_size if team_size is None else team_size
        return team * self.cases_per_analyst_day * self.working_days_per_month * months

    def with_(self, **changes: Any) -> Assumptions:
        return replace(self, **changes)

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@cache
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_assumptions(path: Path = CONFIG_PATH) -> Assumptions:
    cfg = load_config(path)
    values = {name: cfg[sec][key]["value"] for name, (sec, key) in _YAML_KEYS.items()}
    values["team_size"] = int(values["team_size"])
    return Assumptions(**values)


def assumption_ranges(path: Path = CONFIG_PATH) -> dict[str, tuple[float, float]]:
    """(low, high) for every assumption that has a plausible range."""
    cfg = load_config(path)
    out = {}
    for name, (sec, key) in _YAML_KEYS.items():
        entry = cfg[sec][key]
        if "low" in entry and "high" in entry:
            out[name] = (float(entry["low"]), float(entry["high"]))
    return out


def assumption_records(path: Path = CONFIG_PATH) -> list[dict[str, Any]]:
    """Flat rows for the assumptions table in the docs and the app."""
    cfg = load_config(path)
    rows = []
    for name, (sec, key) in _YAML_KEYS.items():
        e = cfg[sec][key]
        rows.append(
            {
                "symbol": name,
                "assumption": LABELS[name],
                "value": e["value"],
                "low": e.get("low"),
                "high": e.get("high"),
                "unit": e.get("unit", ""),
                "source": " ".join(str(e.get("source", "")).split()),
                "checked": e.get("checked"),
            }
        )
    return rows
