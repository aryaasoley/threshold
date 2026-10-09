"""Pack scored applicants into a probability histogram.

Every number the policy reports is a sum over applicants, and every decision depends on
the applicant only through p. So instead of carrying 200,000 scored rows around we keep,
for ~1,000 probability buckets, how many applicants in each were genuine and how many
were fraud. The app re-optimises from that table in milliseconds and never sees raw data.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

N_BINS = 1000
_FLOOR = 1e-5


def bin_edges(n_bins: int = N_BINS) -> np.ndarray:
    """[0, 1e-5) then log-spaced up to 1: ~1.2% relative width, fine where cut-offs live."""
    return np.concatenate([[0.0], np.logspace(np.log10(_FLOOR), 0.0, n_bins)])


def bin_centres(n_bins: int = N_BINS) -> np.ndarray:
    e = bin_edges(n_bins)
    centres = np.sqrt(e[1:] * np.maximum(e[:-1], _FLOOR / 10))
    centres[0] = _FLOOR / 2
    return centres


def assign_bins(p: np.ndarray, n_bins: int = N_BINS) -> np.ndarray:
    """Bucket index for each probability."""
    e = bin_edges(n_bins)
    return np.clip(np.searchsorted(e, p, side="right") - 1, 0, n_bins - 1)


@dataclass(frozen=True)
class ScoreHist:
    """Counts of genuine and fraudulent applicants per probability bucket."""

    p: np.ndarray  # representative probability of each bucket
    n_good: np.ndarray
    n_fraud: np.ndarray
    months: float = 1.0  # calendar months covered; review capacity scales with this
    sum_p: np.ndarray | None = None  # Σ predicted p in each bucket (expected fraud)

    @property
    def n(self) -> np.ndarray:
        return self.n_good + self.n_fraud

    @property
    def applications(self) -> float:
        return float(self.n.sum())

    def __add__(self, other: ScoreHist) -> ScoreHist:
        sum_p = None
        if self.sum_p is not None and other.sum_p is not None:
            sum_p = self.sum_p + other.sum_p
        return ScoreHist(
            self.p,
            self.n_good + other.n_good,
            self.n_fraud + other.n_fraud,
            self.months + other.months,
            sum_p,
        )

    def resample(self, rng: np.random.Generator) -> ScoreHist:
        """Poisson bootstrap: equivalent to resampling applicants with replacement."""
        return replace(
            self,
            n_good=rng.poisson(self.n_good).astype(float),
            n_fraud=rng.poisson(self.n_fraud).astype(float),
            sum_p=None,
        )


def build_hist_table(scored: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Turn scored rows (variant, month, age_group, p, is_fraud) into the histogram table."""
    df = scored.assign(bin=assign_bins(scored["p"].to_numpy(), n_bins))
    out = (
        df.groupby(["variant", "month", "age_group", "bin"], observed=True)
        .agg(
            n_good=("is_fraud", lambda s: int((s == 0).sum())),
            n_fraud=("is_fraud", "sum"),
            sum_p=("p", "sum"),
        )
        .reset_index()
    )
    out["n_fraud"] = out["n_fraud"].astype(int)
    return out


def bins_table(n_bins: int = N_BINS) -> pd.DataFrame:
    e = bin_edges(n_bins)
    return pd.DataFrame(
        {"bin": np.arange(n_bins), "p_lo": e[:-1], "p_hi": e[1:], "p": bin_centres(n_bins)}
    )


def load_hist(
    table: pd.DataFrame,
    variant: str = "full",
    months: tuple[int, ...] = (6, 7),
    age_group: str | None = None,
    n_bins: int = N_BINS,
) -> ScoreHist:
    """Collapse the histogram table into one ScoreHist for the chosen slice."""
    mask = (table["variant"] == variant) & table["month"].isin(months)
    if age_group is not None:
        mask &= table["age_group"] == age_group
    sub = table.loc[mask].groupby("bin")[["n_good", "n_fraud", "sum_p"]].sum()
    full = sub.reindex(np.arange(n_bins), fill_value=0)
    return ScoreHist(
        p=bin_centres(n_bins),
        n_good=full["n_good"].to_numpy(dtype=float),
        n_fraud=full["n_fraud"].to_numpy(dtype=float),
        months=float(len(months)),
        sum_p=full["sum_p"].to_numpy(dtype=float),
    )
