"""Shared pieces for every page: artifacts, the assumption sliders, chart styling."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # so the hosted app can import the package without installing

from threshold.config import (
    CALIBRATION_MONTH,
    LABELS,
    TEST_MONTHS,
    Assumptions,
    load_assumptions,
)
from threshold.histogram import ScoreHist, load_hist
from threshold.style import ACCENT, AXIS, GRID, INK, INK_2, MUTED, SURFACE

# --- artifacts ---------------------------------------------------------------------------


def artifacts_dir() -> Path | None:
    env = os.environ.get("THRESHOLD_ARTIFACTS")
    candidates = [Path(env)] if env else [ROOT / "artifacts", ROOT / "artifacts_fake"]
    return next((c for c in candidates if (c / "meta.json").exists()), None)


@st.cache_data(show_spinner=False)
def _read(folder: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(Path(folder) / f"{name}.parquet")


@st.cache_data(show_spinner=False)
def _read_json(folder: str, name: str) -> dict:
    return json.loads((Path(folder) / f"{name}.json").read_text())


class Data:
    def __init__(self, folder: Path):
        self.folder = str(folder)

    def table(self, name: str) -> pd.DataFrame:
        return _read(self.folder, name)

    def json(self, name: str) -> dict:
        return _read_json(self.folder, name)

    @property
    def meta(self) -> dict:
        return self.json("meta")

    def hist(self, variant: str = "full", months=(CALIBRATION_MONTH,), age_group=None) -> ScoreHist:
        return load_hist(self.table("score_hist"), variant, tuple(months), age_group)

    def val_test(self, variant: str = "full") -> tuple[ScoreHist, ScoreHist]:
        return self.hist(variant, (CALIBRATION_MONTH,)), self.hist(variant, TEST_MONTHS)


def page_setup(title: str) -> Data:
    st.set_page_config(page_title=f"Threshold · {title}", page_icon="⚖️", layout="wide")
    st.markdown(
        """<style>
        [data-testid="stMetricValue"] { font-size: 1.7rem; }
        .what-it-means { color: #52514e; font-size: 0.92rem; margin-top: -0.4rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    folder = artifacts_dir()
    if folder is None:
        st.error("No results found. Run `make all` (real data) or `make fake` (synthetic) first.")
        st.stop()
    data = Data(folder)
    if data.meta["data_source"] == "synthetic":
        st.warning(
            "**Synthetic data.** These results come from generated data with the real column "
            "names: they show the engine working, not anything about real applicants.",
            icon="⚠️",
        )
    return data


def means(text: str) -> None:
    """The one-sentence 'what this means' under every chart."""
    st.markdown(f'<p class="what-it-means">{text}</p>', unsafe_allow_html=True)


# --- assumptions in the sidebar ------------------------------------------------------------

SLIDERS: dict[str, tuple[float, float, float]] = {  # name: (min, max, step)
    "V": (10.0, 200.0, 5.0),
    "L": (100.0, 5000.0, 50.0),
    "s": (0.0, 10.0, 0.25),
    "a_g": (0.0, 0.6, 0.01),
    "a_f": (0.0, 1.0, 0.01),
    "r": (0.0, 1.0, 0.01),
    "f": (0.0, 0.5, 0.01),
    "analyst_hourly_cost": (10.0, 80.0, 1.0),
    "minutes_per_case": (2.0, 60.0, 1.0),
    "team_size": (0, 40, 1),
}
MORE = {
    "productive_share": (0.4, 1.0, 0.05),
    "tooling_cost_per_case": (0.0, 10.0, 0.5),
}
MAIN = ("V", "L", "team_size")


def _slider(container, name: str, bounds: tuple[float, float, float], default: float) -> float:
    store, widget = f"v_{name}", f"w_{name}"
    if store not in st.session_state:
        st.session_state[store] = default

    def keep() -> None:
        st.session_state[store] = st.session_state[widget]

    lo, hi, step = bounds
    is_int = name == "team_size"
    cast = int if is_int else float
    return container.slider(
        LABELS[name],
        min_value=cast(lo),
        max_value=cast(hi),
        value=cast(st.session_state[store]),
        step=cast(step),
        key=widget,
        on_change=keep,
    )


def assumptions_sidebar() -> Assumptions:
    """Sliders for every business assumption; values persist across pages."""
    defaults = load_assumptions()
    sb = st.sidebar
    sb.header("Business assumptions")
    values = {}
    for name in MAIN:
        values[name] = _slider(sb, name, SLIDERS[name], getattr(defaults, name))
    with sb.expander("Verification and review", expanded=False):
        for name in SLIDERS:
            if name not in MAIN:
                values[name] = _slider(st, name, SLIDERS[name], getattr(defaults, name))
    with sb.expander("Team arithmetic", expanded=False):
        for name, bounds in MORE.items():
            values[name] = _slider(st, name, bounds, getattr(defaults, name))
    if sb.button("Reset to defaults", use_container_width=True):
        for key in list(st.session_state):
            if key.startswith(("v_", "w_")):
                del st.session_state[key]
        st.rerun()
    a = defaults.with_(**values)
    sb.caption(
        f"Review cost c = £{a.c:.2f} per case · {a.cases_per_analyst_day:.0f} cases per analyst "
        f"per day · analyst day £{a.analyst_daily_cost:.0f}. Sources: `config/assumptions.yaml`."
    )
    return a


# --- charts --------------------------------------------------------------------------------


def style(fig: go.Figure, height: int = 420, **layout) -> go.Figure:
    settings = {
        "height": height,
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": INK,
            "size": 13,
        },
        "margin": {"l": 80, "r": 24, "t": 40, "b": 64},
        "showlegend": False,
        "hoverlabel": {"bgcolor": "white", "font_color": INK},
    }
    fig.update_layout(**(settings | layout))
    axis = {
        "gridcolor": GRID,
        "linecolor": AXIS,
        "zeroline": False,
        "tickfont_color": MUTED,
        "title_font_color": INK_2,
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def money_ticks(fig: go.Figure, lo: float, hi: float, axis: str = "y", n: int = 6) -> go.Figure:
    """Tick labels the way Finance writes them (−£250, £1.2k), which Plotly's prefix can't do."""
    from matplotlib.ticker import MaxNLocator

    from threshold.style import money

    ticks = [t for t in MaxNLocator(n).tick_values(lo, hi) if lo - 1e-9 <= t <= hi + 1e-9]
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    update(tickvals=ticks, ticktext=[money(t) for t in ticks], tickprefix="")
    return fig


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})


__all__ = ["ACCENT", "MUTED", "Data", "assumptions_sidebar", "means", "page_setup", "plot", "style"]
