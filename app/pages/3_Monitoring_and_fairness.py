"""Monitoring and fairness: is the policy still working, and who bears the checks?"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _common import ACCENT, MUTED, assumptions_sidebar, means, page_setup, plot, style

from threshold.fairness import fairness_table
from threshold.monitoring import monthly_monitor
from threshold.style import ACTION_COLORS, STATUS_LABELS, pct
from threshold.vendors import VARIANT_LABELS

data = page_setup("Monitoring and fairness")
a = assumptions_sidebar()
hist = data.table("score_hist")

st.title("Monitoring and fairness")
st.subheader("Month by month, with the policy fixed on month 5")
summary, bands = monthly_monitor(hist, a)
table = pd.DataFrame(
    {
        "Month": [
            f"{int(m)} ({r})" for m, r in zip(summary["month"], summary["role"], strict=True)
        ],
        "Approved": summary["approve_rate"].map(pct),
        "Fraud rate": summary["fraud_rate"].map(lambda x: pct(x, 2)),
        "Fraud stopped": summary["fraud_stopped_share"].map(pct),
        "Score drift (PSI)": [
            f"{STATUS_LABELS[s]} · {v:.3f}"
            for s, v in zip(summary["score_status"], summary["score_psi"], strict=True)
        ],
        "Actual vs expected fraud": [
            f"{STATUS_LABELS[s]} · {g:+.0%}"
            for s, g in zip(summary["calibration_status"], summary["calibration_gap"], strict=True)
        ],
        "Review queue": [
            f"{STATUS_LABELS[s]} · {w:,.0f} / {c:,.0f}"
            for s, w, c in zip(
                summary["capacity_status"],
                summary["reviews_wanted"],
                summary["review_capacity"],
                strict=True,
            )
        ],
    }
)
st.dataframe(table, hide_index=True, use_container_width=True)
means(
    "● Stable, ▲ Watch, ■ Act. PSI under 0.10 is stable, 0.10–0.25 watch, over 0.25 act. PSI needs "
    "no fraud labels, which arrive weeks late in practice, so it is the early warning."
)

st.subheader("Actual vs expected fraud in each band")
fig = go.Figure()
test_bands = bands[bands["month"] != bands["month"].min()]
for action in ("approve", "verify", "review", "decline"):
    d = test_bands[test_bands["action"] == action]
    fig.add_trace(
        go.Bar(
            x=[f"Month {m}" for m in d["month"]],
            y=d["actual_fraud"],
            name=action.capitalize(),
            marker_color=ACTION_COLORS[action],
            offsetgroup=action,
            hovertemplate=f"{action}: %{{y:,.0f}} actual frauds<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[f"Month {m}" for m in d["month"]],
            y=d["expected_fraud"],
            mode="markers",
            marker={"symbol": "line-ew", "size": 22, "line": {"width": 3, "color": "#0b0b0b"}},
            offsetgroup=action,
            name="Expected (calibrated scores)",
            legendgroup="expected",
            showlegend=action == "approve",
            hovertemplate=f"{action}: %{{y:,.1f}} expected<extra></extra>",
        )
    )
fig.update_layout(
    barmode="group", scattermode="group", showlegend=True, legend={"orientation": "h", "y": 1.12}
)
fig.update_yaxes(title="Frauds")
plot(style(fig, height=360, showlegend=True))
means(
    "Bars are the frauds that actually arrived in each band; black ticks are what the calibrated "
    "scores expected. Bars well above their ticks mean the model is under-predicting: recalibrate."
)

psi = data.table("feature_psi")
drift = psi[psi["month"] != psi["month"].min()].pivot_table(
    index="feature", columns="month", values="psi"
)
drift = drift.loc[drift.max(axis=1).sort_values(ascending=False).index[:8]]
st.subheader("Features that moved most (PSI vs the training months)")
fig = go.Figure()
for col, color in zip(drift.columns, (MUTED, ACCENT), strict=False):
    fig.add_trace(
        go.Bar(
            y=drift.index,
            x=drift[col],
            orientation="h",
            name=f"Month {col}",
            marker_color=color,
            hovertemplate=f"Month {col}: PSI %{{x:.3f}}<extra></extra>",
        )
    )
fig.add_vline(x=0.10, line_color="#fab219", line_width=1)
fig.add_vline(x=0.25, line_color="#d03b3b", line_width=1)
fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.1})
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_xaxes(title="PSI")
plot(style(fig, height=380, showlegend=True))
means(
    "Amber line at 0.10 (watch), red at 0.25 (act). Feature drift explains score drift before labels arrive."
)

st.divider()
st.subheader("Fairness: genuine applicants checked or declined, by age group")
variant = st.radio(
    "Model", ["full", "no_age"], format_func=lambda v: VARIANT_LABELS[v], horizontal=True
)
fair = fairness_table(hist, a, variant)
pick = fair[fair["metric"] == "genuine_checked_rate"]
fig = go.Figure()
for group, col, color in (("Under 50", "under_50", MUTED), ("50+", "over_50", ACCENT)):
    fig.add_trace(
        go.Bar(
            x=pick["policy_label"],
            y=pick[col],
            name=group,
            marker_color=color,
            text=[pct(v) for v in pick[col]],
            textposition="outside",
            hovertemplate=f"{group}: %{{y:.1%}}<extra></extra>",
        )
    )
fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.12})
fig.update_yaxes(title="Genuine applicants given any check or declined", tickformat=".0%")
plot(style(fig, height=360, showlegend=True))
flags = pick[pick["flag"]]
ratio_txt = ", ".join(f"{r.policy_label}: {r.ratio:.2f}×" for r in pick.itertuples())
means(
    f"Ratio of 50+ to under-50 rates — {ratio_txt}. "
    + (
        "⚑ Outside the 0.8×–1.25× band: escalate to Compliance."
        if len(flags)
        else "Within the 0.8×–1.25× band."
    )
)
with st.expander("Every rate, every policy"):
    st.dataframe(
        fair.assign(
            under_50=fair["under_50"].map(pct),
            over_50=fair["over_50"].map(pct),
            ratio=fair["ratio"].map(lambda x: f"{x:.2f}×"),
        )[["policy_label", "metric_label", "under_50", "over_50", "ratio", "flag"]],
        hide_index=True,
        use_container_width=True,
    )
st.caption(
    "Age is a protected characteristic under the Equality Act 2010, so the policy does not use "
    "different cut-offs by age. The gap is shown and flagged instead. Switch to 'All data except "
    "age' to see whether removing age closes it: other features can stand in for age."
)

st.divider()
st.subheader("Approval and fraud rates by segment (default assumptions, from SQL)")
seg = data.table("kpi_policy_segments")
dim = st.selectbox("Segment by", ["channel", "device", "age group"])
view = seg[seg["dimension"] == dim].copy()
view["Approve"] = view["approve_rate"].map(pct)
view["Verify"] = view["verify_rate"].map(pct)
view["Review"] = view["review_rate"].map(pct)
view["Decline"] = view["decline_rate"].map(pct)
view["Fraud rate"] = view["fraud_rate"].map(lambda x: pct(x, 2))
st.dataframe(
    view[
        ["month", "segment", "applications", "Approve", "Verify", "Review", "Decline", "Fraud rate"]
    ],
    hide_index=True,
    use_container_width=True,
)
means(
    "Built by sql/04_segments.sql from the scored decisions; this is the table a Power BI page would sit on."
)
