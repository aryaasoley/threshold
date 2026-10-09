"""Policy: move an assumption and watch the whole policy re-optimise."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _common import (
    ACCENT,
    MUTED,
    assumptions_sidebar,
    means,
    money_ticks,
    page_setup,
    plot,
    style,
)

from threshold.economics import ACTIONS, approve_decline_breakeven, expected_values
from threshold.policy import POLICY_LABELS, compare, decision_bands, describe_bands, tradeoff_curves
from threshold.style import ACTION_COLORS, INK_2, money, pct
from threshold.uncertainty import quick_ranges

data = page_setup("Policy")
a = assumptions_sidebar()
econ = a.economics()
val, test = data.val_test()

st.title("Threshold")
st.markdown(
    "**How many good customers should a bank turn away to stop one fraudster?** "
    "Each applicant's fraud probability becomes approve, verify, review or decline. Cut-offs "
    "are fitted on month 5 and scored on months 6–7, which the engine has never seen."
)

comp = compare(val, test, a)
t, best = comp.table.loc["threshold"], comp.table.loc["best_cutoff"]
ranges = st.cache_data(show_spinner=False)(quick_ranges)(val, test, a)
days = test.months * a.working_days_per_month

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Good customers approved",
    pct(t["genuine_onboarded_share"]),
    f"{(t['genuine_onboarded_share'] - best['genuine_onboarded_share']) * 100:+.1f} pts vs one cut-off",
)
c2.metric(
    "Fraud stopped",
    pct(t["fraud_stopped_share"]),
    f"{(t['fraud_stopped_share'] - best['fraud_stopped_share']) * 100:+.1f} pts vs one cut-off",
)
r = ranges.loc["threshold_profit_per_1k"]
c3.metric(
    "Profit per 1,000 applications",
    money(t["profit_per_1k"]),
    f"{money(t['profit_per_1k'] - best['profit_per_1k'])} vs one cut-off",
    help=f"95% bootstrap range {money(r['low'])} to {money(r['high'])}",
)
wanted, cap = comp.reviews_wanted / days, comp.review_capacity / days
if wanted > cap * 1.02:
    queue = "queue overflows"
elif wanted >= cap * 0.98:
    queue = "at capacity"
else:
    queue = "spare capacity"
c4.metric(
    "Reviews needed vs capacity, per day",
    f"{wanted:,.0f} / {cap:,.0f}",
    queue,
    delta_color="inverse" if queue == "queue overflows" else "off",
)
u = ranges.loc["uplift_per_1k"]
means(
    f"Range on the extra profit vs the best single cut-off: {money(u['low'])} to "
    f"{money(u['high'])} per 1,000 applications (95% bootstrap over both months)."
)

# --- the straight-line chart --------------------------------------------------------------
st.subheader("Each action's value is a straight line in p")
bands = decision_bands(comp.policy)
decline_from = next((b.start for b in bands if b.name == "decline"), 1.0)
p_max = float(min(1.0, max(0.10, decline_from * 1.8)))
p = np.linspace(0, p_max, 801)
ev = expected_values(p, econ)
fig = go.Figure()
for b in bands:
    if b.start >= p_max:
        continue
    fig.add_vrect(
        x0=b.start,
        x1=min(b.end, p_max),
        fillcolor=ACTION_COLORS[b.name],
        opacity=0.09,
        line_width=0,
        layer="below",
    )
for j, action in enumerate(ACTIONS):
    fig.add_trace(
        go.Scatter(
            x=p,
            y=ev[:, j],
            mode="lines",
            name=action.capitalize(),
            line={"color": ACTION_COLORS[action], "width": 2.5},
            hovertemplate=f"{action}: %{{y:£,.2f}} at p = %{{x:.2%}}<extra></extra>",
        )
    )
    k = int(len(p) * (0.42 if action == "approve" else 0.94))
    fig.add_annotation(
        x=p[k],
        y=ev[k, j],
        text=f"<b>{action.capitalize()}</b>",
        showarrow=False,
        yshift=-14 if action == "approve" else 12,
        font={"color": INK_2},
    )
p_star = approve_decline_breakeven(econ)
if p_star < p_max:
    fig.add_vline(x=p_star, line_color=MUTED, line_width=1)
    fig.add_annotation(
        x=p_star,
        y=1,
        yref="paper",
        text=f"V/(V+L) = {p_star:.1%}",
        showarrow=False,
        xanchor="left",
        xshift=4,
        font={"color": INK_2, "size": 12},
    )
fig.update_xaxes(
    title="Applicant's probability of being a fraudster (p)", tickformat=".0%", range=[0, p_max]
)
fig.update_yaxes(title="Expected £ per applicant")
money_ticks(fig, float(ev[:, :].min()), float(ev.max()))
plot(style(fig, hovermode="x unified"))
binding = (
    f" On month 5, where the cut-offs were fitted, the review team was the bottleneck: one more "
    f"slot was worth {money(comp.policy.shadow_price + a.review_time_cost)}, so the review band was "
    "narrowed to the cases where a review adds most."
    if comp.policy.capacity_binds
    else " On month 5 the review team had spare capacity, so every case where a review pays gets one."
)
means(f"Shaded bands are the decisions: {describe_bands(bands)}.{binding}")

# --- trade-off ------------------------------------------------------------------------------
st.subheader("Fraud losses against good customers approved")
curves = st.cache_data(show_spinner=False)(tradeoff_curves)(val, test, a, 40)
apps = t["applications"]
fig = go.Figure()
for family, color in (("single cut-off", MUTED), ("Threshold", ACCENT)):
    c = curves[curves["family"] == family].sort_values("genuine_onboarded_share")
    fig.add_trace(
        go.Scatter(
            x=c["genuine_onboarded_share"],
            y=c["fraud_loss"] / apps * 1000,
            mode="lines",
            line={"color": color, "width": 2.5},
            name=family.capitalize(),
            hovertemplate=f"{family}: %{{x:.1%}} approved, %{{y:£,.0f}} lost per 1,000<extra></extra>",
        )
    )
for key, color, label in (
    ("paper_cutoff", MUTED, "Cut-off at 5% FPR"),
    ("best_cutoff", MUTED, "Best single cut-off"),
    ("threshold", ACCENT, "Your settings"),
):
    row = comp.table.loc[key]
    fig.add_trace(
        go.Scatter(
            x=[row["genuine_onboarded_share"]],
            y=[row["fraud_loss"] / apps * 1000],
            mode="markers+text",
            marker={"size": 12, "color": color, "line": {"color": "white", "width": 2}},
            text=[label],
            textposition="top left",
            textfont={"color": INK_2},
            name=label,
            hovertemplate=f"{label}: %{{x:.1%}} approved, %{{y:£,.0f}} lost per 1,000<extra></extra>",
        )
    )
lo = float(min(comp.table.loc["paper_cutoff", "genuine_onboarded_share"], 0.9)) - 0.02
visible = curves.loc[curves["genuine_onboarded_share"] >= lo, "fraud_loss"] / apps * 1000
fig.update_xaxes(title="Genuine applicants onboarded", tickformat=".0%", range=[lo, 1.0])
fig.update_yaxes(
    title="Fraud losses per 1,000 applications",
    range=[0, float(visible.max()) * 1.05],
)
money_ticks(fig, 0, float(visible.max()) * 1.05)
fig.add_annotation(
    x=lo,
    y=0,
    xanchor="left",
    yanchor="bottom",
    showarrow=False,
    align="left",
    text=f"<b style='color:{ACCENT}'>Threshold</b><br><b style='color:{MUTED}'>Single cut-off</b>",
)
plot(style(fig))
means(
    "Every point on a curve is a policy. Lower and further right is better: the gap between the "
    "curves is the good customers a single cut-off turns away for no reduction in fraud."
)

# --- the table and the download ---------------------------------------------------------
st.subheader("Against the simple alternatives, on the test months")
show = comp.table.copy()
show.index = [POLICY_LABELS[i] for i in show.index]
table = pd.DataFrame(
    {
        "Profit per 1,000": show["profit_per_1k"].map(money),
        "Good customers approved": show["genuine_onboarded_share"].map(pct),
        "Fraud stopped": show["fraud_stopped_share"].map(pct),
        "Approve": show["approve_rate"].map(pct),
        "Verify": show["verify_rate"].map(pct),
        "Review": show["review_rate"].map(pct),
        "Decline": show["decline_rate"].map(pct),
    }
)
st.dataframe(table, width="stretch")
means(
    "Every policy's cut-offs were fixed on month 5; nothing here was tuned on the months it is "
    "scored on."
)

policy_doc = {
    "assumptions": a.to_dict(),
    "review_cost_per_case": a.c,
    "bands": [{"action": b.name, "from_p": b.start, "to_p": b.end} for b in bands],
    "shadow_price_per_review_slot": comp.policy.shadow_price,
    "results_on_test_months": {k: float(v) for k, v in t.items()},
    "fitted_on_month": 5,
    "data_source": data.meta["data_source"],
}
st.sidebar.download_button(
    "Download this policy",
    json.dumps(policy_doc, indent=2),
    file_name="threshold_policy.json",
    mime="application/json",
    width="stretch",
)
