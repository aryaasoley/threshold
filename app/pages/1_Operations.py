"""Operations: what is one more analyst worth, and how big should the review team be?"""

from __future__ import annotations

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

from threshold.policy import optimise, recommended_team, team_curve
from threshold.style import INK_2, money

data = page_setup("Operations")
a = assumptions_sidebar()
val, test = data.val_test()

st.title("Operations: sizing the review team")
curve = team_curve(val, test, a)
rec = recommended_team(curve)
pol = optimise(val, a.economics(), a.review_capacity(val.months))
now = (
    curve.set_index("analysts").loc[a.team_size] if a.team_size in set(curve["analysts"]) else None
)
nxt = curve.set_index("analysts").get("marginal_value_per_day", pd.Series()).get(a.team_size + 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analysts today", f"{a.team_size}")
c2.metric(
    "Recommended team",
    f"{rec}",
    f"{rec - a.team_size:+d} analysts" if rec != a.team_size else "no change",
    delta_color="off",
)
slot = pol.shadow_price + a.review_time_cost if pol.capacity_binds else 0.0
c3.metric(
    "Value of one more slot",
    money(slot),
    help="The gain on the best case left without a slot, before the analyst's time",
)
c4.metric(
    f"Analyst {a.team_size + 1} adds per day",
    money(nxt) if nxt is not None and pd.notna(nxt) else "–",
    f"vs {money(a.analyst_daily_cost)} daily cost",
    delta_color="off",
)

st.subheader("What each extra analyst adds per working day")
shown = curve.dropna(subset=["marginal_value_per_day"])
last = shown.loc[shown["marginal_value_per_day"] > 1e-6, "analysts"].max()
shown = shown[shown["analysts"] <= max(int(last if pd.notna(last) else 1) + 2, a.team_size + 2, 4)]
fig = go.Figure(
    go.Bar(
        x=shown["analysts"],
        y=shown["marginal_value_per_day"],
        marker_color=[ACCENT if w else MUTED for w in shown["worth_hiring"]],
        hovertemplate="Analyst %{x}: adds %{y:£,.0f} a day<extra></extra>",
    )
)
fig.add_hline(y=a.analyst_daily_cost, line_color=INK_2, line_width=1)
fig.add_annotation(
    x=shown["analysts"].max(),
    y=a.analyst_daily_cost,
    text=f"Daily cost {money(a.analyst_daily_cost)}",
    showarrow=False,
    yshift=12,
    xanchor="right",
    font={"color": INK_2},
)
fig.update_xaxes(title="The nth analyst", dtick=1, showgrid=False)
fig.update_yaxes(title="Value added per working day")
money_ticks(fig, 0, max(float(shown["marginal_value_per_day"].max()), a.analyst_daily_cost) * 1.1)
plot(style(fig, bargap=0.3))
means(
    f"Each analyst works the most valuable cases left, so each adds less than the last. Blue bars "
    f"pay for themselves; the recommendation is the last blue bar: {rec} analysts."
)

st.subheader("Reviews wanted vs capacity")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=curve["analysts"],
        y=curve["capacity_per_day"],
        mode="lines",
        line={"color": MUTED, "width": 2},
        name="Capacity",
        hovertemplate="%{x} analysts: %{y:,.0f} slots a day<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=curve["analysts"],
        y=curve["reviews_per_day"],
        mode="lines+markers",
        line={"color": ACCENT, "width": 2},
        marker={"size": 8},
        name="Reviews done",
        hovertemplate="%{x} analysts: %{y:,.0f} reviews a day<extra></extra>",
    )
)
for name, color, row in (
    ("Capacity", MUTED, curve.iloc[-1]),
    ("Reviews done", ACCENT, curve.iloc[-1]),
):
    y = row["capacity_per_day"] if name == "Capacity" else row["reviews_per_day"]
    fig.add_annotation(
        x=row["analysts"],
        y=y,
        text=f"<b>{name}</b>",
        showarrow=False,
        xanchor="right",
        yshift=12,
        font={"color": color},
    )
fig.update_xaxes(title="Analysts", dtick=1)
fig.update_yaxes(title="Cases per working day")
plot(style(fig, height=340))
means(
    "Where the blue line leaves the grey one, the team has more slots than worthwhile cases: "
    "extra analysts would sit idle on this queue."
)

with st.expander("The table behind the charts"):
    st.dataframe(
        pd.DataFrame(
            {
                "Analysts": curve["analysts"],
                "Slots per day": curve["capacity_per_day"].round(0),
                "Reviews per day": curve["reviews_per_day"].round(0),
                "Value of a slot": curve["slot_value"].map(money),
                "This analyst adds / day": curve["marginal_value_per_day"].map(money),
                "Profit net of team / day": curve["net_per_day"].map(money),
            }
        ),
        hide_index=True,
        width="stretch",
    )
st.caption(
    "Method: for each team size the policy is re-fitted on month 5 with that many review slots and "
    "scored on months 6–7. Salaries are paid whether or not a slot is used, so an analyst's value "
    "is the step up in profit before analyst time; hire while it beats a day's cost."
)
