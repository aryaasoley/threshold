"""Vendors: what each data source is worth, and the most to pay for it."""

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

from threshold.style import money
from threshold.vendors import vendor_values

data = page_setup("Vendors")
a = assumptions_sidebar()

st.title("Vendors: what each data source is worth")
live = vendor_values(data.table("score_hist"), a)
saved = data.table("vendor_values").set_index("variant")[["low", "high", "value_per_application"]]
live = live.join(saved.rename(columns={"value_per_application": "default_value"}), on="variant")

singles = live[live["variant"] != "minus_external"]
together = live[live["variant"] == "minus_external"]
external = singles[singles["external"]].sort_values("value_per_application", ascending=False)

cols = st.columns(len(external) + 1)
for col, row in zip(cols, external.itertuples(), strict=False):
    col.metric(f"{row.source}: most to pay per check", money(row.value_per_application))
if len(together):
    tg = together.iloc[0]
    cols[-1].metric(
        "All three vendors, bought together",
        money(tg["value_per_application"]),
        f"vs {money(external['value_per_application'].sum())} if you add them up",
        delta_color="off",
    )

st.subheader("Profit lost per application without each source")
order = live.sort_values("value_per_application")
fig = go.Figure(
    go.Bar(
        y=order["source"],
        x=order["value_per_application"],
        orientation="h",
        marker_color=[ACCENT if e else MUTED for e in order["external"]],
        hovertemplate="%{y}: %{x:£,.2f} per application<extra></extra>",
    )
)
has_range = order["low"].notna()
fig.add_trace(
    go.Scatter(
        y=order.loc[has_range, "source"],
        x=order.loc[has_range, "default_value"],
        mode="markers",
        marker={"color": "rgba(0,0,0,0)"},
        error_x={
            "type": "data",
            "symmetric": False,
            "color": "#52514e",
            "thickness": 1.2,
            "array": order.loc[has_range, "high"] - order.loc[has_range, "default_value"],
            "arrayminus": order.loc[has_range, "default_value"] - order.loc[has_range, "low"],
        },
        hovertemplate="95% range at default assumptions<extra></extra>",
    )
)
fig.add_vline(x=0, line_color="#c3c2b7", line_width=1)
fig.update_xaxes(title="£ per application = the break-even price per check")
span = pd.concat(
    [order["value_per_application"], order["low"], order["high"], pd.Series([0.0])]
).dropna()
money_ticks(fig, float(span.min()), float(span.max()), axis="x")
fig.update_yaxes(showgrid=False)
plot(style(fig, height=380))
means(
    "Blue sources are bought from vendors. A source's value is the profit you would lose without "
    "it, so paying more than that per check loses money. Whiskers show the 95% range at the default "
    "assumptions; the bars move with the sliders."
)

st.subheader("How to use this in a negotiation")
if len(external):
    top = external.iloc[0]
    st.markdown(
        f"""
- **Walk-away price.** {top["source"]} is worth about **{money(top["value_per_application"])}** per
  application at these assumptions. That is the ceiling, not the target.
- **Ask for a free retro trial.** Have the vendor score past applications so the value can be measured
  on our own data before signing.
- **Volume tiers.** Propose lower prices at higher volumes; the value per check does not grow with volume.
- **Only call it when it can change the decision.** Most applicants sit deep in the approve band, where
  no extra data would flip the outcome. Paying only for borderline cases cuts the bill without losing much value.
"""
    )
st.caption(
    "Method: the model is retrained without each supplier's features, recalibrated on month 5, the "
    "policy is re-fitted, and profit on months 6–7 is compared with the full model. Values do not "
    "add up: sources that overlap are worth less together, sources that complement each other more."
)
with st.expander("Which features come from which source"):
    from threshold.model import FEATURE_GROUPS, GROUP_LABELS

    st.dataframe(
        pd.DataFrame(
            [
                {"Source": GROUP_LABELS[g], "Features": ", ".join(f)}
                for g, f in FEATURE_GROUPS.items()
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
