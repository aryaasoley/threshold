"""Turn the artifacts into figures and Markdown: results, model card, assumptions table.

    python -m threshold.report                          # from artifacts/ → reports/, README
    python -m threshold.report --artifacts artifacts_fake   # synthetic run → artifacts_fake/report/

Results from synthetic data never touch reports/ or the README.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from threshold.config import (
    ARTIFACTS,
    CALIBRATION_MONTH,
    DOCS,
    LABELS,
    REPORTS,
    ROOT,
    TEST_MONTHS,
    assumption_records,
    load_assumptions,
)
from threshold.economics import ACTIONS, approve_decline_breakeven, expected_values
from threshold.histogram import load_hist
from threshold.policy import POLICY_LABELS, apply, optimise
from threshold.style import (
    ACCENT,
    ACTION_COLORS,
    AXIS,
    GRID,
    INK,
    INK_2,
    MUTED,
    SURFACE,
    count,
    money,
    pct,
)
from threshold.vendors import VARIANT_LABELS


class Artifacts:
    """Lazy access to everything the pipeline wrote."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.meta = json.loads((folder / "meta.json").read_text())
        self.metrics = json.loads((folder / "model_metrics.json").read_text())

    def __getattr__(self, name: str) -> pd.DataFrame:
        path = self.folder / f"{name}.parquet"
        if not path.exists():
            raise AttributeError(name)
        df = pd.read_parquet(path)
        setattr(self, name, df)
        return df

    @property
    def synthetic(self) -> bool:
        return self.meta["data_source"] == "synthetic"


# --- figures -----------------------------------------------------------------------------


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "lines.linewidth": 2,
            "legend.frameon": False,
            "axes.axisbelow": True,
        }
    )


def _money_axis(ax, axis: str = "y") -> None:
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: money(v))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def fig_value_lines(art: Artifacts, out: Path) -> Path:
    """The chart that explains the whole engine."""
    a = load_assumptions()
    econ = a.economics()
    val = load_hist(art.score_hist, "full", (CALIBRATION_MONTH,))
    pol = optimise(val, econ, a.review_capacity(val.months))
    p_star = approve_decline_breakeven(econ)
    p_max = 0.30
    p = np.linspace(0, p_max, 1201)
    ev = expected_values(p, econ)
    decided = apply(pol, p).argmax(axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    start, row = 0, 0
    for i in range(1, len(p) + 1):
        if i == len(p) or decided[i] != decided[start]:
            action = ACTIONS[decided[start]]
            ax.axvspan(p[start], p[i - 1], color=ACTION_COLORS[action], alpha=0.10, lw=0)
            mid = (p[start] + p[i - 1]) / 2
            narrow = (p[i - 1] - p[start]) < 0.06 * p_max
            row = (row + 1) % 2 if narrow else 0
            ax.text(
                mid,
                1.02 + 0.06 * row,
                action,
                transform=ax.get_xaxis_transform(),
                ha="left" if narrow else "center",
                va="bottom",
                fontsize=9,
                color=INK_2,
            )
            start = i
    for j, action in enumerate(ACTIONS):
        ax.plot(p, ev[:, j], color=ACTION_COLORS[action], lw=2)
        at, dy = (0.40, -14) if action == "approve" else (0.90, 6)
        k = int(np.argmin(np.abs(p - p_max * at)))
        ax.annotate(
            action.capitalize(),
            (p[k], ev[k, j]),
            xytext=(0, dy),
            textcoords="offset points",
            color=INK,
            fontsize=9,
            fontweight="bold",
        )
    ax.axvline(p_star, color=MUTED, lw=1)
    ax.annotate(
        f"V/(V+L) = {pct(p_star)}",
        (p_star, ax.get_ylim()[1]),
        xytext=(4, -14),
        textcoords="offset points",
        fontsize=9,
        color=INK_2,
    )
    ax.set_xlim(0, p_max)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _money_axis(ax)
    ax.set_xlabel("Applicant's probability of being a fraudster (p)")
    ax.set_ylabel("Expected value per applicant")
    ax.set_title(
        "Each action's value is a straight line; the highest wins, within review capacity", pad=38
    )
    fig.tight_layout()
    path = out / "value_lines.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_tradeoff(art: Artifacts, out: Path) -> Path:
    curves = art.tradeoff.copy()
    comp = art.policy_comparison.set_index("policy")
    apps = comp.loc["threshold", "applications"]
    curves["loss_per_1k"] = curves["fraud_loss"] / apps * 1000
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cut = curves[curves["family"] == "single cut-off"].sort_values("genuine_onboarded_share")
    thr = curves[curves["family"] == "Threshold"].sort_values("genuine_onboarded_share")
    ax.plot(cut["genuine_onboarded_share"], cut["loss_per_1k"], color=MUTED, lw=2)
    ax.plot(thr["genuine_onboarded_share"], thr["loss_per_1k"], color=ACCENT, lw=2)
    for key, color, label in (
        ("best_cutoff", MUTED, "Best single cut-off"),
        ("paper_cutoff", MUTED, "Cut-off at 5% FPR"),
        ("threshold", ACCENT, "Threshold, default assumptions"),
    ):
        row = comp.loc[key]
        x, y = row["genuine_onboarded_share"], row["fraud_loss"] / apps * 1000
        ax.scatter([x], [y], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=5)
        above = key == "best_cutoff"
        ax.annotate(
            label,
            (x, y),
            xytext=(-8, 10) if above else (8, -16),
            textcoords="offset points",
            fontsize=9,
            ha="right" if above else "left",
        )
    lo = min(comp.loc["paper_cutoff", "genuine_onboarded_share"], 0.9) - 0.02
    ax.set_xlim(lo, 1.0)
    visible = curves[curves["genuine_onboarded_share"] >= lo]["loss_per_1k"]
    ax.set_ylim(0, float(visible.max()) * 1.05)
    ax.text(0.02, 0.06, "Threshold", color=ACCENT, transform=ax.transAxes, fontweight="bold")
    ax.text(0.02, 0.12, "Single cut-off", color=MUTED, transform=ax.transAxes, fontweight="bold")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _money_axis(ax)
    ax.set_xlabel("Genuine applicants onboarded")
    ax.set_ylabel("Fraud losses per 1,000 applications")
    ax.set_title("Lower and further right is better: same fraud loss, more good customers")
    fig.tight_layout()
    path = out / "tradeoff.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_team(art: Artifacts, out: Path) -> Path:
    curve = art.team_curve.dropna(subset=["marginal_value_per_day"])
    last = curve.loc[curve["marginal_value_per_day"] > 1e-6, "analysts"].max()
    curve = curve[curve["analysts"] <= max(int(last if pd.notna(last) else 1) + 1, 3)]
    cost = float(curve["analyst_daily_cost"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = [ACCENT if w else MUTED for w in curve["worth_hiring"]]
    ax.bar(curve["analysts"], curve["marginal_value_per_day"], color=colors, width=0.7)
    ax.axhline(cost, color=INK_2, lw=1)
    ax.text(
        0.99,
        0.97,
        f"Line: an analyst's daily cost, {money(cost)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=INK_2,
    )
    rec = int(art.meta["recommended_team"])
    ax.set_xticks(curve["analysts"])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Analyst number (the nth analyst hired)")
    ax.set_ylabel("Value added per working day")
    _money_axis(ax)
    ax.set_title(f"Each extra analyst adds less; hire while the bar clears the line (→ {rec})")
    fig.tight_layout()
    path = out / "analyst_value.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_vendors(art: Artifacts, out: Path) -> Path:
    v = art.vendor_values.sort_values("value_per_application")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    y = np.arange(len(v))
    colors = [ACCENT if e else MUTED for e in v["external"]]
    ax.barh(y, v["value_per_application"], color=colors, height=0.6)
    ax.errorbar(
        v["value_per_application"],
        y,
        xerr=[v["value_per_application"] - v["low"], v["high"] - v["value_per_application"]],
        fmt="none",
        ecolor=INK_2,
        elinewidth=1,
        capsize=3,
    )
    ax.set_yticks(y, v["source"])
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(axis="y", visible=False)
    _money_axis(ax, "x")
    ax.set_xlabel("Profit lost per application without it = the most to pay per check")
    ax.set_title("What each data source is worth per application", pad=20)
    ax.text(
        0,
        1.01,
        "Blue: bought from a vendor. Whiskers: 95% bootstrap range.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
        va="bottom",
    )
    fig.tight_layout()
    path = out / "vendor_value.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_tornado(art: Artifacts, out: Path) -> Path:
    t = art.tornado.sort_values("swing")
    base = float(t["base"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    y = np.arange(len(t))
    lo = np.minimum(t["metric_at_low"], t["metric_at_high"])
    hi = np.maximum(t["metric_at_low"], t["metric_at_high"])
    ax.barh(y, hi - lo, left=lo, color=ACCENT, height=0.55, alpha=0.85)
    ax.axvline(base, color=INK_2, lw=1)
    ax.axvline(0, color="#d03b3b", lw=1)
    for yi, (_, row) in zip(y, t.iterrows(), strict=True):
        ends = sorted(
            [(row["metric_at_low"], row["low_value"]), (row["metric_at_high"], row["high_value"])]
        )
        for (x, value), side in zip(ends, (-1, 1), strict=True):
            ax.annotate(
                f"{value:g}",
                (x, yi),
                xytext=(5 * side, 0),
                textcoords="offset points",
                ha="right" if side < 0 else "left",
                va="center",
                fontsize=8,
                color=INK_2,
            )
    ax.set_yticks(y, [LABELS[n] for n in t["assumption"]])
    ax.grid(axis="y", visible=False)
    _money_axis(ax, "x")
    ax.set_xlabel("Extra profit vs best single cut-off, per 1,000 applications")
    ax.set_xlim(min(0, float(lo.min())) - 0.12 * float(hi.max()), float(hi.max()) * 1.12)
    ax.set_title("Which assumption moves the answer most", pad=20)
    ax.text(
        0,
        1.01,
        "Each end is labelled with the value that produced it. Grey: defaults. Red: no advantage.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK_2,
        va="bottom",
    )
    fig.tight_layout()
    path = out / "sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def fig_reliability(art: Artifacts, out: Path) -> Path:
    cal = art.calibration
    cal = cal[cal["model"] == "gbm"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    top = float(max(cal["mean_predicted"].max(), cal["observed"].max())) * 1.05
    ax.plot([0, top], [0, top], color=AXIS, lw=1)
    for stage, color, label in (
        ("raw", MUTED, "Before calibration"),
        ("calibrated", ACCENT, "After (Platt, month 5)"),
    ):
        d = cal[cal["stage"] == stage]
        ax.plot(
            d["mean_predicted"], d["observed"], color=color, marker="o", ms=5, lw=1.5, label=label
        )
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Predicted fraud rate")
    ax.set_ylabel("Observed fraud rate (test months)")
    ax.set_title("Calibration on the test months")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = out / "calibration.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


# --- markdown ----------------------------------------------------------------------------


def _range(row: pd.Series, fmt) -> str:
    return f"{fmt(row['estimate'])} ({fmt(row['low'])} to {fmt(row['high'])})"


def headline_lines(art: Artifacts) -> list[str]:
    h = art.headline.set_index("metric")
    v = art.vendor_values
    vendors_ext = v[v["external"] & (v["variant"] != "minus_external")]
    best = vendors_ext.iloc[0] if len(vendors_ext) else None
    curve = art.team_curve
    rec = int(art.meta["recommended_team"])
    nxt = curve.loc[curve["analysts"] == rec + 1]
    lines = [
        f"- **{_range(h.loc['more_genuine_approved'], pct)}** more genuine applicants onboarded at the "
        f"same fraud loss as the best single cut-off "
        f"({_range(h.loc['fewer_turned_away'], pct)} fewer turned away)",
        f"- **{_range(h.loc['profit_uplift_per_1k'], money)}** extra profit per 1,000 applications "
        "vs the best single cut-off",
    ]
    if best is not None:
        lines.append(
            f"- {best['source']} worth **{money(best['value_per_application'])}** per application "
            f"({money(best['low'])} to {money(best['high'])}) → the most to pay that vendor per check"
        )
    if len(nxt):
        lines.append(
            f"- Recommended review team: **{rec} analysts**. Analyst {rec + 1} would add "
            f"{money(float(nxt['marginal_value_per_day'].iloc[0]))} a day against a "
            f"{money(float(nxt['analyst_daily_cost'].iloc[0]))} daily cost, so hiring stops there"
        )
    return lines


def results_markdown(art: Artifacts, fig_dir: str) -> str:
    m, meta = art.metrics, art.meta
    comp = art.policy_comparison.set_index("policy")
    banner = (
        "> ⚠️ **Synthetic data.** These numbers come from generated data with the real column "
        "names. They test the code and mean nothing about real applicants.\n\n"
        if art.synthetic
        else ""
    )
    rows = []
    for key, label in POLICY_LABELS.items():
        r = comp.loc[key]
        rows.append(
            f"| {label} | {money(r['profit_per_1k'])} | {pct(r['genuine_onboarded_share'])} | "
            f"{pct(r['fraud_stopped_share'])} | {pct(r['approve_rate'])} / {pct(r['verify_rate'])} / "
            f"{pct(r['review_rate'])} / {pct(r['decline_rate'])} |"
        )
    gbm, base = m["gbm"]["calibrated"], m["baseline"]["calibrated"]
    team = art.team_curve
    team_rows = [
        f"| {int(r.analysts)} | {r.capacity_per_day:.0f} | {r.reviews_per_day:.0f} | "
        f"{money(r.marginal_value_per_day) if pd.notna(r.marginal_value_per_day) else '–'} | "
        f"{'yes' if r.worth_hiring else 'no'} |"
        for r in team.itertuples()
    ]
    vend = [
        f"| {r.source} | {'vendor' if r.external else 'in-house'} | {money(r.value_per_application)} "
        f"| {money(r.low)} to {money(r.high)} |"
        for r in art.vendor_values.itertuples()
    ]
    fair = art.fairness
    fair = fair[(fair["metric"] == "genuine_checked_rate")]
    fair_rows = [
        f"| {VARIANT_LABELS[r.variant]} | {r.policy_label} | {pct(r.under_50)} | {pct(r.over_50)} | "
        f"{r.ratio:.2f}× {'⚑' if r.flag else ''} |"
        for r in fair.itertuples()
    ]
    mon = art.monitoring
    mon_rows = [
        f"| {int(r.month)} ({r.role}) | {pct(r.approve_rate)} | {pct(r.fraud_rate, 2)} | "
        f"{pct(r.fraud_stopped_share)} | {r.score_psi:.3f} ({r.score_status}) | "
        f"{r.calibration_gap:+.0%} ({r.calibration_status}) | {r.reviews_wanted:,.0f} / {r.review_capacity:,.0f} |"
        for r in mon.itertuples()
    ]
    psi = art.feature_psi
    drifted = psi[psi["month"].isin(TEST_MONTHS)].sort_values("psi", ascending=False).head(5)
    drift_rows = [f"| {r.feature} | {int(r.month)} | {r.psi:.3f} |" for r in drifted.itertuples()]
    timings = meta["timings_seconds"]
    return f"""# Results

{banner}Generated by `make report` on {meta["created_at"]}. Data: `{meta["data_source"]}`,
{count(meta["applications"])} applications; the test months ({", ".join(map(str, TEST_MONTHS))}) hold
{count(meta["test_applications"])} applications and {count(meta["test_frauds"])} frauds. Cut-offs were
fixed on month {CALIBRATION_MONTH} before the test months were scored. Ranges are 95% bootstrap intervals.

## Headline

{chr(10).join(headline_lines(art))}

![Value lines]({fig_dir}/value_lines.png)

## Policies compared on the test months

| Policy | Profit per 1,000 | Genuine onboarded | Fraud stopped | Approve / verify / review / decline |
|---|---|---|---|---|
{chr(10).join(rows)}

On the test months the cut-offs fixed on month {CALIBRATION_MONTH} asked for {meta["test_reviews_wanted"]:,.0f} reviews
against capacity for {meta["test_review_capacity"]:,.0f}; {comp.loc["threshold", "reviews"]:,.0f} were done.

![Trade-off]({fig_dir}/tradeoff.png)

## Model (test months, after calibration)

| Model | Fraud caught at 5% FPR | ROC AUC | Average precision | Brier |
|---|---|---|---|---|
| Logistic regression (baseline) | {pct(base["tpr_at_5pct_fpr"])} | {base["roc_auc"]:.3f} | {base["average_precision"]:.3f} | {base["brier"]:.4f} |
| Gradient boosting ({m["backend"]}, {m["rounds"]} trees) | {pct(gbm["tpr_at_5pct_fpr"])} | {gbm["roc_auc"]:.3f} | {gbm["average_precision"]:.3f} | {gbm["brier"]:.4f} |

![Calibration]({fig_dir}/calibration.png)

## Review team

| Analysts | Capacity / day | Reviews / day | Value of this analyst / day | Worth hiring? |
|---|---|---|---|---|
{chr(10).join(team_rows)}

![Analyst value]({fig_dir}/analyst_value.png)

## Data sources

| Source | Supplier | Value per application | 95% range |
|---|---|---|---|
{chr(10).join(vend)}

![Vendor value]({fig_dir}/vendor_value.png)

## Fairness: genuine applicants given any extra check or declined

| Model | Policy | Under 50 | 50+ | Ratio |
|---|---|---|---|---|
{chr(10).join(fair_rows)}

⚑ = outside the 0.8×–1.25× band set in `config/assumptions.yaml`; escalate to Compliance.

## Monitoring (policy fixed on month {CALIBRATION_MONTH}, replayed month by month)

| Month | Approved | Fraud rate | Fraud stopped | Score PSI | Actual vs expected fraud | Reviews wanted / capacity |
|---|---|---|---|---|---|---|
{chr(10).join(mon_rows)}

Most-shifted features in the test months (PSI vs training months):

| Feature | Month | PSI |
|---|---|---|
{chr(10).join(drift_rows)}

## Sensitivity

![Sensitivity]({fig_dir}/sensitivity.png)

## Speed

Full pipeline: {timings["total"]:.0f}s on this machine. Stages: {", ".join(f"{k} {v:.1f}s" for k, v in timings.items() if k != "total")}.
"""


def model_card(art: Artifacts) -> str:
    m = art.metrics
    gbm, raw, base = m["gbm"]["calibrated"], m["gbm"]["raw"], m["baseline"]["calibrated"]
    fair = art.fairness
    row = fair[
        (fair["variant"] == "full")
        & (fair["policy"] == "paper_cutoff")
        & (fair["metric"] == "genuine_checked_rate")
    ]
    noage = fair[
        (fair["variant"] == "no_age")
        & (fair["policy"] == "paper_cutoff")
        & (fair["metric"] == "genuine_checked_rate")
    ]
    note = (
        "⚠️ Generated from synthetic data — rerun on the real data before reading anything into it.\n\n"
        if art.synthetic
        else ""
    )
    return f"""# Model card: Threshold fraud score

{note}## What it is
A gradient-boosted tree model ({m["backend"]}, {m["rounds"]} trees, chosen by early stopping on month 4)
estimating the probability that a bank-account application is fraudulent, followed by Platt
calibration fitted on month {CALIBRATION_MONTH} (a = {m["gbm"]["calibrator"]["a"]:.3f}, b = {m["gbm"]["calibrator"]["b"]:.3f}).
It feeds a decision policy; it is not a decision on its own.

## Data
Feedzai Bank Account Fraud (NeurIPS 2022), `Base.csv`: 1M synthetic-but-realistic applications
over 8 months, 30 features. See `DATA.md`. Learn on months 0–4, calibrate on month 5, test on
months 6–7. Missing values (−1 sentinels) are converted to NULL in SQL and left for the trees to route.

## Performance on the test months
| | Fraud caught at 5% FPR | ROC AUC | Average precision | Brier | Mean predicted vs observed |
|---|---|---|---|---|---|
| Baseline (logistic) | {pct(base["tpr_at_5pct_fpr"])} | {base["roc_auc"]:.3f} | {base["average_precision"]:.3f} | {base["brier"]:.4f} | {pct(base["mean_predicted"], 2)} vs {pct(base["observed_rate"], 2)} |
| GBM, raw | {pct(raw["tpr_at_5pct_fpr"])} | {raw["roc_auc"]:.3f} | {raw["average_precision"]:.3f} | {raw["brier"]:.4f} | {pct(raw["mean_predicted"], 2)} vs {pct(raw["observed_rate"], 2)} |
| GBM, calibrated | {pct(gbm["tpr_at_5pct_fpr"])} | {gbm["roc_auc"]:.3f} | {gbm["average_precision"]:.3f} | {gbm["brier"]:.4f} | {pct(gbm["mean_predicted"], 2)} vs {pct(gbm["observed_rate"], 2)} |

"Fraud caught at 5% FPR" is the dataset paper's headline metric. Calibration does not change
the ranking (so ROC AUC is unchanged); it makes the probabilities usable as £ inputs.

## Fairness
At the paper's 5%-FPR cut-off, genuine applicants aged 50+ are flagged
{pct(row["over_50"].iloc[0]) if len(row) else "–"} of the time vs {pct(row["under_50"].iloc[0]) if len(row) else "–"} for under-50s
(ratio {row["ratio"].iloc[0]:.2f}×). Retraining without age gives {noage["ratio"].iloc[0]:.2f}×: other features carry
much of the same information. Age-specific cut-offs are not used (Equality Act 2010); the gap is
reported on every policy for Compliance to judge.

## Limits
- The data is partly synthetic and US-flavoured; the method transfers, the numbers do not.
- The month-to-month fraud rate drifts; calibration is fitted on one month and needs refitting.
- Reviewer and verification outcomes (r, f, a_g, a_f) are assumptions, not measurements.
- Confirmed fraud labels arrive late in practice; PSI is the early warning, not outcome metrics.
"""


def assumptions_markdown() -> str:
    rows = [
        f"| {r['symbol']} | {r['assumption']} | {r['value']:g} | "
        f"{'' if r['low'] is None else f'{r["low"]:g} – {r["high"]:g}'} | {r['unit']} | {r['source']} | {r['checked'] or '–'} |"
        for r in assumption_records()
    ]
    return (
        "# Assumptions and sources\n\n"
        "Generated from `config/assumptions.yaml` by `make report`. Edit the YAML, not this file.\n\n"
        "| Symbol | Assumption | Default | Range tested | Unit | Source | Checked |\n"
        "|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
    )


def update_readme(art: Artifacts) -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    block = "\n".join(headline_lines(art))
    new = re.sub(
        r"(<!-- results:start -->\n).*?(\n<!-- results:end -->)",
        lambda mt: mt.group(1) + block + mt.group(2),
        text,
        flags=re.S,
    )
    readme.write_text(new)


def build(artifacts: Path = ARTIFACTS) -> Path:
    _style()
    art = Artifacts(artifacts)
    if art.synthetic:
        out, fig_rel = artifacts / "report", "figures"
    else:
        out, fig_rel = REPORTS, "figures"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for make in (
        fig_value_lines,
        fig_tradeoff,
        fig_team,
        fig_vendors,
        fig_tornado,
        fig_reliability,
    ):
        make(art, figs)
    (out / "results.md").write_text(results_markdown(art, fig_rel))
    (out / "model_card.md").write_text(model_card(art))
    (DOCS / "assumptions.md").write_text(assumptions_markdown())
    if not art.synthetic:
        update_readme(art)
    print(f"✓ report written to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    build(parser.parse_args().artifacts)


if __name__ == "__main__":
    main()
