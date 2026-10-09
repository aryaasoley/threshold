"""Colours and number formats shared by the app and the static report.

Calm and neutral: one accent colour for Threshold, grey for every alternative. Three
actions take the three categorical slots that stay distinct for every colour-vision type;
decline is neutral grey. Actions are always labelled directly, so colour never carries
meaning alone.
"""

from __future__ import annotations

import math

ACCENT = "#2a78d6"
MUTED = "#898781"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ACTION_COLORS = {
    "approve": "#2a78d6",  # blue
    "verify": "#1baf7a",  # aqua
    "review": "#eb6834",  # orange
    "decline": "#52514e",  # neutral: declining is worth £0 either way
}

STATUS_COLORS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "grey": "#898781"}
STATUS_LABELS = {"green": "● Stable", "amber": "▲ Watch", "red": "■ Act", "grey": "– n/a"}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def money(x: float, decimals: int | None = None) -> str:
    """£ the way Finance writes it: £38, £1.2k, £3.4m, −£250."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    if x == 0:
        return "£0"
    sign = "−" if x < 0 else ""
    v = abs(x)
    if v >= 1e6:
        return f"{sign}£{v / 1e6:.{1 if decimals is None else decimals}f}m"
    if v >= 1e4:
        return f"{sign}£{v / 1e3:.{0 if decimals is None else decimals}f}k"
    if v >= 1e3:
        return f"{sign}£{v / 1e3:.{1 if decimals is None else decimals}f}k"
    if v >= 10:
        return f"{sign}£{v:.{0 if decimals is None else decimals}f}"
    return f"{sign}£{v:.{2 if decimals is None else decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x * 100:.{decimals}f}%"


def signed_pct(x: float, decimals: int = 1) -> str:
    return ("+" if x >= 0 else "−") + pct(abs(x), decimals)


def count(x: float) -> str:
    return f"{x:,.0f}"
