"""Small display-formatting helpers shared by the UI and the insight text."""
from __future__ import annotations

import math

import pandas as pd


def _bad(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value)


def fmt_int(value) -> str:
    return "—" if _bad(value) else f"{int(round(float(value))):,}"


def fmt_num(value, digits: int = 1) -> str:
    return "—" if _bad(value) else f"{float(value):,.{digits}f}"


def fmt_lakh(value, digits: int = 1) -> str:
    return "—" if _bad(value) else f"₹{float(value):,.{digits}f} lakh"


def fmt_crore(value_lakh, digits: int = 2) -> str:
    return "—" if _bad(value_lakh) else f"₹{float(value_lakh) / 100:,.{digits}f} crore"


def fmt_sqft(value) -> str:
    return "—" if _bad(value) else f"{float(value):,.0f} sq ft"


def fmt_rupees(value) -> str:
    return "—" if _bad(value) else f"₹{float(value):,.0f}"


def fmt_pct(value, digits: int = 0) -> str:
    return "—" if _bad(value) else f"{float(value) * 100:.{digits}f}%"
