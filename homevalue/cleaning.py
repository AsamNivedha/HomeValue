"""Validation-aware cleaning and feature engineering.

The module turns the (column-mapped) uploaded table into three views:

* ``analytic``  - every original record, enriched with parsed/derived columns and
                  quality flags. Nothing is dropped here, so Explore can show
                  "N of 13,320 properties".
* ``model_df``  - the subset of records that are safe to learn from (no duplicates,
                  usable price/area/BHK, plausible configuration).
* ``issues``    - a plain-language data-quality report.

The uploaded ``raw`` frame is never mutated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as cfg

# ------------------------------------------------------------------ exclusion reasons
R_DUP = "Exact duplicate"
R_PRICE = "Missing or non-positive price"
R_AREA = "Area could not be interpreted"
R_BHK = "Bedroom count unavailable"
R_PLAUS = "Implausible property configuration"
R_PPS = "Extreme price per sq ft"
REASON_ORDER = [R_DUP, R_PRICE, R_AREA, R_BHK, R_PLAUS, R_PPS]

_RANGE = re.compile(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$")
_UNIT = re.compile(r"^\d+(?:\.\d+)?\s*[A-Za-z][A-Za-z.\s]*$")
_BHK = re.compile(r"^\s*(\d+)")


# ------------------------------------------------------------------ parsing helpers
def clean_text(value):
    """Collapse whitespace; empty strings become missing."""
    if pd.isna(value):
        return np.nan
    text = " ".join(str(value).split())
    return text if text else np.nan


def parse_area(value) -> tuple[float, str]:
    """Return ``(square_feet, status)`` for one total_sqft value.

    status is one of: numeric, range, unit, malformed, missing. Values quoted in
    other units (sq. metres, perches, ...) are *not* converted silently; they are
    reported as ``unit`` and excluded from the model-ready data.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return (float(value), "numeric") if np.isfinite(value) else (np.nan, "malformed")
    text = str(value).strip().replace(",", "")
    if not text:
        return np.nan, "missing"
    try:
        number = float(text)
        return (number, "numeric") if np.isfinite(number) else (np.nan, "malformed")
    except ValueError:
        pass
    match = _RANGE.match(text)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return (low + high) / 2.0, "range"
    if _UNIT.match(text):
        return np.nan, "unit"
    return np.nan, "malformed"


def parse_bhk(value) -> float:
    """'2 BHK' -> 2, '4 Bedroom' -> 4, '1 RK' -> 1; anything else -> NaN."""
    if pd.isna(value):
        return np.nan
    match = _BHK.match(str(value))
    return float(match.group(1)) if match else np.nan


def availability_group(value):
    """Collapse availability into 'Ready to Move' or 'Available Later'."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "ready" in text or "immediate" in text:
        return "Ready to Move"
    return "Available Later"


# ------------------------------------------------------------------ result containers
@dataclass
class QualityIssue:
    key: str
    title: str
    count: int
    severity: str  # "attention" | "minor"
    impact: str
    action: str


@dataclass
class PreparedData:
    analytic: pd.DataFrame
    model_df: pd.DataFrame
    funnel: list[tuple[str, int]]
    issues: list[QualityIssue]
    checks_passed: list[str]
    summary: dict
    column_map: dict[str, str] = field(default_factory=dict)

    @property
    def quality_status(self) -> str:
        if any(i.severity == "attention" for i in self.issues):
            return "Needs attention"
        if self.issues:
            return "Good"
        return "Excellent"


# ------------------------------------------------------------------ main entry point
def prepare(working: pd.DataFrame, column_map: dict[str, str] | None = None) -> PreparedData:
    """Build the analytic table, the model-ready table and the quality report."""
    w = working
    n = len(w)
    a = pd.DataFrame(index=w.index)

    for col in ("area_type", "availability", "location", "size", "society"):
        a[col] = w[col].map(clean_text) if col in w.columns else np.nan

    parsed_area = w["total_sqft"].map(parse_area)
    a["total_sqft"] = parsed_area.map(lambda t: t[0]).astype(float)
    a["bath"] = pd.to_numeric(w["bath"], errors="coerce")
    a["balcony"] = pd.to_numeric(w["balcony"], errors="coerce")
    price_numeric = pd.to_numeric(w[cfg.TARGET], errors="coerce")
    a[cfg.TARGET] = price_numeric

    a["bhk"] = a["size"].map(parse_bhk).astype(float)
    a["availability_group"] = a["availability"].map(availability_group)
    valid_pps = (a["total_sqft"] > 0) & (a[cfg.TARGET] > 0)
    a["price_per_sqft"] = np.where(valid_pps, a[cfg.TARGET] * cfg.LAKH / a["total_sqft"], np.nan)
    a["total_sqft_raw"] = w["total_sqft"]
    a["area_status"] = parsed_area.map(lambda t: t[1])
    a["is_duplicate"] = w.duplicated(keep="first")

    for col in w.columns:  # keep any extra columns the user uploaded
        if col not in a.columns:
            a[col] = w[col]

    # ---- exclusion cascade: each record gets the first reason that applies
    reason = pd.Series("", index=a.index, dtype=object)

    def mark(mask: pd.Series, label: str) -> None:
        reason.loc[(reason == "") & mask.fillna(False)] = label

    sqft_per_bhk = a["total_sqft"] / a["bhk"].where(a["bhk"] > 0)
    implausible = (sqft_per_bhk < cfg.MIN_SQFT_PER_BHK) | (a["bath"] > a["bhk"] + cfg.MAX_EXTRA_BATHROOMS)

    mark(a["is_duplicate"], R_DUP)
    mark(a[cfg.TARGET].isna() | (a[cfg.TARGET] <= 0), R_PRICE)
    mark(a["total_sqft"].isna() | (a["total_sqft"] <= 0), R_AREA)
    mark(a["bhk"].isna() | (a["bhk"] <= 0), R_BHK)
    mark(implausible, R_PLAUS)

    pps_low = pps_high = np.nan
    pps_pool = a.loc[reason == "", "price_per_sqft"].dropna()
    if len(pps_pool) >= 50:
        pps_low, pps_high = pps_pool.quantile(cfg.PPS_TRIM[0]), pps_pool.quantile(cfg.PPS_TRIM[1])
        mark((a["price_per_sqft"] < pps_low) | (a["price_per_sqft"] > pps_high), R_PPS)

    a["exclusion_reason"] = reason
    model_df = a[reason == ""].copy()

    funnel = [("Original records", n)]
    funnel += [(label, int((reason == label).sum())) for label in REASON_ORDER]
    funnel.append(("Model-ready records", len(model_df)))

    # ---- summary figures used by validation, overview and insights
    price_ok = a[cfg.TARGET].where(a[cfg.TARGET] > 0)
    area_ok = a["total_sqft"].where(a["total_sqft"] > 0)
    summary = {
        "n_raw": n,
        "n_columns": int(w.shape[1]),
        "n_model": len(model_df),
        "price_mean": float(price_ok.mean()) if price_ok.notna().any() else np.nan,
        "price_median": float(price_ok.median()) if price_ok.notna().any() else np.nan,
        "area_mean": float(area_ok.mean()) if area_ok.notna().any() else np.nan,
        "price_numeric_share": float(price_numeric.notna().mean()) if n else 0.0,
        "area_parse_share": float(a["area_status"].isin(["numeric", "range"]).mean()) if n else 0.0,
        "bhk_share": float(a["bhk"].notna().mean()) if n else 0.0,
        "location_share": float(a["location"].notna().mean()) if n else 0.0,
        "pps_low": pps_low,
        "pps_high": pps_high,
    }

    issues, passed = _quality_report(w, a, model_df, implausible, pps_low, pps_high)
    return PreparedData(
        analytic=a,
        model_df=model_df,
        funnel=funnel,
        issues=issues,
        checks_passed=passed,
        summary=summary,
        column_map=column_map or {},
    )


# ------------------------------------------------------------------ quality report
_MISSING_NOTES: dict[str, tuple[str, str, bool]] = {
    # column: (impact, action, used_by_model)
    "bath": (
        "Bathroom count is a model input, and a linear model needs a value for every feature.",
        "Missing counts are filled with the training-data median inside the model pipeline; the properties are kept.",
        True,
    ),
    "balcony": (
        "Balcony count is a model input, and a linear model needs a value for every feature.",
        "Missing counts are filled with the training-data median inside the model pipeline; the properties are kept.",
        True,
    ),
    "size": (
        "The bedroom count (BHK) is read from this column, so the property cannot be described to the model without it.",
        "These records stay in the explorer but are left out of the model-ready dataset.",
        True,
    ),
    "location": (
        "Location is an important model input.",
        "The record is kept and treated like a rarely-seen location (grouped into Other).",
        True,
    ),
    "society": (
        "Society names are sparse and have very many distinct values, which makes them a poor model input.",
        "Society is kept for exploration only and is not used by the model.",
        False,
    ),
    "availability": (
        "Availability is a model input.",
        "Missing values are filled with the most common value inside the model pipeline.",
        True,
    ),
    "area_type": (
        "Area type is a model input.",
        "Missing values are filled with the most common value inside the model pipeline.",
        True,
    ),
}

_MISSING_TITLES = {
    "bath": "Bathroom information is missing for {n} listings",
    "balcony": "Balcony information is missing for {n} listings",
    "size": "Bedroom information (size) is missing for {n} listings",
    "location": "Location is missing for {n} listing(s)",
    "society": "Society is missing for {n} listings ({pct})",
    "availability": "Availability is missing for {n} listings",
    "area_type": "Area type is missing for {n} listings",
}


def _share_severity(count: int, total: int, threshold: float = 0.02) -> str:
    return "attention" if total and count / total >= threshold else "minor"


def _quality_report(
    w: pd.DataFrame,
    a: pd.DataFrame,
    model_df: pd.DataFrame,
    implausible: pd.Series,
    pps_low: float,
    pps_high: float,
) -> tuple[list[QualityIssue], list[str]]:
    n = len(a)
    issues: list[QualityIssue] = []
    passed: list[str] = []

    def add(key, title, count, severity, impact, action):
        issues.append(QualityIssue(key, title, int(count), severity, impact, action))

    # duplicates
    dups = int(a["is_duplicate"].sum())
    if dups:
        add(
            "duplicates",
            f"{dups:,} duplicate records detected",
            dups,
            _share_severity(dups, n),
            "Duplicate records can give repeated observations disproportionate influence during training.",
            "Exact duplicates are removed from the model-ready dataset; the first occurrence is kept.",
        )
    else:
        passed.append("Duplicate records")

    # missing target
    missing_price = int(a[cfg.TARGET].isna().sum())
    if missing_price:
        add(
            "target_missing",
            f"Price is missing or non-numeric for {missing_price:,} listings",
            missing_price,
            "attention",
            "A model cannot learn from a record that has no price.",
            "These records are excluded from the model-ready dataset.",
        )
    else:
        passed.append("Target (price) availability")

    nonpositive_price = int((a[cfg.TARGET] <= 0).sum())
    if nonpositive_price:
        add(
            "price_nonpositive",
            f"{nonpositive_price:,} listings have a price of zero or less",
            nonpositive_price,
            "attention",
            "A zero or negative price is not a meaningful listed price.",
            "These records are excluded from the model-ready dataset.",
        )
    else:
        passed.append("Non-positive prices")

    # missing values in each column
    any_missing = False
    for col in ["bath", "balcony", "size", "location", "society", "availability", "area_type"]:
        if col not in a.columns:
            continue
        count = int(a[col].isna().sum())
        if count == 0:
            continue
        any_missing = True
        impact, action, used = _MISSING_NOTES[col]
        severity = _share_severity(count, n, 0.05) if used else "minor"
        title = _MISSING_TITLES[col].format(n=f"{count:,}", pct=f"{count / n:.0%}")
        add(f"missing_{col}", title, count, severity, impact, action)
    if not any_missing:
        passed.append("Missing values")

    # area values that needed attention
    status = a["area_status"]
    ranges = int((status == "range").sum())
    if ranges:
        add(
            "area_ranges",
            f"{ranges:,} listings give the area as a range",
            ranges,
            "minor",
            "A range such as '2100 - 2850' is not a single number the model can use.",
            "The midpoint of the range is used (for example 2475 sq ft).",
        )
    units = int((status == "unit").sum())
    if units:
        add(
            "area_units",
            f"{units:,} listings give the area in other units",
            units,
            _share_severity(units, n, 0.01),
            "Areas quoted in sq. metres, perches, yards and similar units are not comparable with square feet.",
            "They are not converted silently; these records are excluded from the model-ready dataset.",
        )
    malformed = int(status.isin(["malformed", "missing"]).sum())
    if malformed:
        add(
            "area_malformed",
            f"{malformed:,} listings have an area that cannot be interpreted",
            malformed,
            _share_severity(malformed, n, 0.01),
            "Some area values cannot be read as a number, so no reliable area is available.",
            "These records are excluded from the model-ready dataset.",
        )
    if not (ranges or units or malformed):
        passed.append("Malformed area values")

    nonpositive_area = int((a["total_sqft"] <= 0).sum())
    if nonpositive_area:
        add(
            "area_nonpositive",
            f"{nonpositive_area:,} listings have an area of zero or less",
            nonpositive_area,
            "attention",
            "An area of zero or less cannot describe a property.",
            "These records are excluded from the model-ready dataset.",
        )
    else:
        passed.append("Non-positive areas")

    # numeric conversion for bath / balcony
    for col in ("bath", "balcony"):
        raw_present = w[col].notna()
        failed = int((raw_present & a[col].isna()).sum())
        if failed:
            add(
                f"{col}_nonnumeric",
                f"{failed:,} {col} values are not numeric",
                failed,
                "minor",
                "Non-numeric entries cannot be used as counts.",
                "They are treated as missing and handled like other missing values.",
            )

    # unusual property characteristics (over non-duplicate records)
    nd = ~a["is_duplicate"]
    sqft_per_bhk = a["total_sqft"] / a["bhk"].where(a["bhk"] > 0)
    few_sqft = (sqft_per_bhk < cfg.MIN_SQFT_PER_BHK) & nd
    many_bath = (a["bath"] > a["bhk"] + cfg.MAX_EXTRA_BATHROOMS) & nd
    extreme_pps = pd.Series(False, index=a.index)
    if not np.isnan(pps_low):
        extreme_pps = ((a["price_per_sqft"] < pps_low) | (a["price_per_sqft"] > pps_high)) & nd
    unusual = (few_sqft | many_bath | extreme_pps).sum()
    if unusual:
        parts = []
        if few_sqft.sum():
            parts.append(f"{int(few_sqft.sum()):,} with less than {cfg.MIN_SQFT_PER_BHK} sq ft per bedroom")
        if many_bath.sum():
            parts.append(
                f"{int(many_bath.sum()):,} with more than {cfg.MAX_EXTRA_BATHROOMS} bathrooms beyond the bedroom count"
            )
        if extreme_pps.sum():
            parts.append(f"{int(extreme_pps.sum()):,} with an extreme price per sq ft")
        add(
            "unusual",
            f"{int(unusual):,} listings have unusual property characteristics",
            unusual,
            _share_severity(int(unusual), n, 0.02),
            "Unusual records (" + "; ".join(parts) + ") are often data-entry errors and can distort a linear model.",
            "Records outside plausible ranges are excluded from the model-ready dataset but remain visible in the explorer.",
        )
    else:
        passed.append("Unusual property characteristics")

    # sparse location categories (in the model-ready data)
    if len(model_df):
        counts = model_df["location"].fillna("Unknown").value_counts()
        rare = counts[counts < cfg.MIN_LOCATION_COUNT]
        if len(rare):
            add(
                "sparse_locations",
                f"{len(rare):,} locations have fewer than {cfg.MIN_LOCATION_COUNT} listings",
                int(rare.sum()),
                "minor",
                "Very rare categories create unstable model features that do not generalise.",
                "They are grouped into Other, consistently during training and prediction.",
            )
        else:
            passed.append("Sparse location categories")

    # empty / constant columns
    empty_cols = [c for c in w.columns if w[c].isna().all()]
    constant_cols = [c for c in w.columns if c not in empty_cols and w[c].nunique(dropna=True) <= 1]
    if empty_cols:
        add(
            "empty_columns",
            f"{len(empty_cols)} column(s) are completely empty: {', '.join(map(str, empty_cols))}",
            len(empty_cols),
            "minor",
            "An empty column carries no information.",
            "Empty columns contribute nothing to the model.",
        )
    if constant_cols:
        add(
            "constant_columns",
            f"{len(constant_cols)} column(s) hold a single value: {', '.join(map(str, constant_cols))}",
            len(constant_cols),
            "minor",
            "A column that never changes cannot explain differences between properties.",
            "Constant columns have no influence on the estimate.",
        )
    if not (empty_cols or constant_cols):
        passed.append("Empty or constant columns")

    order = {"attention": 0, "minor": 1}
    issues.sort(key=lambda i: (order[i.severity], -i.count))
    return issues, passed
