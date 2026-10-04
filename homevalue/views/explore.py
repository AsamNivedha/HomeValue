"""Explore: dataset explorer, data quality center and market exploration.

Everything on this page is local to the Explore view. Filters change what is shown
here only; they never touch the model-ready dataset or trigger retraining.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .. import charts, ui
from .. import config as cfg
from ..cleaning import R_AREA, R_BHK, R_DUP, R_PLAUS, R_PPS, R_PRICE
from ..formatting import fmt_int
from ..state import LoadedDataset

VIEWS = ["Dataset explorer", "Data quality", "Market exploration"]

DEFAULT_COLUMNS = [
    "location", "size", "total_sqft", "bath", "balcony", "price", "area_type", "availability", "society",
]
LABELS = {
    "price": "Price (₹ lakh)",
    "total_sqft": "Total area (sq ft)",
    "price_per_sqft": "Price per sq ft (₹)",
    "total_sqft_raw": "Area as uploaded",
    "exclusion_reason": "Excluded because",
    "availability_group": "Availability group",
    "area_status": "Area format",
    "is_duplicate": "Duplicate",
    "bhk": "BHK",
}
REASON_TEXT = {
    R_DUP: "The same record appears more than once; the first copy is kept.",
    R_PRICE: "No usable price to learn from.",
    R_AREA: "The area is missing, in another unit, or cannot be read as a number.",
    R_BHK: "The bedroom count cannot be read from the size description.",
    R_PLAUS: f"Less than {cfg.MIN_SQFT_PER_BHK} sq ft per bedroom, or more than {cfg.MAX_EXTRA_BATHROOMS} bathrooms beyond the bedroom count.",
    R_PPS: "Price per sq ft is outside the central 98% of the dataset, which usually points to data-entry errors.",
}
TEXT_SEARCH_COLUMNS = ["location", "society", "area_type", "size", "availability"]


# ------------------------------------------------------------------ explorer state
def _bounds(a: pd.DataFrame) -> dict:
    def span(col, integer=False):
        s = a[col].dropna()
        if s.empty:
            lo, hi = 0, 1
        else:
            lo, hi = s.min(), s.max()
        if integer:
            lo, hi = int(np.floor(lo)), int(np.ceil(hi))
            return lo, max(hi, lo + 1)
        lo, hi = float(np.floor(lo)), float(np.ceil(hi))
        return lo, max(hi, lo + 1.0)

    return {"bhk": span("bhk", True), "price": span("price"), "area": span("total_sqft")}


def _filter_defaults(a: pd.DataFrame) -> dict:
    b = _bounds(a)
    return {
        "ex_search": "",
        "ex_area_types": [],
        "ex_avail": [],
        "ex_locations": [],
        "ex_status": "All records",
        "ex_bhk": b["bhk"],
        "ex_price": b["price"],
        "ex_area": b["area"],
        "ex_cols": [c for c in DEFAULT_COLUMNS if c in a.columns],
        "ex_sort_col": "(original order)",
        "ex_sort_asc": True,
    }


def _init_filters(a: pd.DataFrame) -> None:
    for key, value in _filter_defaults(a).items():
        st.session_state.setdefault(key, value)


def _reset_filters(a: pd.DataFrame) -> None:
    for key, value in _filter_defaults(a).items():
        st.session_state[key] = value


def apply_filters(a: pd.DataFrame, state) -> pd.DataFrame:
    """Return the rows of ``a`` that match the current explorer filters (a copy-free view)."""
    mask = pd.Series(True, index=a.index)
    query = str(state["ex_search"]).strip()
    if query:
        hit = pd.Series(False, index=a.index)
        for col in TEXT_SEARCH_COLUMNS:
            if col in a.columns:
                hit |= a[col].fillna("").astype(str).str.contains(query, case=False, regex=False)
        mask &= hit
    if state["ex_area_types"]:
        mask &= a["area_type"].isin(state["ex_area_types"])
    if state["ex_avail"]:
        mask &= a["availability_group"].isin(state["ex_avail"])
    if state["ex_locations"]:
        mask &= a["location"].isin(state["ex_locations"])
    status = state["ex_status"]
    if status == "Model-ready only":
        mask &= a["exclusion_reason"] == ""
    elif status == "Excluded from the model only":
        mask &= a["exclusion_reason"] != ""
    b = _bounds(a)
    for key, col, bound in (("ex_bhk", "bhk", b["bhk"]), ("ex_price", "price", b["price"]), ("ex_area", "total_sqft", b["area"])):
        lo, hi = state[key]
        if (lo, hi) != tuple(bound):
            mask &= a[col].between(lo, hi)
    return a[mask]


# ------------------------------------------------------------------ dataset explorer
def _column_config(columns: list[str]) -> dict:
    config = {}
    for col in columns:
        label = LABELS.get(col, col)
        if col == "price":
            config[col] = st.column_config.NumberColumn(label, format="%.1f")
        elif col in ("total_sqft", "price_per_sqft"):
            config[col] = st.column_config.NumberColumn(label, format="%.0f")
        elif col in ("bath", "balcony", "bhk"):
            config[col] = st.column_config.NumberColumn(label, format="%d")
        elif col == "is_duplicate":
            config[col] = st.column_config.CheckboxColumn(label)
        else:
            config[col] = st.column_config.TextColumn(label)
    return config


def _explorer(ds: LoadedDataset) -> pd.DataFrame:
    a = ds.prep.analytic
    ui.section("Dataset explorer", "Inspect the property records. Filters here are local to this view.")
    with st.expander("Search and filters", expanded=True):
        st.text_input("Search locations, societies, area types and sizes", key="ex_search", placeholder="e.g. Whitefield")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.multiselect("Location", sorted(a["location"].dropna().unique()), key="ex_locations", placeholder="All locations")
        with c2:
            st.multiselect("Area type", sorted(a["area_type"].dropna().unique()), key="ex_area_types", placeholder="All area types")
        with c3:
            st.multiselect("Availability", ["Ready to Move", "Available Later"], key="ex_avail", placeholder="Any availability")
        b = _bounds(a)
        c4, c5, c6 = st.columns(3)
        with c4:
            st.slider("BHK", b["bhk"][0], b["bhk"][1], key="ex_bhk")
        with c5:
            st.slider("Price (₹ lakh)", b["price"][0], b["price"][1], step=1.0, key="ex_price")
        with c6:
            st.slider("Total area (sq ft)", b["area"][0], b["area"][1], step=10.0, key="ex_area")
        c7, c8 = st.columns([2, 1])
        with c7:
            st.radio("Records", ["All records", "Model-ready only", "Excluded from the model only"], key="ex_status", horizontal=True)
        with c8:
            ui.button("Reset filters", key="btn_ex_reset", on_click=_reset_filters, args=(a,))

    filtered = apply_filters(a, st.session_state)
    st.markdown(f"**Showing {fmt_int(len(filtered))} of {fmt_int(len(a))} properties**")

    all_cols = [c for c in a.columns]
    with st.expander("Columns and sorting"):
        st.multiselect("Visible columns", all_cols, key="ex_cols", format_func=lambda c: LABELS.get(c, c))
        shown_cols = [c for c in st.session_state["ex_cols"] if c in all_cols] or DEFAULT_COLUMNS[:3]
        sort_options = ["(original order)"] + shown_cols
        if st.session_state["ex_sort_col"] not in sort_options:
            st.session_state["ex_sort_col"] = "(original order)"
        s1, s2 = st.columns([2, 1])
        with s1:
            st.selectbox("Sort by", sort_options, key="ex_sort_col", format_func=lambda c: LABELS.get(c, c))
        with s2:
            st.toggle("Ascending", key="ex_sort_asc")

    shown_cols = [c for c in st.session_state["ex_cols"] if c in all_cols] or DEFAULT_COLUMNS[:3]
    view = filtered[shown_cols].copy()
    if st.session_state["ex_sort_col"] in shown_cols:
        view = view.sort_values(st.session_state["ex_sort_col"], ascending=st.session_state["ex_sort_asc"], na_position="last")
    if "exclusion_reason" in view.columns:
        view["exclusion_reason"] = view["exclusion_reason"].replace("", "Included in the model")
    if view.empty:
        ui.callout("info", "No properties match these filters", "Widen or reset the filters to see records.")
    else:
        ui.table(view, hide_index=True, height=440, column_config=_column_config(shown_cols))
        ui.download(
            "Download filtered data (CSV)",
            view.to_csv(index=False).encode("utf-8"),
            "homevalue_filtered_data.csv",
            "text/csv",
            key="dl_filtered",
        )

    _column_inspector(ds)
    return filtered


def _column_inspector(ds: LoadedDataset) -> None:
    a = ds.prep.analytic
    ui.section("Column inspection", "Select a column to see what it contains.")
    options = [c for c in a.columns if c not in ("exclusion_reason",)]
    if st.session_state.get("ex_inspect") not in options:
        st.session_state["ex_inspect"] = "location" if "location" in options else options[0]
    col = st.selectbox("Column", options, key="ex_inspect", format_func=lambda c: LABELS.get(c, c))
    series = a[col]
    is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
    missing = int(series.isna().sum())
    ui.key_values(
        [
            ("Data type", "Number" if is_numeric else "Yes / no" if pd.api.types.is_bool_dtype(series) else "Text"),
            ("Missing values", f"{fmt_int(missing)} ({missing / len(a):.1%})"),
            ("Unique values", fmt_int(series.nunique(dropna=True))),
        ]
    )
    non_null = series.dropna()
    if non_null.empty:
        ui.callout("attention", "This column is empty", "It carries no information for the analysis.")
        return
    examples = pd.Series(non_null.unique()).head(6).astype(str).tolist()
    ui.note("Example values: " + ", ".join(examples))
    if col == "total_sqft":
        status = a["area_status"].value_counts()
        ui.note(
            f"Area as uploaded: {fmt_int(status.get('numeric', 0))} plain numbers, {fmt_int(status.get('range', 0))} ranges "
            f"(midpoint used), {fmt_int(status.get('unit', 0))} in other units and "
            f"{fmt_int(status.get('malformed', 0) + status.get('missing', 0))} unreadable (these last two have no area)."
        )
    if is_numeric:
        desc = non_null.describe(percentiles=[0.25, 0.5, 0.75])
        stats = pd.DataFrame(
            {
                "Statistic": ["Minimum", "25th percentile", "Median", "Mean", "75th percentile", "Maximum"],
                "Value": [desc["min"], desc["25%"], desc["50%"], desc["mean"], desc["75%"], desc["max"]],
            }
        )
        ui.table(stats, hide_index=True, column_config={"Value": st.column_config.NumberColumn(format="%.2f")})
    else:
        counts = non_null.astype(str).value_counts()
        top = counts.head(10)
        out = pd.DataFrame({"Value": top.index, "Records": top.values, "Share (%)": top.values / len(non_null) * 100})
        ui.table(out, hide_index=True, column_config={"Share (%)": st.column_config.NumberColumn(format="%.1f")})
        if len(counts) > 10:
            ui.note(f"Showing the 10 most frequent of {fmt_int(len(counts))} values.")


# ------------------------------------------------------------------ data quality center
def _quality(ds: LoadedDataset) -> None:
    prep = ds.prep
    ui.section("Data quality center", "What was detected, why it matters and what HomeValue does about it.")
    status = prep.quality_status
    kind = {"Needs attention": "attention", "Good": "info", "Excellent": "success"}[status]
    if prep.issues:
        n_att = sum(1 for i in prep.issues if i.severity == "attention")
        ui.callout(
            kind,
            {"Needs attention": "Data quality needs attention", "Good": "Data quality is good, with minor notes"}[status],
            f"{len(prep.issues)} finding(s), {n_att} of them marked as needing attention. All are handled automatically.",
        )
        for issue in prep.issues:
            ui.issue_card(issue)
    else:
        ui.callout("success", "No data-quality issues detected", "Every check passed.")
    if prep.checks_passed:
        ui.note("Checks that found nothing: " + ", ".join(prep.checks_passed) + ".")

    ui.section("Original data vs model-ready data", "How the uploaded records become the records the model learns from.")
    ui.flow(prep.funnel)
    rows = [
        {"Reason": label, "Records": count, "Why": REASON_TEXT[label]}
        for label, count in prep.funnel[1:-1]
        if count
    ]
    if rows:
        ui.table(pd.DataFrame(rows), hide_index=True)
    excluded = prep.analytic[prep.analytic["exclusion_reason"] != ""]
    with st.expander(f"View the {fmt_int(len(excluded))} excluded records"):
        cols = [c for c in DEFAULT_COLUMNS + ["exclusion_reason"] if c in excluded.columns]
        ui.table(excluded[cols], hide_index=True, height=360, column_config=_column_config(cols))
        ui.download(
            "Download excluded records (CSV)",
            excluded[cols].to_csv(index=False).encode("utf-8"),
            "homevalue_excluded_records.csv",
            "text/csv",
            key="dl_excluded",
        )


# ------------------------------------------------------------------ market exploration
def _market(ds: LoadedDataset, filtered: pd.DataFrame | None) -> None:
    prep = ds.prep
    model_df = prep.model_df
    ui.section("Market exploration", "Interactive charts built from the model-ready records.")
    use_filters = st.toggle("Apply the dataset explorer filters to these charts", key="ex_use_filters")
    df = model_df
    if use_filters:
        if filtered is None:
            filtered = apply_filters(prep.analytic, st.session_state)
        df = model_df.loc[model_df.index.intersection(filtered.index)]
    if len(df) < 20:
        ui.callout("info", "Too few properties for charts", "Fewer than 20 model-ready properties match. Widen the filters or switch them off.")
        return
    ui.note(
        f"Charts use {fmt_int(len(df))} model-ready properties"
        + (" matching your filters." if use_filters else ".")
        + " Turning filters on or off here never changes the data the model learns from."
    )

    st.markdown("#### Price distribution")
    ui.plot(charts.price_distribution(df), "ch_price_dist")

    st.markdown("#### Price vs total area")
    log_scale = st.toggle("Logarithmic axes", key="ex_log")
    ui.plot(charts.price_vs_area(df, log_scale), "ch_price_area")

    st.markdown("#### Price by BHK")
    ui.plot(charts.price_by_bhk(df), "ch_price_bhk")

    st.markdown("#### Location comparison")
    counts = df["location"].dropna().value_counts()
    eligible = counts[counts >= 20].index.tolist() or counts.head(15).index.tolist()
    options = sorted(eligible)
    if "ex_locs" not in st.session_state:
        st.session_state["ex_locs"] = [x for x in counts.index if x in options][:8]
    st.session_state["ex_locs"] = [x for x in st.session_state["ex_locs"] if x in options]
    st.multiselect("Locations to compare (locations with at least 20 listings)", options, key="ex_locs")
    metric = ui.segmented("Metric", ["Median price", "Median price per sq ft"], "ex_loc_metric")
    if st.session_state["ex_locs"]:
        ui.plot(charts.location_comparison(df, st.session_state["ex_locs"], metric), "ch_loc")
    else:
        ui.note("Select at least one location to compare.")

    st.markdown("#### Price per sq ft")
    ui.note("Price per sq ft is calculated as price ÷ total area for analysis only. It is never used as a model input.")
    p1, p2 = st.columns(2)
    with p1:
        ui.plot(charts.pps_distribution(df), "ch_pps")
    with p2:
        ui.plot(charts.pps_by_area_type(df), "ch_pps_type")

    st.markdown("#### Correlation")
    ui.plot(charts.correlation_heatmap(df), "ch_corr")
    corr = df[["total_sqft", "bhk", "bath", "balcony", "price"]].corr()
    pairs = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    best = pairs.abs().idxmax()
    ui.note(
        f"The strongest relationship among these variables is between {best[0]} and {best[1]} (correlation "
        f"{pairs[best]:+.2f}). Only numerical variables are included; correlating categories such as location "
        "would be misleading."
    )


# ------------------------------------------------------------------ entry point
def render(ds: LoadedDataset) -> None:
    _init_filters(ds.prep.analytic)
    view = ui.segmented("Explore", VIEWS, "ex_view")
    if view == "Dataset explorer":
        _explorer(ds)
    elif view == "Data quality":
        _quality(ds)
    else:
        _market(ds, None)
    st.write("")
    ui.button("Continue to Model", key="btn_ex_next", type="primary", on_click=ui.goto, args=("Model",))
