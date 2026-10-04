"""Plotly figures. Every function takes plain DataFrames and returns a Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import config as cfg

PRIMARY = "#2B59C3"
ACCENT = "#E8590C"
MUTED = "#9AA3AF"
GRID = "#EEF0F3"
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "homevalue_chart"},
}

PRICE_AXIS = f"Price ({cfg.PRICE_UNIT})"
AREA_AXIS = f"Total area ({cfg.AREA_UNIT})"
PPS_AXIS = "Price per sq ft (₹)"


def _style(fig: go.Figure, title: str, x: str | None, y: str | None, height: int = 420, legend: bool = False):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=16)),
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(family=FONT, size=13, color="#1F2937"),
        showlegend=legend,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=x, showgrid=True, gridcolor=GRID, zeroline=False, automargin=True)
    fig.update_yaxes(title_text=y, showgrid=True, gridcolor=GRID, zeroline=False, automargin=True)
    return fig


def _clip(series: pd.Series, q: float = 0.99) -> pd.Series:
    s = series.dropna()
    return s[s <= s.quantile(q)] if len(s) else s


# ------------------------------------------------------------------ market exploration
def price_distribution(df: pd.DataFrame) -> go.Figure:
    prices = df[cfg.TARGET].dropna()
    shown = _clip(prices)
    median = float(prices.median())
    fig = go.Figure(
        go.Histogram(
            x=shown,
            nbinsx=50,
            marker=dict(color=PRIMARY, line=dict(color="white", width=0.5)),
            hovertemplate="Price: %{x} lakh<br>Properties: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=median, line_dash="dash", line_color=ACCENT,
        annotation_text=f"Median ₹{median:,.0f} lakh", annotation_position="top right",
    )
    return _style(fig, "Price distribution (up to the 99th percentile)", PRICE_AXIS, "Properties")


def price_vs_area(df: pd.DataFrame, log_scale: bool = False, max_points: int = 4000) -> go.Figure:
    d = df.dropna(subset=["total_sqft", cfg.TARGET])
    if len(d) > max_points:
        d = d.sample(max_points, random_state=cfg.RANDOM_STATE)
    fig = go.Figure(
        go.Scatter(
            x=d["total_sqft"],
            y=d[cfg.TARGET],
            mode="markers",
            marker=dict(size=5, opacity=0.45, color=PRIMARY),
            customdata=np.column_stack([d["location"].fillna("Unknown").astype(str), d["bhk"].astype(int)]),
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]} BHK<br>Area: %{x:,.0f} sq ft"
                "<br>Price: ₹%{y:,.1f} lakh<extra></extra>"
            ),
        )
    )
    _style(fig, "Price vs total area", AREA_AXIS, PRICE_AXIS)
    if log_scale:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return fig


def price_by_bhk(df: pd.DataFrame, min_group: int = 10) -> go.Figure:
    d = df.assign(b=df["bhk"].astype(int))
    counts = d["b"].value_counts()
    keep = sorted(counts[counts >= min_group].index)
    fig = go.Figure()
    for b in keep:
        fig.add_trace(
            go.Box(
                y=d.loc[d["b"] == b, cfg.TARGET],
                name=f"{b} BHK",
                marker_color=PRIMARY,
                boxpoints=False,
                hovertemplate="%{y:,.1f} lakh<extra>" + f"{b} BHK" + "</extra>",
            )
        )
    top = float(_clip(d[cfg.TARGET], 0.995).max()) if len(d) else 1.0
    _style(fig, "Price by BHK", "Bedrooms (BHK)", PRICE_AXIS)
    fig.update_yaxes(range=[0, top * 1.05])
    return fig


def location_comparison(df: pd.DataFrame, locations: list[str], metric: str) -> go.Figure:
    col = cfg.TARGET if metric == "Median price" else "price_per_sqft"
    g = (
        df[df["location"].isin(locations)]
        .groupby("location")[col]
        .agg(["median", "size"])
        .sort_values("median")
    )
    label = PRICE_AXIS if col == cfg.TARGET else PPS_AXIS
    text = [f"{v:,.0f}" if col != cfg.TARGET else f"{v:,.1f}" for v in g["median"]]
    fig = go.Figure(
        go.Bar(
            x=g["median"], y=g.index, orientation="h", marker_color=PRIMARY, text=text, textposition="outside",
            customdata=g["size"],
            hovertemplate="%{y}<br>Median: %{x:,.1f}<br>Listings: %{customdata}<extra></extra>",
        )
    )
    height = max(320, 40 * len(g) + 120)
    _style(fig, f"{metric} by location", f"Median {label}", None, height=height)
    fig.update_yaxes(showgrid=False)
    return fig


def pps_distribution(df: pd.DataFrame) -> go.Figure:
    s = df["price_per_sqft"].dropna()
    shown = _clip(s)
    fig = go.Figure(
        go.Histogram(
            x=shown, nbinsx=50, marker=dict(color=PRIMARY, line=dict(color="white", width=0.5)),
            hovertemplate="₹%{x} per sq ft<br>Properties: %{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=float(s.median()), line_dash="dash", line_color=ACCENT,
        annotation_text=f"Median ₹{s.median():,.0f}", annotation_position="top right",
    )
    return _style(fig, "Price per sq ft distribution", PPS_AXIS, "Properties")


def pps_by_area_type(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, group in df.groupby("area_type"):
        if len(group) < 10:
            continue
        fig.add_trace(go.Box(y=group["price_per_sqft"], name=str(name), marker_color=PRIMARY, boxpoints=False))
    _style(fig, "Price per sq ft by area type", None, PPS_AXIS)
    return fig


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = {"total_sqft": "Area", "bhk": "BHK", "bath": "Bathrooms", "balcony": "Balconies", cfg.TARGET: "Price"}
    corr = df[list(cols)].corr(method="pearson")
    labels = list(cols.values())
    fig = go.Figure(
        go.Heatmap(
            z=corr.values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu",
            text=np.round(corr.values, 2), texttemplate="%{text}",
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Correlation"),
        )
    )
    _style(fig, "Correlation between numerical variables", None, None, height=420)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ model evaluation
def actual_vs_predicted(test: pd.DataFrame) -> go.Figure:
    lo = float(min(0.0, test["predicted"].min()))
    hi = float(max(test["actual"].max(), test["predicted"].max())) * 1.02
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test["actual"], y=test["predicted"], mode="markers", name="Test properties",
            marker=dict(size=5, opacity=0.5, color=PRIMARY),
            customdata=np.column_stack(
                [test["location"].fillna("Unknown").astype(str), test["bhk"].astype(int), test["total_sqft"].round(0)]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]} BHK · %{customdata[2]:,.0f} sq ft"
                "<br>Actual: ₹%{x:,.1f} lakh<br>Predicted: ₹%{y:,.1f} lakh<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
                   line=dict(color=ACCENT, dash="dash", width=2), hoverinfo="skip")
    )
    _style(fig, "Actual vs predicted price (test set)", f"Actual {PRICE_AXIS}", f"Predicted {PRICE_AXIS}", height=480, legend=True)
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(range=[lo, hi])
    return fig


def residual_distribution(test: pd.DataFrame) -> go.Figure:
    e = test["error"]
    shown = e[(e >= e.quantile(0.01)) & (e <= e.quantile(0.99))]
    fig = go.Figure(
        go.Histogram(x=shown, nbinsx=60, marker=dict(color=PRIMARY, line=dict(color="white", width=0.5)),
                     hovertemplate="Error: %{x} lakh<br>Properties: %{y}<extra></extra>")
    )
    fig.add_vline(x=0, line_dash="dash", line_color=ACCENT, annotation_text="No error", annotation_position="top right")
    return _style(fig, "Prediction errors (1st–99th percentile)", f"Predicted minus actual ({cfg.PRICE_UNIT})", "Properties")


def error_by_price_range(test: pd.DataFrame) -> go.Figure:
    band = pd.qcut(test["actual"], 4, duplicates="drop")
    g = test.groupby(band, observed=True).agg(mae=("abs_error", "mean"), n=("abs_error", "size"),
                                              lo=("actual", "min"), hi=("actual", "max"))
    labels = [f"₹{r.lo:,.0f}–{r.hi:,.0f} lakh" for r in g.itertuples()]
    fig = go.Figure(
        go.Bar(x=labels, y=g["mae"], marker_color=PRIMARY, text=[f"{v:,.1f}" for v in g["mae"]], textposition="outside",
               customdata=g["n"], hovertemplate="%{x}<br>Mean absolute error: %{y:,.1f} lakh<br>Properties: %{customdata}<extra></extra>")
    )
    return _style(fig, "Mean absolute error by actual price range (quarters)", "Actual price range", f"Mean absolute error ({cfg.PRICE_UNIT})")


def error_vs_predicted(test: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=test["predicted"], y=test["error"], mode="markers", marker=dict(size=5, opacity=0.45, color=PRIMARY),
                   hovertemplate="Predicted: ₹%{x:,.1f} lakh<br>Error: %{y:,.1f} lakh<extra></extra>")
    )
    fig.add_hline(y=0, line_dash="dash", line_color=ACCENT)
    return _style(fig, "Error vs predicted price", f"Predicted {PRICE_AXIS}", f"Predicted minus actual ({cfg.PRICE_UNIT})")


def error_by_bhk(test: pd.DataFrame, min_group: int = 20) -> go.Figure:
    g = test.assign(b=test["bhk"].astype(int)).groupby("b")["abs_error"].agg(["mean", "size"])
    g = g[g["size"] >= min_group]
    fig = go.Figure(
        go.Bar(x=[f"{b} BHK" for b in g.index], y=g["mean"], marker_color=PRIMARY, text=[f"{v:,.1f}" for v in g["mean"]],
               textposition="outside", customdata=g["size"],
               hovertemplate="%{x}<br>Mean absolute error: %{y:,.1f} lakh<br>Test properties: %{customdata}<extra></extra>")
    )
    return _style(fig, f"Mean absolute error by BHK (groups with ≥ {min_group} test properties)", "Bedrooms (BHK)", f"Mean absolute error ({cfg.PRICE_UNIT})")


def location_effects(coefs: pd.DataFrame, n: int = 10) -> go.Figure:
    loc = coefs[coefs["group"] == "location"].sort_values("relative_effect")
    picked = pd.concat([loc.head(n), loc.tail(n)]).drop_duplicates("label")
    colors = [PRIMARY if v >= 0 else MUTED for v in picked["relative_effect"]]
    fig = go.Figure(
        go.Bar(x=picked["relative_effect"], y=picked["label"], orientation="h", marker_color=colors,
               text=[f"{v:+,.0f}" for v in picked["relative_effect"]], textposition="outside",
               hovertemplate="%{y}<br>Relative effect: %{x:+,.1f} lakh<extra></extra>")
    )
    height = max(380, 28 * len(picked) + 120)
    _style(fig, f"Locations with the largest model adjustments (top and bottom {n})",
           f"Adjustment relative to the average location ({cfg.PRICE_UNIT})", None, height=height)
    fig.update_yaxes(showgrid=False)
    return fig
