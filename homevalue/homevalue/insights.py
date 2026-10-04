"""Dynamically generated observations about the data and the model.

Every statement is computed from the active dataset (and trained model). The
wording describes associations observed in this data; it never claims causation
and never makes statements about the wider real-estate market.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from .cleaning import PreparedData
from .formatting import fmt_int, fmt_lakh, fmt_num, fmt_pct, fmt_rupees
from .modeling import TrainedModel

CATEGORIES = ["Market", "Property characteristics", "Location", "Model", "Data quality"]


@dataclass
class Insight:
    key: str
    category: str
    text: str


def _strength(r: float) -> str:
    a = abs(r)
    return "weak" if a < 0.3 else "moderate" if a < 0.6 else "strong"


def _association(r: float, subject: str, other: str) -> str:
    direction = "higher" if r > 0 else "lower"
    if abs(r) < 0.1:
        return f"There is little association between {subject} and {other} in this dataset (rank correlation {r:+.2f})."
    return (
        f"Properties with larger {subject} tend to have {direction} {other} in this dataset "
        f"({_strength(r)} association, rank correlation {r:+.2f})."
    )


# ------------------------------------------------------------------ market
def _median_vs_mean(prep: PreparedData):
    p = prep.model_df[cfg.TARGET]
    mean, median = p.mean(), p.median()
    gap = (mean - median) / median
    if gap > 0.1:
        text = (
            f"The average listed price ({fmt_lakh(mean)}) is {gap:.0%} above the median ({fmt_lakh(median)}), "
            "which means a minority of high-priced listings pull the average up."
        )
    else:
        text = (
            f"The average listed price ({fmt_lakh(mean)}) and the median ({fmt_lakh(median)}) are close, "
            "so the price distribution is fairly balanced around the middle."
        )
    return Insight("median_mean", "Market", text)


def _price_spread(prep: PreparedData):
    p = prep.model_df[cfg.TARGET]
    lo, hi = p.quantile(0.1), p.quantile(0.9)
    return Insight(
        "price_spread",
        "Market",
        f"The middle 80% of prepared listings are priced between {fmt_lakh(lo)} and {fmt_lakh(hi)}; "
        f"the most expensive listing is {fmt_lakh(p.max(), 0)}.",
    )


# ------------------------------------------------------------------ property
def _common_bhk(prep: PreparedData):
    counts = prep.model_df["bhk"].astype(int).value_counts()
    top = counts.index[0]
    return Insight(
        "common_bhk",
        "Property characteristics",
        f"{top} BHK is the most common configuration, covering {counts.iloc[0] / counts.sum():.0%} "
        f"of prepared listings ({fmt_int(counts.iloc[0])} properties).",
    )


def _area_price(prep: PreparedData):
    r = prep.model_df[["total_sqft", cfg.TARGET]].corr(method="spearman").iloc[0, 1]
    return Insight("area_price", "Property characteristics", _association(r, "areas", "listed prices"))


def _bath_price(prep: PreparedData):
    d = prep.model_df[["bath", cfg.TARGET]].dropna()
    r = d.corr(method="spearman").iloc[0, 1]
    return Insight("bath_price", "Property characteristics", _association(r, "bathroom counts", "listed prices"))


def _bhk_price(prep: PreparedData):
    d = prep.model_df.assign(b=prep.model_df["bhk"].astype(int))
    g = d.groupby("b")[cfg.TARGET].agg(["median", "size"])
    g = g[g["size"] >= cfg.INSIGHT_MIN_GROUP]
    if len(g) < 2:
        return None
    lo, hi = g.index.min(), g.index.max()
    return Insight(
        "bhk_price",
        "Property characteristics",
        f"Median listed price is {fmt_lakh(g.loc[lo, 'median'])} for {lo} BHK properties and "
        f"{fmt_lakh(g.loc[hi, 'median'])} for {hi} BHK properties (groups with at least "
        f"{cfg.INSIGHT_MIN_GROUP} listings).",
    )


def _area_type(prep: PreparedData):
    g = prep.model_df.groupby("area_type")["price_per_sqft"].agg(["median", "size"])
    g = g[g["size"] >= cfg.INSIGHT_MIN_GROUP].sort_values("median")
    if len(g) < 2:
        return None
    return Insight(
        "area_type",
        "Property characteristics",
        f"Median price per sq ft is highest for '{g.index[-1]}' listings ({fmt_rupees(g['median'].iloc[-1])}) "
        f"and lowest for '{g.index[0]}' listings ({fmt_rupees(g['median'].iloc[0])}). "
        "Area types measure space differently, so part of this gap may reflect how area is reported.",
    )


def _availability(prep: PreparedData):
    g = prep.model_df.groupby("availability_group")["price_per_sqft"].agg(["median", "size"])
    if not {"Ready to Move", "Available Later"} <= set(g.index) or (g["size"] < cfg.INSIGHT_MIN_GROUP).any():
        return None
    return Insight(
        "availability",
        "Property characteristics",
        f"{g.loc['Ready to Move', 'size'] / g['size'].sum():.0%} of prepared listings are ready to move. "
        f"Median price per sq ft is {fmt_rupees(g.loc['Ready to Move', 'median'])} for ready-to-move listings versus "
        f"{fmt_rupees(g.loc['Available Later', 'median'])} for those available later.",
    )


# ------------------------------------------------------------------ location
def _top_location(prep: PreparedData):
    counts = prep.model_df["location"].dropna().value_counts()
    top = counts.index[0]
    return Insight(
        "top_location",
        "Location",
        f"{top} is the most represented location with {fmt_int(counts.iloc[0])} listings "
        f"({counts.iloc[0] / counts.sum():.1%} of prepared records); the data covers {fmt_int(len(counts))} locations in total.",
    )


def _location_spread(prep: PreparedData):
    g = prep.model_df.groupby("location")["price_per_sqft"].agg(["median", "size"])
    g = g[g["size"] >= cfg.INSIGHT_MIN_GROUP].sort_values("median")
    if len(g) < 3:
        return None
    return Insight(
        "location_spread",
        "Location",
        f"Among the {len(g)} locations with at least {cfg.INSIGHT_MIN_GROUP} listings, median price per sq ft ranges "
        f"from {fmt_rupees(g['median'].iloc[0])} ({g.index[0]}) to {fmt_rupees(g['median'].iloc[-1])} ({g.index[-1]}), "
        f"a {g['median'].iloc[-1] / g['median'].iloc[0]:.1f}× difference.",
    )


def _location_sparsity(prep: PreparedData):
    counts = prep.model_df["location"].dropna().value_counts()
    rare = counts[counts < cfg.MIN_LOCATION_COUNT]
    return Insight(
        "location_sparsity",
        "Location",
        f"{len(rare) / len(counts):.0%} of locations have fewer than {cfg.MIN_LOCATION_COUNT} listings; together they cover "
        f"{rare.sum() / counts.sum():.0%} of prepared records, so estimates for them rely on the shared 'Other' group.",
    )


# ------------------------------------------------------------------ model
def _model_fit(model: TrainedModel):
    m = model.metrics
    leftover = 1 - m["r2"]
    text = (
        f"On the {fmt_int(model.n_test)} held-out test properties the model explains about {m['r2']:.0%} of the "
        f"variation in listed prices"
    )
    text += f", leaving {leftover:.0%} unexplained by the included characteristics." if leftover > 0.05 else "."
    return Insight("model_fit", "Model", text)


def _model_error(model: TrainedModel):
    m = model.metrics
    text = (
        f"The typical absolute error on the test set is {fmt_lakh(m['mae'])} (MAE), about "
        f"{m['mae'] / m['median_price_test']:.0%} of the median test price. This describes average test-set error, "
        "not an interval for any single estimate."
    )
    if m["rmse"] > 1.5 * m["mae"]:
        text += f" RMSE ({fmt_lakh(m['rmse'])}) is much larger than MAE, which indicates a minority of large errors."
    return Insight("model_error", "Model", text)


def _model_pattern(model: TrainedModel):
    t = model.test_frame
    band = pd.qcut(t["actual"], 4, labels=["lowest", "lower-middle", "upper-middle", "highest"], duplicates="drop")
    g = t.groupby(band, observed=True)["abs_error"].median()
    if len(g) < 2:
        return None
    lo, hi = g.iloc[0], g.iloc[-1]
    if hi > 1.5 * lo:
        text = (
            f"Absolute errors are larger for higher-priced properties: the median error is {fmt_lakh(hi)} in the "
            f"highest price quarter versus {fmt_lakh(lo)} in the lowest."
        )
    elif lo > 1.5 * hi:
        text = (
            f"Absolute errors are larger for lower-priced properties: the median error is {fmt_lakh(lo)} in the "
            f"lowest price quarter versus {fmt_lakh(hi)} in the highest."
        )
    else:
        text = (
            f"Median absolute error is similar across price quarters ({fmt_lakh(lo)} in the lowest, "
            f"{fmt_lakh(hi)} in the highest)."
        )
    return Insight("model_pattern", "Model", text)


def _model_negative(model: TrainedModel):
    neg = int((model.test_frame["predicted"] < 0).sum())
    if not neg:
        return None
    return Insight(
        "model_negative",
        "Model",
        f"{neg} of {fmt_int(model.n_test)} test predictions were below zero. A linear model can extrapolate below zero "
        "for small, low-priced properties; HomeValue does not present such values as estimates.",
    )


def _model_area_coef(model: TrainedModel):
    row = model.coefficients[model.coefficients["label"] == "total_sqft"]
    if row.empty:
        return None
    c = float(row["coefficient"].iloc[0])
    return Insight(
        "model_area_coef",
        "Model",
        f"Within the fitted model, each additional 100 sq ft is associated with a change of {fmt_lakh(c * 100)} in the "
        "estimate when the other included characteristics are held fixed. This is a statement about the model, "
        "not about what causes prices.",
    )


# ------------------------------------------------------------------ data quality
def _largest_issue(prep: PreparedData):
    relevant = [i for i in prep.issues if i.key not in {"sparse_locations", "empty_columns", "constant_columns", "missing_society"}]
    if not relevant:
        return Insight("largest_issue", "Data quality", "No data-quality issues were detected in this dataset.")
    top = max(relevant, key=lambda i: i.count)
    return Insight(
        "largest_issue",
        "Data quality",
        f"The data-quality issue affecting the most records: {top.title[0].lower() + top.title[1:]}. {top.action}",
    )


def _retained(prep: PreparedData):
    s = prep.summary
    return Insight(
        "retained",
        "Data quality",
        f"{fmt_int(s['n_model'])} of {fmt_int(s['n_raw'])} original records ({s['n_model'] / s['n_raw']:.0%}) are "
        "model-ready. Estimates reflect these records, so unusual or very rare property types are less well represented.",
    )


def _society(prep: PreparedData):
    a = prep.analytic
    if "society" not in a.columns:
        return None
    share = a["society"].isna().mean()
    if share < 0.2:
        return None
    return Insight(
        "society",
        "Data quality",
        f"Society is missing for {share:.0%} of records, which is one reason it is not used as a model input.",
    )


_DATA_FUNCS = [
    _median_vs_mean, _price_spread, _common_bhk, _area_price, _bath_price, _bhk_price, _area_type,
    _availability, _top_location, _location_spread, _location_sparsity, _largest_issue, _retained, _society,
]
_MODEL_FUNCS = [_model_fit, _model_error, _model_pattern, _model_negative, _model_area_coef]


def _run(funcs, arg) -> list[Insight]:
    out = []
    for fn in funcs:
        try:
            result = fn(arg)
        except Exception:  # an insight that cannot be computed is simply skipped
            result = None
        if result is not None:
            out.append(result)
    return out


def compute_insights(prep: PreparedData, model: TrainedModel | None = None) -> dict[str, list[Insight]]:
    """All insights grouped by category (model insights only when a model exists)."""
    items = _run(_DATA_FUNCS, prep)
    if model is not None:
        items += _run(_MODEL_FUNCS, model)
    grouped = {c: [] for c in CATEGORIES}
    for item in items:
        grouped[item.category].append(item)
    return grouped


def overview_observations(prep: PreparedData) -> list[str]:
    """A short, curated list for the Overview page."""
    wanted = ["median_mean", "top_location", "common_bhk", "area_price", "largest_issue"]
    items = {i.key: i for i in _run(_DATA_FUNCS, prep)}
    return [items[k].text for k in wanted if k in items]


def insights_markdown(grouped: dict[str, list[Insight]], dataset_name: str) -> str:
    lines = [f"# HomeValue insights — {dataset_name}", ""]
    for category, items in grouped.items():
        if not items:
            continue
        lines.append(f"## {category}")
        lines += [f"- {i.text}" for i in items]
        lines.append("")
    lines.append(f"_{cfg.DISCLAIMER}_")
    return "\n".join(lines)


def describe_actual_vs_predicted(model: TrainedModel) -> str:
    """Interpretation text shown below the Actual vs Predicted chart."""
    t = model.test_frame
    corr = float(np.corrcoef(t["actual"], t["predicted"])[0, 1])
    within20 = float((t["pct_error"].abs() <= 0.2).mean())
    slope = float(np.polyfit(t["actual"], t["predicted"], 1)[0])
    parts = [
        f"Predictions follow the observed price pattern with a correlation of {corr:.2f} on the test set, and "
        f"{within20:.0%} of test properties are predicted within 20% of their listed price. "
        "Points farther from the reference line represent larger prediction errors."
    ]
    if slope < 0.8:
        parts.append(
            "The predictions rise more slowly than the observed prices, so the highest-priced properties tend to be "
            "predicted below their listed price."
        )
    elif slope > 1.2:
        parts.append("The predictions rise faster than the observed prices across the range.")
    return " ".join(parts)


def describe_error_view(model: TrainedModel, view: str) -> str:
    t = model.test_frame
    if view == "Residual distribution":
        bias = t["error"].mean()
        skew = "over" if bias > 0 else "under"
        return (
            f"Errors (predicted minus actual) average {fmt_num(bias)} lakh, so on average the model {skew}-predicts. "
            f"Half of test errors are within {fmt_lakh(t['abs_error'].median())}, while the largest reaches "
            f"{fmt_lakh(t['abs_error'].max(), 0)}."
        )
    if view == "Error by price range":
        band = pd.qcut(t["actual"], 4, duplicates="drop")
        g = t.groupby(band, observed=True)["abs_error"].mean()
        r = t.groupby(band, observed=True)["pct_error"].apply(lambda s: s.abs().median())
        return (
            f"Mean absolute error is {fmt_lakh(g.iloc[0])} in the lowest price quarter and {fmt_lakh(g.iloc[-1])} in the highest. "
            f"Median relative error is {fmt_pct(r.iloc[0])} versus {fmt_pct(r.iloc[-1])}."
        )
    if view == "Error vs predicted price":
        corr = float(np.corrcoef(t["predicted"], t["abs_error"])[0, 1])
        if corr > 0.2:
            return "Absolute errors tend to grow as predicted prices increase."
        if corr < -0.2:
            return "Absolute errors tend to shrink as predicted prices increase."
        return "There is no strong relationship between the predicted price and the size of the error."
    g = t.assign(b=t["bhk"].astype(int)).groupby("b")["abs_error"].agg(["mean", "size"])
    g = g[g["size"] >= 20]
    if len(g) < 2:
        return "Too few test properties per BHK group for a reliable comparison."
    worst = g["mean"].idxmax()
    return f"Among BHK groups with at least 20 test properties, mean absolute error is highest for {worst} BHK ({fmt_lakh(g.loc[worst, 'mean'])})."
