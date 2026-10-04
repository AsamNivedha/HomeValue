"""Overview: what am I looking at?"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import ui
from ..formatting import fmt_int, fmt_lakh, fmt_sqft
from ..insights import overview_observations
from ..state import LoadedDataset

_MODEL_INPUTS = {
    "total_sqft": "Model input (total area)",
    "size": "Model input (read as BHK)",
    "location": "Model input",
    "bath": "Model input",
    "balcony": "Model input",
    "area_type": "Model input",
    "availability": "Model input (Ready to Move / Available Later)",
    "price": "Target",
    "society": "Exploration only",
}


def _variables_table(ds: LoadedDataset) -> pd.DataFrame:
    rows = []
    for col in ds.raw.columns:
        mapped = ds.prep.column_map.get(str(col), str(col))
        series = ds.raw[col]
        rows.append(
            {
                "Column": str(col),
                "Type": "Number" if pd.api.types.is_numeric_dtype(series) else "Text",
                "Missing": int(series.isna().sum()),
                "Role in HomeValue": _MODEL_INPUTS.get(mapped, "Not used by the model"),
            }
        )
    rows.append({"Column": "price_per_sqft (derived)", "Type": "Number", "Missing": None,
                 "Role in HomeValue": "Analysis only; never a model input"})
    return pd.DataFrame(rows)


def render(ds: LoadedDataset) -> None:
    prep, s = ds.prep, ds.prep.summary
    ui.section("Overview", "What am I looking at?")
    ui.metric_grid(
        [
            ("Properties", fmt_int(s["n_raw"]), "listings in the uploaded file"),
            ("Variables", fmt_int(s["n_columns"]), "original columns"),
            ("Average price", fmt_lakh(s["price_mean"]), ""),
            ("Median price", fmt_lakh(s["price_median"]), ""),
            ("Average area", fmt_sqft(s["area_mean"]), ""),
        ]
    )

    ui.section("Data quality", "A plain-language health check of the uploaded data.")
    status = prep.quality_status
    kind = {"Needs attention": "attention", "Good": "info", "Excellent": "success"}[status]
    headline = {
        "Needs attention": "Data quality needs attention",
        "Good": "Data quality is good, with minor notes",
        "Excellent": "No data-quality issues detected",
    }[status]
    if prep.issues:
        ui.callout(kind, headline, "HomeValue found the following. None of them stops the analysis; each is handled automatically.")
        ui.bullets([f"{i.title}. {i.action}" for i in prep.issues[:4]])
    else:
        ui.callout(kind, headline, "Every check passed.")
    ui.button(
        "Open the data quality center",
        key="ov_dq",
        on_click=ui.goto,
        args=("Explore",),
        kwargs={"ex_view": "Data quality"},
    )

    ui.section("From original data to model-ready data", "What was uploaded versus what the model learns from.")
    ui.flow(prep.funnel)
    ui.note(
        "Your original file is preserved untouched. Exploring and filtering never change the model-ready data, "
        "and the model is trained only on the model-ready records."
    )

    ui.section("What stands out?", "Observations computed from this dataset. They describe associations, not causes.")
    observations = overview_observations(prep)
    if observations:
        ui.bullets(observations)
    else:
        ui.note("Not enough prepared data to summarise.")

    with st.expander("Variables and their role"):
        ui.table(_variables_table(ds), hide_index=True, column_config={"Missing": st.column_config.NumberColumn(format="%d")})

    st.write("")
    ui.button("Continue to Explore", key="ov_next", type="primary", on_click=ui.goto, args=("Explore",))
