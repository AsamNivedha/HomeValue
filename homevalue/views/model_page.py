"""Model: how well does it work, and what has it learned?"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import charts, config as cfg, ui
from ..formatting import fmt_int, fmt_lakh, fmt_num
from ..insights import describe_actual_vs_predicted, describe_error_view
from ..state import LoadedDataset, safe_get_model

ERROR_VIEWS = ["Residual distribution", "Error by price range", "Error vs predicted price", "Error by BHK"]


def _features_table(model) -> pd.DataFrame:
    n_loc_kept = len(model.kept_locations)
    rows = [
        ("BHK", "Number", "Read from the size description; missing values filled with the training median"),
        ("Total area", "Number", "Square feet; ranges use their midpoint; missing values filled with the training median"),
        ("Bathrooms", "Number", "Missing values filled with the training median"),
        ("Balconies", "Number", "Missing values filled with the training median"),
        ("Location", "Category", f"One column per location with at least {cfg.MIN_LOCATION_COUNT} training listings "
                                 f"({n_loc_kept:,} locations); all others and unseen locations share an 'Other' column"),
        ("Area type", "Category", "One column per area type; unseen values are ignored safely"),
        ("Availability", "Category", "Ready to Move or Available Later"),
    ]
    return pd.DataFrame(rows, columns=["Feature", "Type", "How it is prepared"])


def render(ds: LoadedDataset) -> None:
    ui.section("Model", "How well does the model work, and what has it learned?")
    st.markdown(
        "HomeValue uses Linear Regression to learn the relationship between property characteristics and listed "
        "prices in the training data."
    )
    model, err = safe_get_model()
    if err is not None:
        ui.error_panel(err)
        return
    m = model.metrics

    # ---- configuration
    ui.section("Configuration")
    ui.metric_grid(
        [
            ("Model", "Linear Regression", "ordinary least squares"),
            ("Target", f"Price ({cfg.PRICE_UNIT})", "listed price"),
            ("Features", f"{len(cfg.FEATURES)} inputs", f"{fmt_int(model.n_encoded_features)} columns after encoding"),
            ("Training records", fmt_int(model.n_train), "80% of model-ready records"),
            ("Testing records", fmt_int(model.n_test), "20%, never seen during training"),
        ]
    )
    with st.expander("Features used"):
        ui.table(_features_table(model), hide_index=True)
        ui.note(
            "Not used as inputs: society (many distinct values, many missing) and price per sq ft (derived from the "
            "target). Preprocessing is fitted on the training records only, and the same fitted pipeline makes every "
            "prediction on the Predict page."
        )

    # ---- performance
    ui.section("Performance", "Measured on the held-out test records.")
    ui.metric_grid(
        [
            ("R²", fmt_num(m["r2"], 3), "Measures how much of the variation in observed prices is explained by the model."),
            ("MAE", fmt_lakh(m["mae"]), "The average absolute difference between predicted and observed prices."),
            ("RMSE", fmt_lakh(m["rmse"]), "Similar to MAE, but larger errors have greater influence."),
        ]
    )
    ui.note(
        f"For context, MAE is about {m['mae'] / m['median_price_test']:.0%} of the median test price "
        f"({fmt_lakh(m['median_price_test'])}). Training R² is {fmt_num(m['train_r2'], 3)} versus {fmt_num(m['r2'], 3)} on "
        "the test set. MAE describes typical error across the test set; it is not a confidence interval for any single estimate."
    )

    # ---- actual vs predicted
    ui.section("Actual vs predicted")
    fig = charts.actual_vs_predicted(model.test_frame)
    ui.plot(fig, "ch_avp")
    ui.callout("info", "What this shows", describe_actual_vs_predicted(model))
    ui.download(
        "Download this chart (HTML)",
        fig.to_html(include_plotlyjs="cdn", full_html=True),
        "homevalue_actual_vs_predicted.html",
        "text/html",
        key="dl_avp",
    )

    # ---- error analysis
    ui.section("Error analysis", "Where does the model struggle?")
    view = ui.segmented("Error view", ERROR_VIEWS, "nav_err_view")
    builders = {
        "Residual distribution": charts.residual_distribution,
        "Error by price range": charts.error_by_price_range,
        "Error vs predicted price": charts.error_vs_predicted,
        "Error by BHK": charts.error_by_bhk,
    }
    ui.plot(builders[view](model.test_frame), f"ch_err_{view}")
    ui.callout("info", "What this shows", describe_error_view(model, view))

    # ---- coefficients
    _coefficients(model)

    # ---- export
    ui.section("Export")
    metrics_df = pd.DataFrame(
        [
            ("Model", "Linear Regression"),
            ("R2 (test)", m["r2"]),
            ("MAE lakh (test)", m["mae"]),
            ("RMSE lakh (test)", m["rmse"]),
            ("Median absolute error lakh (test)", m["median_ae"]),
            ("R2 (train)", m["train_r2"]),
            ("MAE lakh (train)", m["train_mae"]),
            ("Training records", model.n_train),
            ("Testing records", model.n_test),
            ("Encoded features", model.n_encoded_features),
            ("Dataset", ds.filename),
        ],
        columns=["metric", "value"],
    )
    e1, e2 = st.columns(2)
    with e1:
        ui.download("Download model metrics (CSV)", metrics_df.to_csv(index=False).encode("utf-8"),
                    "homevalue_model_metrics.csv", "text/csv", key="dl_metrics")
    with e2:
        cols = ["location", "bhk", "total_sqft", "bath", "balcony", "area_type", "availability_group", "actual", "predicted", "error"]
        ui.download("Download test-set predictions (CSV)", model.test_frame[cols].to_csv(index=False).encode("utf-8"),
                    "homevalue_test_predictions.csv", "text/csv", key="dl_test_pred")

    st.write("")
    ui.button("Continue to Predict", key="md_next", type="primary", on_click=ui.goto, args=("Predict",))


def _coefficients(model) -> None:
    coefs = model.coefficients
    ui.section("What the model learned", "Coefficients in plain language.")
    ui.callout(
        "info",
        "How to read coefficients",
        "A coefficient is the model's learned relationship with price while accounting for the other included features. "
        "It is not a universal measure of feature importance. Features that move together (such as area, BHK and "
        "bathrooms) share their influence, so a single coefficient can look counter-intuitive on its own.",
    )
    unit = {"bhk": "per additional bedroom", "total_sqft": "per additional sq ft", "bath": "per additional bathroom",
            "balcony": "per additional balcony"}
    numeric = coefs[coefs["group"].isin(unit)]
    rows = [
        {"Feature": cfg.FEATURE_LABELS[r.group], "Change in estimate (₹ lakh)": r.coefficient,
         "Reading": f"{r.coefficient:+,.2f} lakh {unit[r.group]}, other features held fixed"}
        for r in numeric.itertuples()
    ]
    ui.table(pd.DataFrame(rows), hide_index=True,
             column_config={"Change in estimate (₹ lakh)": st.column_config.NumberColumn(format="%.3f")})

    st.markdown("**Area type and availability**")
    cat = coefs[coefs["group"].isin(["area_type", "availability_group"])].copy()
    cat["Feature"] = cat["group"].map(cfg.FEATURE_LABELS)
    cat = cat.rename(columns={"label": "Category", "relative_effect": "Adjustment (₹ lakh)"})
    ui.table(cat[["Feature", "Category", "Adjustment (₹ lakh)"]].sort_values(["Feature", "Adjustment (₹ lakh)"], ascending=[True, False]),
             hide_index=True, column_config={"Adjustment (₹ lakh)": st.column_config.NumberColumn(format="%.2f")})

    st.markdown("**Locations**")
    ui.note(
        "Location coefficients are differences relative to the average location in the model, not independent "
        "guarantees about property value. Locations with few listings share the 'Other' adjustment."
    )
    ui.plot(charts.location_effects(coefs), "ch_loc_effects")
    loc = coefs[coefs["group"] == "location"].copy()
    loc["Listings in training data"] = loc["label"].map(model.location_counts).fillna(0).astype(int)
    rare_total = int(model.location_counts[model.location_counts < cfg.MIN_LOCATION_COUNT].sum())
    loc.loc[loc["label"] == "Other (rarely listed locations)", "Listings in training data"] = rare_total
    loc = loc.rename(columns={"label": "Location", "relative_effect": "Adjustment (₹ lakh)"})
    loc = loc[["Location", "Adjustment (₹ lakh)", "Listings in training data"]].sort_values("Adjustment (₹ lakh)", ascending=False)
    with st.expander(f"All {len(loc):,} location adjustments"):
        ui.table(loc, hide_index=True, height=360,
                 column_config={"Adjustment (₹ lakh)": st.column_config.NumberColumn(format="%.2f")})
