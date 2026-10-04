"""Predict: tell HomeValue about a property."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from .. import config as cfg, ui
from ..errors import HomeValueError
from ..formatting import fmt_crore, fmt_int, fmt_lakh, fmt_sqft
from ..modeling import TrainedModel, predict_one, validate_inputs
from ..state import LoadedDataset, add_history, clear_history, safe_get_model


def _init_inputs(model: TrainedModel) -> None:
    d = model.defaults
    defaults = {
        "pr_bhk": d["bhk"],
        "pr_sqft": d["total_sqft"],
        "pr_bath": d["bath"],
        "pr_balcony": d["balcony"],
        "pr_location": d["location"],
        "pr_area_type": d["area_type"],
        "pr_avail": d["availability_group"],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["pr_location"] not in model.locations + [cfg.OTHER_LOCATION]:
        st.session_state["pr_location"] = d["location"]
    if st.session_state["pr_area_type"] not in model.area_types:
        st.session_state["pr_area_type"] = d["area_type"]


def _form(model: TrainedModel) -> bool:
    with st.form("pr_form"):
        st.markdown("**Property**")
        c1, c2 = st.columns(2)
        with c1:
            st.number_input("BHK (bedrooms)", min_value=0, max_value=30, step=1, key="pr_bhk")
        with c2:
            st.number_input("Total area (sq ft)", min_value=0, max_value=200000, step=50, key="pr_sqft")
        c3, c4 = st.columns(2)
        with c3:
            st.number_input("Bathrooms", min_value=0, max_value=30, step=1, key="pr_bath")
        with c4:
            st.number_input("Balconies", min_value=0, max_value=20, step=1, key="pr_balcony")

        st.markdown("**Location**")
        st.selectbox(
            "Location (type to search)",
            model.locations + [cfg.OTHER_LOCATION],
            key="pr_location",
            help=f"{len(model.locations):,} locations are available. Choose '{cfg.OTHER_LOCATION}' if yours is missing.",
        )

        st.markdown("**Property characteristics**")
        c5, c6 = st.columns(2)
        with c5:
            st.selectbox("Area type", model.area_types, key="pr_area_type")
        with c6:
            st.radio("Availability", model.availability_options, key="pr_avail", horizontal=True)
        return ui.submit("Estimate value", type="primary")


def _collect() -> dict:
    s = st.session_state
    return {
        "bhk": s["pr_bhk"],
        "total_sqft": s["pr_sqft"],
        "bath": s["pr_bath"],
        "balcony": s["pr_balcony"],
        "location": s["pr_location"],
        "area_type": s["pr_area_type"],
        "availability_group": s["pr_avail"],
    }


def _run_prediction(model: TrainedModel) -> list[str]:
    """Validate, predict and store the result. Returns validation errors (if any)."""
    inputs = _collect()
    errors, warnings = validate_inputs(inputs, model)
    if errors:
        st.session_state["last_prediction"] = None
        return errors
    try:
        estimate = predict_one(model, inputs)
    except Exception:
        st.session_state["last_prediction"] = None
        return ["The estimate could not be calculated for these values. Please check the inputs and try again."]
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "location": inputs["location"],
        "bhk": int(inputs["bhk"]),
        "total_sqft": float(inputs["total_sqft"]),
        "bath": int(inputs["bath"]),
        "balcony": int(inputs["balcony"]),
        "area_type": inputs["area_type"],
        "availability": inputs["availability_group"],
        "estimated_value_lakh": round(estimate, 2) if estimate > 0 else None,
    }
    st.session_state["last_prediction"] = {"inputs": inputs, "estimate": estimate, "warnings": warnings, "record": record}
    if estimate > 0:
        add_history(record)
    return []


def _result(model: TrainedModel) -> None:
    last = st.session_state.get("last_prediction")
    if not last:
        return
    estimate, inputs = last["estimate"], last["inputs"]
    if estimate <= 0:
        ui.callout(
            "attention",
            "HomeValue could not produce a meaningful estimate",
            "The linear model returned a value of zero or below for these characteristics, which can happen for very "
            "small properties that lie at the edge of the training data. Try adjusting the area or BHK.",
        )
    else:
        crore = f" · about {fmt_crore(estimate)}" if estimate >= 100 else ""
        ui.md(
            '<div class="hv-result"><div class="lab">Estimated value</div>'
            f'<div class="val">{ui.esc(fmt_lakh(estimate))}</div>'
            f'<div class="sub">Model estimate{ui.esc(crore)}</div>'
            '<div class="ctx">Based on the property\'s characteristics and patterns learned from the historical housing dataset.</div></div>'
        )
    for w in last["warnings"]:
        ui.callout("attention", "Keep in mind", w)

    ui.section("Property summary")
    ui.key_values(
        [
            ("Location", inputs["location"]),
            ("BHK", fmt_int(inputs["bhk"])),
            ("Total area", fmt_sqft(inputs["total_sqft"])),
            ("Bathrooms", fmt_int(inputs["bath"])),
            ("Balconies", fmt_int(inputs["balcony"])),
            ("Area type", inputs["area_type"]),
            ("Availability", inputs["availability_group"]),
        ]
    )
    ui.section("Model context")
    ui.key_values(
        [
            ("Model", "Linear Regression"),
            ("Test-set MAE", fmt_lakh(model.metrics["mae"])),
            ("Training records", fmt_int(model.n_train)),
        ]
    )
    ui.note(
        "Test-set MAE describes the typical size of errors across the test properties. It is not a prediction "
        "interval for this individual estimate, which can be higher or lower."
    )


def _history() -> None:
    history = st.session_state.get("history", [])
    if not history:
        return
    ui.section("Session history", "Estimates made in this session. Nothing is stored beyond it.")
    df = pd.DataFrame(history).rename(
        columns={
            "timestamp": "Time", "location": "Location", "bhk": "BHK", "total_sqft": "Area (sq ft)", "bath": "Bathrooms",
            "balcony": "Balconies", "area_type": "Area type", "availability": "Availability",
            "estimated_value_lakh": "Estimated value (₹ lakh)",
        }
    )
    ui.table(df.iloc[::-1], hide_index=True, height=min(360, 60 + 36 * len(df)))
    h1, h2 = st.columns(2)
    with h1:
        ui.download("Download prediction history (CSV)", df.to_csv(index=False).encode("utf-8"),
                    "homevalue_prediction_history.csv", "text/csv", key="dl_history")
    with h2:
        ui.button("Clear history", key="pr_clear", on_click=clear_history)


def render(ds: LoadedDataset) -> None:
    ui.section("Predict", "Tell HomeValue about a property.")
    model, err = safe_get_model()
    if err is not None:
        ui.error_panel(err)
        return
    _init_inputs(model)
    submitted = _form(model)
    if submitted:
        errors = _run_prediction(model)
        if errors:
            ui.callout_list("error", "Please correct the following before estimating", errors)
    _result(model)
    _history()
    st.write("")
    ui.button("Continue to Insights", key="pr_next", type="primary", on_click=ui.goto, args=("Insights",))
