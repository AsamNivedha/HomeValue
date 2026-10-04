"""HomeValue - property analytics and price estimation (Streamlit entry point)."""
from __future__ import annotations

import logging

import streamlit as st

st.set_page_config(page_title="HomeValue", layout="wide", initial_sidebar_state="collapsed")

from homevalue import config as cfg  # noqa: E402
from homevalue import state, ui  # noqa: E402
from homevalue.errors import HomeValueError  # noqa: E402
from homevalue.views import explore, insights_page, model_page, overview, predict  # noqa: E402

logger = logging.getLogger("homevalue")

PAGES = ["Overview", "Explore", "Model", "Predict", "Insights"]
RENDERERS = {
    "Overview": overview.render,
    "Explore": explore.render,
    "Model": model_page.render,
    "Predict": predict.render,
    "Insights": insights_page.render,
}
QUALITY_BADGE = {"Needs attention": "▲ Needs attention", "Good": "● Good", "Excellent": "✓ Excellent"}


def _load(loader, *args) -> None:
    """Run a dataset loader, converting every failure into a displayable error."""
    try:
        with st.spinner("Loading and validating the data…"):
            loader(*args)
    except HomeValueError as err:
        st.session_state["load_error"] = err
        return
    except Exception:
        logger.exception("Unexpected error while loading data")
        st.session_state["load_error"] = HomeValueError(
            "The file could not be processed",
            "Something unexpected happened while reading or validating this file.",
            [f"{n} — {d}" for n, d in cfg.REQUIRED_COLUMNS.items()],
            "Check that the file is a CSV with the expected columns, or explore the sample data.",
        )
        return
    st.rerun()


def landing() -> None:
    ui.brand()
    ui.md(
        '<div class="hv-hero"><div class="hv-eyebrow">Property analytics</div>'
        f"<h1>{ui.esc(cfg.TAGLINE)}</h1>"
        '<div class="hv-lede">HomeValue combines property characteristics, historical housing data and Linear '
        "Regression to produce an estimated property value, and shows you how the estimate was reached.</div></div>"
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Upload housing data**")
        upload = st.file_uploader("Upload housing data", type=["csv"], label_visibility="collapsed", key="landing_upload")
    with right:
        st.markdown("**Or start with a ready-made dataset**")
        sample_ok = cfg.SAMPLE_DATA_PATH.exists()
        if ui.button("Explore sample data", key="sample_btn", type="primary", disabled=not sample_ok):
            _load(state.load_sample)
        ui.note(
            "Loads the included Bengaluru house price dataset." if sample_ok
            else f"The sample file data/{cfg.SAMPLE_FILENAME} was not found in this project."
        )

    if upload is not None:
        upload_id = getattr(upload, "file_id", None) or f"{upload.name}:{upload.size}"
        if st.session_state.get("last_upload_id") != upload_id:
            st.session_state["last_upload_id"] = upload_id
            _load(state.load_dataset, upload.getvalue(), upload.name, False)

    error = st.session_state.get("load_error")
    if error is not None:
        ui.error_panel(error)

    ui.md(
        '<div class="hv-steps">'
        '<div class="hv-step"><div class="n">1</div><div class="t">Explore</div>'
        '<div class="d">Understand the property data and identify important patterns.</div></div>'
        '<div class="hv-step"><div class="n">2</div><div class="t">Model</div>'
        '<div class="d">Train and evaluate a Linear Regression model.</div></div>'
        '<div class="hv-step"><div class="n">3</div><div class="t">Predict</div>'
        '<div class="d">Enter property characteristics and receive an estimated value.</div></div></div>'
    )
    ui.legal_footer()


def workspace(ds: state.LoadedDataset) -> None:
    top_left, top_right = st.columns([5, 1])
    with top_left:
        ui.brand()
    with top_right:
        ui.button("Change dataset", key="change_ds", on_click=state.clear_dataset)

    s = ds.prep.summary
    ui.md(
        '<div class="hv-context">'
        f'<span class="f">{ui.esc(ds.filename)}</span>'
        f'<span class="m">{s["n_raw"]:,} properties · {s["n_columns"]} original variables</span>'
        "<span class=\"m\">Target: Price</span>"
        f'<span class="m">Data quality: {ui.esc(QUALITY_BADGE[ds.prep.quality_status])}</span></div>'
    )

    page = ui.segmented("Navigate", PAGES, "nav_page")
    try:
        RENDERERS[page](ds)
    except HomeValueError as err:
        ui.error_panel(err)
    except Exception:
        logger.exception("Unexpected error on page %s", page)
        ui.callout(
            "error",
            "Something went wrong while building this page",
            "Your data is safe and nothing was changed. Try another page, reload the app, or load the dataset again.",
        )
        if st.query_params.get("debug") == "1":
            import traceback
            st.code(traceback.format_exc())
    ui.legal_footer()


def main() -> None:
    state.init_state()
    ui.inject_css()
    dataset = state.get_dataset()
    if dataset is None:
        landing()
    else:
        workspace(dataset)


main()
