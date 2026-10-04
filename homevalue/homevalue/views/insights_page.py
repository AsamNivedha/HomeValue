"""Insights: what is the data telling me?"""
from __future__ import annotations

import json

import streamlit as st

from .. import ui
from ..insights import CATEGORIES, compute_insights, insights_markdown
from ..state import LoadedDataset, safe_get_model

BLURBS = {
    "Market": "Observed property-price patterns.",
    "Property characteristics": "Relationships involving area, BHK, bathrooms and more.",
    "Location": "Observed differences across locations.",
    "Model": "What the evaluation on held-out test properties indicates.",
    "Data quality": "Limitations and irregularities discovered in the dataset.",
}


def render(ds: LoadedDataset) -> None:
    ui.section("Insights", "What is the data telling me?")
    ui.note(
        "Every finding below is computed from the active dataset and describes associations observed in this data. "
        "None of them shows that one factor causes another, and none describes the wider property market."
    )
    model, err = safe_get_model()
    grouped = compute_insights(ds.prep, model)
    if err is not None:
        ui.callout("attention", "Model insights are unavailable", err.what)

    for category in CATEGORIES:
        items = grouped.get(category, [])
        if not items:
            continue
        ui.section(category, BLURBS[category])
        ui.bullets([i.text for i in items])

    ui.section("Export")
    payload = {c: [i.text for i in items] for c, items in grouped.items() if items}
    e1, e2 = st.columns(2)
    with e1:
        ui.download("Download insights (Markdown)", insights_markdown(grouped, ds.filename).encode("utf-8"),
                    "homevalue_insights.md", "text/markdown", key="dl_ins_md")
    with e2:
        ui.download("Download insights (JSON)", json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8"),
                    "homevalue_insights.json", "application/json", key="dl_ins_json")
    ui.note(
        "Charts: use the camera icon in any chart's toolbar to save it as an image. Filtered data, model metrics and "
        "prediction history can be downloaded from the Explore, Model and Predict pages."
    )
