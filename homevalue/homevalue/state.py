"""Session state, caching and dataset lifecycle.

Conceptual state kept for the whole session:
  dataset   - original raw frame + prepared data (analytic / model-ready / quality report)
  model     - trained Linear Regression pipeline and its evaluation
  history   - lightweight list of predictions made this session
  explore   - widget keys prefixed ``ex_`` (filters, selections)
  predict   - widget keys prefixed ``pr_`` and the last prediction
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from . import config as cfg
from .cleaning import PreparedData
from .errors import HomeValueError
from .loading import prepare_dataset, read_csv_bytes, read_sample_bytes
from .modeling import TrainedModel, train_model

PERSIST_PREFIXES = ("ex_", "pr_", "nav_")


@dataclass
class LoadedDataset:
    dataset_id: str
    filename: str
    raw: pd.DataFrame          # exactly what was uploaded; never modified
    prep: PreparedData         # analytic + model-ready views and quality report
    is_sample: bool = False


# ------------------------------------------------------------------ cached heavy work
@st.cache_resource(show_spinner=False)
def _prepare_cached(dataset_id: str, _raw: pd.DataFrame) -> PreparedData:
    return prepare_dataset(_raw)


@st.cache_resource(show_spinner=False)
def _train_cached(model_key: str, _model_df: pd.DataFrame) -> TrainedModel:
    return train_model(_model_df, model_key)


# ------------------------------------------------------------------ lifecycle
def init_state() -> None:
    defaults = {
        "dataset": None,
        "model": None,
        "model_key": None,
        "history": [],
        "last_prediction": None,
        "load_error": None,
        "last_upload_id": None,
        "nav_page": "Overview",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    persist_widget_state()


def persist_widget_state() -> None:
    """Keep widget values alive while their page is not on screen."""
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(PERSIST_PREFIXES):
            st.session_state[key] = st.session_state[key]


def reset_widget_state() -> None:
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(("ex_", "pr_", "_last_ex_", "_last_pr_")):
            del st.session_state[key]


def get_dataset() -> LoadedDataset | None:
    return st.session_state.get("dataset")


def load_dataset(data: bytes, filename: str, is_sample: bool = False) -> None:
    """Parse, validate and activate a dataset. Raises HomeValueError on failure."""
    raw = read_csv_bytes(data)
    dataset_id = hashlib.sha256(data).hexdigest()[:16]
    prep = _prepare_cached(dataset_id, raw)
    reset_widget_state()
    st.session_state.update(
        dataset=LoadedDataset(dataset_id, filename, raw, prep, is_sample),
        model=None,
        model_key=None,
        history=[],
        last_prediction=None,
        load_error=None,
        nav_page="Overview",
    )


def load_sample() -> None:
    load_dataset(read_sample_bytes(), cfg.SAMPLE_FILENAME, is_sample=True)


def clear_dataset() -> None:
    reset_widget_state()
    st.session_state.update(
        dataset=None, model=None, model_key=None, history=[], last_prediction=None, load_error=None, nav_page="Overview"
    )


def get_model() -> TrainedModel:
    """Return the trained model, training it only if the dataset requires it."""
    dataset = get_dataset()
    key = f"{dataset.dataset_id}:{cfg.MODEL_VERSION}"
    if st.session_state.get("model_key") == key and st.session_state.get("model") is not None:
        return st.session_state["model"]
    with st.spinner("Training the Linear Regression model…"):
        model = _train_cached(key, dataset.prep.model_df)
    st.session_state["model"] = model
    st.session_state["model_key"] = key
    return model


def add_history(record: dict) -> None:
    st.session_state["history"] = st.session_state["history"] + [record]


def clear_history() -> None:
    st.session_state["history"] = []
    st.session_state["last_prediction"] = None


def safe_get_model() -> tuple[TrainedModel | None, HomeValueError | None]:
    """Like get_model(), but turns every failure into a displayable HomeValueError."""
    try:
        return get_model(), None
    except HomeValueError as err:
        return None, err
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Model training failed")
        return None, HomeValueError(
            "The model could not be trained",
            "Something unexpected happened while training Linear Regression on this dataset.",
            [],
            "Review the Data quality view for problems, then reload the page or load a different dataset.",
        )
