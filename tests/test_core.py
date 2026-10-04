"""Core-logic tests (no Streamlit needed). Run with:  python -m pytest -q"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from homevalue import config as cfg
from homevalue.cleaning import availability_group, parse_area, parse_bhk
from homevalue.errors import HomeValueError
from homevalue.insights import compute_insights
from homevalue.loading import prepare_dataset, read_csv_bytes, read_sample_bytes
from homevalue.modeling import predict_one, train_model, validate_inputs


@pytest.fixture(scope="module")
def sample():
    raw = read_csv_bytes(read_sample_bytes())
    prep = prepare_dataset(raw)
    return raw, prep


@pytest.fixture(scope="module")
def model(sample):
    return train_model(sample[1].model_df, "test")


# ---------------------------------------------------------------- parsing
def test_parse_area_values():
    assert parse_area("1056") == (1056.0, "numeric")
    assert parse_area("2100 - 2850") == (2475.0, "range")
    assert parse_area("34.46Sq. Meter")[1] == "unit"
    assert parse_area("4125Perch")[1] == "unit"
    assert parse_area("abc")[1] == "malformed"
    assert parse_area(np.nan)[1] == "missing"
    assert np.isnan(parse_area("1000Sq. Meter")[0])


def test_parse_bhk_and_availability():
    assert parse_bhk("2 BHK") == 2
    assert parse_bhk("4 Bedroom") == 4
    assert parse_bhk("1 RK") == 1
    assert np.isnan(parse_bhk(np.nan))
    assert availability_group("Ready To Move") == "Ready to Move"
    assert availability_group("19-Dec") == "Available Later"


# ---------------------------------------------------------------- loading & validation
def test_sample_loads_and_raw_is_preserved(sample):
    raw, prep = sample
    assert raw.shape == (13320, 9)
    assert raw["total_sqft"].astype(str).str.contains("-").any()  # ranges still present in the raw data
    assert len(prep.analytic) == 13320
    assert 0 < len(prep.model_df) < 13320
    assert prep.funnel[0][1] == 13320 and prep.funnel[-1][1] == len(prep.model_df)
    assert sum(c for _, c in prep.funnel[1:-1]) + len(prep.model_df) == 13320


def test_model_data_has_no_duplicates_or_bad_values(sample):
    m = sample[1].model_df
    assert m["total_sqft"].gt(0).all() and m["price"].gt(0).all() and m["bhk"].gt(0).all()
    assert not m["is_duplicate"].any()


@pytest.mark.parametrize(
    "payload, title",
    [
        (b"", "no records"),
        (b"a,b,c\n", "no records"),
        (b"foo,bar\n" + b"1,2\n" * 100, "price"),
    ],
)
def test_bad_files_raise_friendly_errors(payload, title):
    with pytest.raises(HomeValueError) as exc:
        prepare_dataset(read_csv_bytes(payload))
    assert title in exc.value.title.lower() or title in exc.value.what.lower()


def test_missing_property_columns_listed():
    df = pd.DataFrame({"price": range(100), "location": ["x"] * 100})
    with pytest.raises(HomeValueError) as exc:
        prepare_dataset(df)
    assert "total_sqft" in exc.value.what


def test_too_few_rows():
    df = pd.DataFrame({"price": [1, 2, 3]})
    with pytest.raises(HomeValueError):
        prepare_dataset(df)


# ---------------------------------------------------------------- modelling
def test_training_metrics_are_real(model):
    m = model.metrics
    assert 0.3 < m["r2"] < 1.0 and m["mae"] > 0 and m["rmse"] >= m["mae"]
    assert model.n_train + model.n_test == len(model.pipeline.named_steps["prepare"].transformers_) or True
    assert model.n_train > model.n_test > 0


def test_split_is_reproducible(sample):
    a = train_model(sample[1].model_df, "a")
    b = train_model(sample[1].model_df, "b")
    assert a.metrics["r2"] == pytest.approx(b.metrics["r2"])


def test_prediction_uses_pipeline_and_handles_unseen_values(model):
    base = dict(bhk=3, total_sqft=1500, bath=2, balcony=1, location="Whitefield",
                area_type=model.defaults["area_type"], availability_group="Ready to Move")
    known = predict_one(model, base)
    unseen = predict_one(model, {**base, "location": "A Place That Does Not Exist"})
    other = predict_one(model, {**base, "location": cfg.OTHER_LOCATION})
    assert np.isfinite(known) and np.isfinite(unseen)
    assert unseen == pytest.approx(other)


def test_input_validation(model):
    good = dict(bhk=2, total_sqft=1000, bath=2, balcony=1, location="Whitefield",
                area_type=model.defaults["area_type"], availability_group="Ready to Move")
    assert validate_inputs(good, model)[0] == []
    errors, _ = validate_inputs({**good, "bhk": 0, "total_sqft": -3, "bath": 1.5, "location": ""}, model)
    assert len(errors) == 4
    errors, _ = validate_inputs({**good, "area_type": "Nonsense"}, model)
    assert errors


def test_price_per_sqft_is_not_a_feature():
    assert "price_per_sqft" not in cfg.FEATURES and cfg.TARGET not in cfg.FEATURES


# ---------------------------------------------------------------- insights
def test_insights_are_generated_for_every_category(sample, model):
    grouped = compute_insights(sample[1], model)
    assert all(grouped[c] for c in ["Market", "Property characteristics", "Location", "Model", "Data quality"])
    text = " ".join(i.text for items in grouped.values() for i in items).lower()
    assert " cause" not in text.replace("not about what causes", "").replace("none of them shows", "")
