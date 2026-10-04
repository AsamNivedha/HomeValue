"""Linear Regression pipeline: training, evaluation, coefficients and prediction.

The same fitted ``Pipeline`` object (preprocessing + estimator) is used for the
evaluation on the test set and for every prediction made on the Predict page.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config as cfg
from .errors import HomeValueError


# ------------------------------------------------------------------ pipeline
def build_pipeline(min_location_count: int = cfg.MIN_LOCATION_COUNT) -> Pipeline:
    """Feature preparation -> ColumnTransformer -> Linear Regression."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    )
    location = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_location_count,
                    sparse_output=False,
                ),
            ),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("num", numeric, cfg.NUMERIC_FEATURES),
            ("loc", location, ["location"]),
            ("cat", categorical, ["area_type", "availability_group"]),
        ]
    )
    return Pipeline([("prepare", preprocess), ("regressor", LinearRegression())])


def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Select the model's feature schema with consistent dtypes."""
    X = df.reindex(columns=cfg.FEATURES).copy()
    for col in cfg.NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    for col in cfg.CATEGORICAL_FEATURES:
        series = X[col]
        X[col] = series.astype(object).where(series.notna(), np.nan)
    return X


# ------------------------------------------------------------------ container
@dataclass
class TrainedModel:
    dataset_id: str
    pipeline: Pipeline
    metrics: dict
    test_frame: pd.DataFrame
    coefficients: pd.DataFrame
    n_train: int
    n_test: int
    n_encoded_features: int
    location_counts: pd.Series
    kept_locations: list[str]
    locations: list[str]
    area_types: list[str]
    availability_options: list[str]
    defaults: dict
    ranges: dict = field(default_factory=dict)


# ------------------------------------------------------------------ training
def train_model(model_df: pd.DataFrame, dataset_id: str) -> TrainedModel:
    """Train and evaluate the Linear Regression pipeline."""
    if len(model_df) < cfg.MIN_MODEL_ROWS:
        raise HomeValueError(
            "The model could not be trained",
            f"Only {len(model_df)} usable records are available; at least {cfg.MIN_MODEL_ROWS} are needed.",
            [],
            "Provide a larger dataset with more complete property records.",
        )
    try:
        X = to_feature_frame(model_df)
        y = model_df[cfg.TARGET].astype(float)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_STATE
        )
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        pred_test = pipeline.predict(X_test)
        pred_train = pipeline.predict(X_train)
    except Exception as exc:  # any numerical / data failure becomes a friendly error
        raise HomeValueError(
            "The model could not be trained",
            "Linear Regression failed on this dataset. The prepared data may contain values the model cannot use.",
            [],
            "Check the Data quality view for issues, fix the CSV and upload it again.",
        ) from exc

    rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    metrics = {
        "r2": float(r2_score(y_test, pred_test)),
        "mae": float(mean_absolute_error(y_test, pred_test)),
        "rmse": rmse,
        "median_ae": float(np.median(np.abs(pred_test - y_test.to_numpy()))),
        "train_r2": float(r2_score(y_train, pred_train)),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "median_price_test": float(y_test.median()),
    }

    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["predicted"] = pred_test
    test_frame["error"] = test_frame["predicted"] - test_frame["actual"]
    test_frame["abs_error"] = test_frame["error"].abs()
    test_frame["pct_error"] = test_frame["error"] / test_frame["actual"]

    location_counts = X_train["location"].fillna("Unknown").value_counts()
    kept = sorted(location_counts[location_counts >= cfg.MIN_LOCATION_COUNT].index.tolist())
    all_locations = (
        model_df["location"].dropna().value_counts().index.tolist()
    )
    availability_options = ["Ready to Move", "Available Later"]
    area_types = sorted(model_df["area_type"].dropna().unique().tolist())

    defaults = {
        "bhk": int(model_df["bhk"].median()),
        "total_sqft": int(round(model_df["total_sqft"].median())),
        "bath": int(model_df["bath"].median()) if model_df["bath"].notna().any() else 2,
        "balcony": int(round(model_df["balcony"].median())) if model_df["balcony"].notna().any() else 1,
        "location": all_locations[0] if all_locations else cfg.OTHER_LOCATION,
        "area_type": model_df["area_type"].mode().iloc[0] if len(area_types) else "",
        "availability_group": (
            model_df["availability_group"].mode().iloc[0]
            if model_df["availability_group"].notna().any()
            else "Ready to Move"
        ),
    }
    ranges = {
        "bhk": (float(model_df["bhk"].min()), float(model_df["bhk"].max())),
        "total_sqft": (float(model_df["total_sqft"].min()), float(model_df["total_sqft"].max())),
        "bath": (float(model_df["bath"].min()), float(model_df["bath"].max())),
    }
    pre = pipeline.named_steps["prepare"]
    return TrainedModel(
        dataset_id=dataset_id,
        pipeline=pipeline,
        metrics=metrics,
        test_frame=test_frame,
        coefficients=coefficient_table(pipeline),
        n_train=len(X_train),
        n_test=len(X_test),
        n_encoded_features=len(pre.get_feature_names_out()),
        location_counts=location_counts,
        kept_locations=kept,
        locations=all_locations,
        area_types=area_types,
        availability_options=availability_options,
        defaults=defaults,
        ranges=ranges,
    )


# ------------------------------------------------------------------ coefficients
def coefficient_table(pipeline: Pipeline) -> pd.DataFrame:
    """One row per encoded feature with a readable label.

    ``relative_effect`` is the coefficient minus the average coefficient of its
    feature group for categorical features (so location effects are read as
    differences between locations); for numeric features it equals the coefficient.
    """
    names = pipeline.named_steps["prepare"].get_feature_names_out()
    coefs = pipeline.named_steps["regressor"].coef_
    rows = []
    for name, coef in zip(names, coefs):
        prefix, _, rest = name.partition("__")
        if prefix == "num":
            group, label = rest, rest
        elif prefix == "loc":
            group = "location"
            label = rest[len("location_"):] if rest.startswith("location_") else rest
            if label == "infrequent_sklearn":
                label = "Other (rarely listed locations)"
        else:
            group = "area_type" if rest.startswith("area_type_") else "availability_group"
            label = rest[len(group) + 1:]
        rows.append({"group": group, "label": label, "coefficient": float(coef)})
    table = pd.DataFrame(rows)
    table["relative_effect"] = table["coefficient"]
    categorical = table["group"].isin(["location", "area_type", "availability_group"])
    centered = table.loc[categorical].groupby("group")["coefficient"].transform(lambda s: s - s.mean())
    table.loc[categorical, "relative_effect"] = centered
    return table


# ------------------------------------------------------------------ prediction
def validate_inputs(inputs: dict, model: TrainedModel) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for a set of user-entered property values."""
    errors: list[str] = []
    warnings: list[str] = []

    def number(key):
        value = inputs.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    bhk = number("bhk")
    if bhk is None or bhk < 1 or bhk != int(bhk) or bhk > 30:
        errors.append("BHK must be a whole number between 1 and 30.")
    area = number("total_sqft")
    if area is None or area <= 0:
        errors.append("Total area must be a positive number of square feet.")
    elif area > 200_000:
        errors.append("Total area looks too large to be a single property (maximum 200,000 sq ft).")
    bath = number("bath")
    if bath is None or bath < 1 or bath != int(bath) or bath > 30:
        errors.append("Bathrooms must be a whole number between 1 and 30.")
    balcony = number("balcony")
    if balcony is None or balcony < 0 or balcony != int(balcony) or balcony > 20:
        errors.append("Balconies must be a whole number between 0 and 20 (use 0 for none).")

    location = inputs.get("location")
    if not location or not isinstance(location, str):
        errors.append("Choose a location (or 'Other / not listed').")
    if inputs.get("area_type") not in model.area_types:
        errors.append("Choose an area type from the list.")
    if inputs.get("availability_group") not in model.availability_options:
        errors.append("Choose whether the property is Ready to Move or Available Later.")

    if not errors:
        lo, hi = model.ranges["total_sqft"]
        if area < lo or area > hi:
            warnings.append(
                f"The area is outside the range of properties the model was trained on "
                f"({lo:,.0f}–{hi:,.0f} sq ft), so the estimate is less reliable."
            )
        blo, bhi = model.ranges["bhk"]
        if bhk < blo or bhk > bhi:
            warnings.append("This BHK value is outside the range seen in the training data.")
        if bath > bhk + cfg.MAX_EXTRA_BATHROOMS:
            warnings.append("This property has many more bathrooms than bedrooms, which is unusual in the training data.")
        if location == cfg.OTHER_LOCATION or location not in model.locations:
            warnings.append(
                "This location is not in the dataset, so it is treated like other rarely listed locations."
            )
        elif location not in model.kept_locations:
            warnings.append(
                "This location has few listings in the training data, so it is grouped with other rarely listed "
                "locations and the estimate does not reflect location-specific pricing."
            )
    return errors, warnings


def predict_one(model: TrainedModel, inputs: dict) -> float:
    """Predict a single property price (₹ lakh) with the already-trained pipeline."""
    location = inputs.get("location")
    row = {
        "bhk": inputs["bhk"],
        "total_sqft": inputs["total_sqft"],
        "bath": inputs["bath"],
        "balcony": inputs["balcony"],
        "location": np.nan if (not location or location == cfg.OTHER_LOCATION) else location,
        "area_type": inputs["area_type"],
        "availability_group": inputs["availability_group"],
    }
    frame = to_feature_frame(pd.DataFrame([row]))
    return float(model.pipeline.predict(frame)[0])
