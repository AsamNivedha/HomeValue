"""CSV reading, column mapping and dataset-level validation."""
from __future__ import annotations

import io
import re

import pandas as pd

from . import config as cfg
from .cleaning import PreparedData, prepare
from .errors import HomeValueError


def _expected_lines() -> list[str]:
    lines = [f"{name} — {desc}" for name, desc in cfg.REQUIRED_COLUMNS.items()]
    lines += [f"{name} — {desc} (optional)" for name, desc in cfg.OPTIONAL_COLUMNS.items()]
    return lines


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse CSV bytes into a DataFrame, raising HomeValueError on any problem."""
    empty = HomeValueError(
        "This file has no records to analyze",
        "The CSV is empty, or it contains column headings but no property rows.",
        _expected_lines(),
        "Upload a CSV that contains one row per property, or explore the sample data.",
    )
    if not data or not data.strip():
        raise empty
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(data), encoding=encoding)
            break
        except pd.errors.EmptyDataError:
            raise empty
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # parser errors, bad quoting, etc.
            raise HomeValueError(
                "The file could not be read as a CSV",
                "HomeValue could not parse this file. It may not be a comma-separated text file.",
                _expected_lines(),
                "Check that the file is a plain CSV (not an Excel workbook) and upload it again.",
            ) from exc
    if df is None:
        raise HomeValueError(
            "The file could not be read as a CSV",
            "The text encoding of the file is not supported.",
            _expected_lines(),
            "Save the file as UTF-8 CSV and upload it again.",
        )
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise empty
    return df


def map_columns(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename recognisable columns to the expected names (the raw frame is untouched)."""
    known = set(cfg.REQUIRED_COLUMNS) | set(cfg.OPTIONAL_COLUMNS)
    used: set[str] = set()
    mapping: dict[str, str] = {}
    names: list[str] = []
    for col in raw.columns:
        norm = _norm(col)
        target = cfg.COLUMN_ALIASES.get(norm, norm)
        if target in known and target not in used:
            used.add(target)
            new = target
        else:
            new = str(col)
        base, i = new, 1
        while new in names:
            i += 1
            new = f"{base}_{i}"
        names.append(new)
        if new != str(col):
            mapping[str(col)] = new
    working = raw.copy()
    working.columns = names
    return working, mapping


def _check_columns(working: pd.DataFrame) -> None:
    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in working.columns]
    if cfg.TARGET in missing:
        raise HomeValueError(
            "HomeValue cannot train without a price column",
            "No column named 'price' was found, so there is no value for the model to learn to estimate.",
            _expected_lines(),
            "Add a numeric 'price' column (in ₹ lakh) to your CSV, or explore the sample data.",
        )
    if missing:
        raise HomeValueError(
            "Required property information is unavailable",
            "These columns were not found in the file: " + ", ".join(missing) + ".",
            _expected_lines(),
            "Add the missing columns (or rename existing ones to match) and upload the file again.",
        )


def prepare_dataset(raw: pd.DataFrame) -> PreparedData:
    """Map columns, validate and prepare a freshly loaded dataset."""
    if len(raw) < cfg.MIN_RAW_ROWS:
        raise HomeValueError(
            "The file has too few records",
            f"The file contains {len(raw)} rows; at least {cfg.MIN_RAW_ROWS} are needed to describe a housing market.",
            _expected_lines(),
            "Upload a larger housing dataset, or explore the sample data.",
        )
    working, mapping = map_columns(raw)
    _check_columns(working)
    prep = prepare(working, mapping)
    s = prep.summary
    if s["price_numeric_share"] < cfg.MIN_PARSE_SHARE:
        raise HomeValueError(
            "The price column is not usable",
            "Most values in the 'price' column are not numbers, so HomeValue cannot learn from them.",
            _expected_lines(),
            "Make sure 'price' contains plain numeric values (in ₹ lakh).",
        )
    if s["area_parse_share"] < cfg.MIN_PARSE_SHARE:
        raise HomeValueError(
            "Area values cannot be interpreted",
            "Most values in 'total_sqft' are not numbers or ranges such as '2100 - 2850'.",
            _expected_lines(),
            "Provide total area in square feet, as a number or a range.",
        )
    if s["bhk_share"] < cfg.MIN_PARSE_SHARE:
        raise HomeValueError(
            "Bedroom information cannot be read",
            "Most values in 'size' do not start with a number (for example '2 BHK' or '4 Bedroom').",
            _expected_lines(),
            "Provide the bedroom count at the start of each 'size' value.",
        )
    if s["location_share"] < cfg.MIN_PARSE_SHARE:
        raise HomeValueError(
            "Location information is unusable",
            "Most records have no location, so location-based estimates are not possible.",
            _expected_lines(),
            "Fill in the 'location' column and upload the file again.",
        )
    if s["n_model"] < cfg.MIN_MODEL_ROWS:
        raise HomeValueError(
            "Too few usable records to train a model",
            f"Only {s['n_model']} records remain after duplicates and unusable values are handled; "
            f"at least {cfg.MIN_MODEL_ROWS} are needed.",
            _expected_lines(),
            "Add more complete property records, or explore the sample data.",
        )
    return prep


def read_sample_bytes() -> bytes:
    if not cfg.SAMPLE_DATA_PATH.exists():
        raise HomeValueError(
            "The sample dataset could not be found",
            f"The file data/{cfg.SAMPLE_FILENAME} is missing from the project.",
            [],
            f"Add {cfg.SAMPLE_FILENAME} to the data folder, or upload your own CSV.",
        )
    return cfg.SAMPLE_DATA_PATH.read_bytes()
