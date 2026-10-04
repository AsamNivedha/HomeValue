"""Central configuration and constants for HomeValue."""
from __future__ import annotations

from pathlib import Path

APP_NAME = "HomeValue"
TAGLINE = "Estimate a home's value from the characteristics that matter."

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_FILENAME = "Bengaluru_House_Data.csv"
SAMPLE_DATA_PATH = ROOT_DIR / "data" / SAMPLE_FILENAME

# ---------------------------------------------------------------- schema
TARGET = "price"
PRICE_UNIT = "₹ lakh"
AREA_UNIT = "sq ft"
LAKH = 100_000  # rupees in one lakh

REQUIRED_COLUMNS: dict[str, str] = {
    "price": "Listed price in ₹ lakh (the value HomeValue learns to estimate)",
    "total_sqft": "Total area; plain numbers or ranges such as '2100 - 2850'",
    "size": "Bedroom description such as '2 BHK' or '4 Bedroom'",
    "location": "Locality or neighbourhood name",
    "bath": "Number of bathrooms",
    "balcony": "Number of balconies",
    "area_type": "Type of area measurement, e.g. 'Super built-up Area'",
    "availability": "'Ready To Move' or a future possession date",
}
OPTIONAL_COLUMNS: dict[str, str] = {
    "society": "Housing society name (kept for exploration only)",
}

# Alternative spellings that are mapped onto the expected column names.
COLUMN_ALIASES: dict[str, str] = {
    "total_area": "total_sqft",
    "totalsqft": "total_sqft",
    "area_sqft": "total_sqft",
    "sqft": "total_sqft",
    "price_lakh": "price",
    "price_in_lakh": "price",
    "bathrooms": "bath",
    "balconies": "balcony",
    "bhk": "size",
    "bedrooms": "size",
    "locality": "location",
}

# ---------------------------------------------------------------- validation
MIN_RAW_ROWS = 50
MIN_MODEL_ROWS = 200
MIN_PARSE_SHARE = 0.5  # a column must be at least this usable to continue

# ---------------------------------------------------------------- cleaning rules
MIN_SQFT_PER_BHK = 300      # fewer sq ft per bedroom is treated as implausible
MAX_EXTRA_BATHROOMS = 2     # more than BHK + 2 bathrooms is treated as implausible
PPS_TRIM = (0.01, 0.99)     # price-per-sq-ft percentiles kept for training
MIN_LOCATION_COUNT = 10     # rarer locations are grouped into "Other"
INSIGHT_MIN_GROUP = 30      # minimum group size for comparisons in insights

# ---------------------------------------------------------------- model
NUMERIC_FEATURES = ["bhk", "total_sqft", "bath", "balcony"]
CATEGORICAL_FEATURES = ["location", "area_type", "availability_group"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
FEATURE_LABELS = {
    "bhk": "BHK",
    "total_sqft": "Total area",
    "bath": "Bathrooms",
    "balcony": "Balconies",
    "location": "Location",
    "area_type": "Area type",
    "availability_group": "Availability",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_VERSION = "1"
OTHER_LOCATION = "Other / not listed"

DISCLAIMER = (
    "HomeValue provides model-based estimates derived from the supplied historical "
    "dataset. Estimates are not professional property valuations or guarantees of "
    "market price."
)
