# HomeValue

**Estimate a home's value from the characteristics that matter.**

HomeValue is a property analytics and price-estimation app built with Streamlit. It takes a housing dataset,
helps you understand and clean it, trains and evaluates a Linear Regression model, lets you describe a property,
and returns an explainable estimate.

> Data → Understanding → Model → Prediction → Explanation → Export

HomeValue is an academic / portfolio project. It is **not** an official valuation service (see [Limitations](#limitations)).

## Features

| Page | Question it answers | What it offers |
|------|--------------------|----------------|
| **Overview** | What am I looking at? | A handful of key metrics, a plain-language data-quality summary, the path from original to model-ready records, computed observations |
| **Explore** | What is in the data? | Searchable, filterable, sortable record table with column visibility and CSV export; column inspector; Data quality center (issue → impact → action); interactive Plotly charts (price distribution, price vs area, price by BHK, location comparison, price per sq ft, correlation) |
| **Model** | How well does it work and what did it learn? | Model configuration, R² / MAE / RMSE, actual vs predicted, four error-analysis views, readable coefficients, metric and prediction exports |
| **Predict** | What is this property worth? | Grouped input form with searchable location, validation, a prominent estimate with model context, session history |
| **Insights** | What is the data telling me? | Market, property, location, model and data-quality findings generated from the active dataset, exportable as Markdown / JSON |

The landing screen offers **Upload housing data** or **Explore sample data** (the bundled Bengaluru dataset), so the whole
product can be demonstrated immediately.

## Workflow

1. **Data** – load a CSV (or the sample). It is validated before the workspace opens; problems are explained in plain language.
2. **Understanding** – Overview and Explore describe the records and their quality. Filtering never changes training data.
3. **Model** – Linear Regression is trained once per dataset and evaluated on a held-out test set.
4. **Prediction** – the same fitted pipeline estimates a property from the values you enter.
5. **Explanation** – each estimate is shown with the property summary, the model's test-set MAE and what that number does and does not mean.
6. **Export** – filtered data, model metrics, test predictions, prediction history and insights, each from the page where it belongs.

## Dataset

`data/Bengaluru_House_Data.csv` – 13,320 listings, 9 columns: `area_type`, `availability`, `location`, `size`, `society`,
`total_sqft`, `bath`, `balcony`, `price` (the target, in ₹ lakh).

Known limitations of the data: it is a historical snapshot of *listed* prices (not transaction prices), many locations have very few
listings, `society` is missing for about 41% of records, area is sometimes a range or quoted in other units, and a number of
records are duplicates or have implausible values.

## Methodology

**Data states are kept separate.** The uploaded file is never modified. HomeValue derives (1) an *analytic* table that keeps every
record plus parsed columns and quality flags, and (2) a *model-ready* table of records that are safe to learn from.

**Cleaning and feature engineering**
- `size` → **BHK** (`2 BHK` → 2, `4 Bedroom` → 4).
- `total_sqft` → numeric area. Plain numbers are used as is, ranges (`2100 - 2850`) use the midpoint (`2475`). Values in other units
  (sq. metres, perches, …) or unreadable text are *not* silently converted; those records are excluded and reported.
- `availability` → **Ready to Move** or **Available Later**.
- Exact duplicates are removed (first occurrence kept).
- Records are excluded when the price is missing or non-positive, the area is unusable, the BHK is unavailable, the configuration is
  implausible (under 300 sq ft per bedroom, or more than 2 bathrooms beyond the bedroom count), or the price per sq ft lies outside the
  1st–99th percentile (typically data-entry errors). Every exclusion is counted and viewable in the Data quality center.
- `society` is retained for exploration only. `price_per_sqft` is used for analysis and for the outlier rule above, **never as a model input**.

**Model inputs:** BHK, total area, bathrooms, balconies, location, area type, availability.

**Pipeline** (one scikit-learn `Pipeline`, fitted on the training split only):

```
Feature frame → ColumnTransformer
                 ├─ numeric:     median imputation
                 ├─ location:    imputation → one-hot (locations with < 10 training listings and unseen locations share "Other")
                 └─ categorical: most-frequent imputation → one-hot (unseen values ignored safely)
              → LinearRegression
```

An 80/20 train/test split with a fixed random state makes results reproducible. Training happens once per dataset (cached);
changing pages, filters or charts, and making predictions, never retrain.

## Machine learning

Linear Regression is the only predictive algorithm. It is fast, transparent and easy to explain: each coefficient is the model's learned
relationship with price while accounting for the other included features. HomeValue deliberately does not call raw coefficients "feature
importance", and it describes location effects relative to the average location rather than as standalone facts. A linear model can
extrapolate poorly at the edges of the data (it can even produce a value below zero for very small properties; HomeValue then declines
to show an estimate).

## Evaluation

All metrics are computed on the held-out test set:

- **R²** – how much of the variation in observed prices the model explains.
- **MAE** – the average absolute difference between predicted and observed prices (₹ lakh).
- **RMSE** – like MAE, but larger errors count more.

MAE is a *typical error across the test set*; it is **not** a confidence interval or prediction interval for any single estimate.
On the bundled dataset a run produced R² ≈ 0.72 and MAE ≈ ₹30 lakh (about 44% of the median test price); exact values depend on library
versions and are always computed live in the app.

## Project structure

```
app.py                      Streamlit entry point: landing state, navigation, error boundary
requirements.txt            Dependencies
.streamlit/config.toml      Light theme and server settings
data/Bengaluru_House_Data.csv   Sample dataset (loaded by relative path)
homevalue/
  config.py                 Constants: schema, thresholds, feature lists, disclaimer
  errors.py                 HomeValueError (user-facing, structured)
  loading.py                CSV reading, column mapping, dataset validation
  cleaning.py               Parsing, feature engineering, exclusions, data-quality report
  modeling.py               Pipeline, training, evaluation, coefficients, input validation, prediction
  insights.py               Dynamically generated observations and chart interpretations
  charts.py                 Plotly figures
  formatting.py             Number / currency formatting
  state.py                  Session state, caching, dataset lifecycle
  ui.py                     CSS, components and Streamlit width-compat wrappers
  views/                    overview.py, explore.py, model_page.py, predict.py, insights_page.py
tests/test_core.py          Tests for parsing, validation, training, prediction and insights
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Local execution

```bash
streamlit run app.py
```

Optional tests (install `pytest` first): `python -m pytest -q`

## Deployment (Streamlit Community Cloud)

1. Push this folder to a GitHub repository (keep `app.py`, `requirements.txt`, `data/` and `homevalue/` at the repository root).
2. Open [share.streamlit.io](https://share.streamlit.io), choose **New app** and select the repository and branch.
3. Set the main file path to `app.py` and deploy.

No API keys or secrets are required, and all paths are relative.

## Limitations

HomeValue provides model-based estimates derived from the supplied historical dataset. Estimates are not professional property
valuations or guarantees of market price. The model only reflects the listings it was trained on, only uses seven property
characteristics, and findings in the app describe associations in this dataset rather than causes or the wider market.
