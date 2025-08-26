# SHIPP Integration — EPFToolbox LEAR 48-Hour Price Forecast

This folder integrates **electricity price forecasting** into the SHIPP workflow by using the **LEAR** model from [`epftoolbox`](https://github.com/jeslago/epftoolbox). It includes:

- a data retriever for ENTSO-E day-ahead prices and system forecasts,
- a 48-hour rolling forecast pipeline that recalibrates LEAR daily and produces **mean** and **P10/P90** bands,
- a small test harness / notebook that demonstrates how SHIPP can call the forecaster and visualize outputs.

> **Placement:** this folder—named **`SHIPP Integration`**—was created to be **dropped into the `examples/` directory** of the `epftoolbox` repository. After you download/clone `epftoolbox` (as described in its documentation), copy or move this folder to:
>
> ```
> epftoolbox/
> └─ examples/
>    └─ SHIPP Integration/
>       ...
> ```
>
> Then work from inside `epftoolbox/examples/SHIPP Integration/` when running the steps below.

---

## Contents

```
SHIPP Integration/
├─ entsoe_price_data.py        # Fetch & assemble ENTSO-E data to CSV (prices, load, wind/solar)
├─ lear_48hrs.py               # 48-hour LEAR forecast + rolling P10/P90 computation
├─ price_forecast.py (or .ipynb)
│                              # Test harness: shows how SHIPP calls the forecaster;
│                              # includes plotting (matplotlib + optional Plotly)
├─ datasets/                   # Created automatically: input CSVs for epftoolbox
└─ experimental_files/         # Created automatically: D+1 and 48h forecast snapshots
```

### What each script does

- **`entsoe_price_data.py`**  
  Downloads **day-ahead prices**, **grid load forecast**, and **wind & solar forecast** from ENTSO-E for a padded date range (calibration window + rolling window + 1 day), converts to an **hourly** time series, fills DST gaps, and writes a CSV under `../datasets/` named like `NL_01012018_01042018.csv`. Returns the dataset **stem** to feed into `epftoolbox.data.read_data`.

- **`lear_48hrs.py`**  
  Orchestrates a **48-hour rolling forecast** with LEAR. For each issue date it:
  1) **Recalibrates** LEAR and forecasts **D+1 (hours 0–23)**;  
  2) Uses those D+1 results to forecast **D+2 (hours 24–47)**;  
  3) Builds a long table of forecasts vs. actuals to compute **rolling residual quantiles** per lead-time and derive **P10/P90** bands;  
  4) Saves intermediate **D+1** CSVs and the **48-hour** CSVs in `experimental_files/`.  
  Returns four lists (per issue): **mean forecast**, **P10**, **P90**, and **actual prices**.

- **`price_forecast.py` / notebook**  
  A reference notebook that shows how to **replace a dummy AR model** with the LEAR-based pipeline above, how to **read datasets**, run the 48-hour forecast, and **plot** results (including an optional interactive Plotly view via `animated_plot.plot_interactive_forecasts`).

---

## Installation & environment

Use the provided `environment.yml` (recommended) or your own Python 3.10+ environment. Then:

```bash
# from the epftoolbox repo root (one level above examples/)
pip install -e .
```

Inside `examples/SHIPP Integration/`, install any optional extras you want (e.g., Plotly for interactive charts) if they’re not already present:

```bash
pip install plotly
```

---

## Required credentials (ENTSO-E)

You need an **ENTSO-E API key** to download data. Pass it directly to the function or expose it as an environment variable and read it in your launcher script.

---

## Data flow & file naming

1) **ENTSO-E data → `datasets/`**  
   `get_entsoe_price_data(api_key, country_code, calibration_window, rolling_window_days, begin_test_date, end_test_date)`  
   - Pulls prices, load, wind/solar; resamples to **hourly**, handles timezone/DST, forward-fills gaps;  
   - Writes `../datasets/{COUNTRY}_{DDMMYYYY}_{DDMMYYYY}.csv`;  
   - Returns the stem (e.g., `NL_01012018_01042018`) used by `epftoolbox.data.read_data`.

2) **Forecast runs → `experimental_files/`**  
   - For each issue date, step **1** (D+1) is saved as `{COUNTRY}_{start}_{end}_d1.csv`; the combined **48-hour** result is `{COUNTRY}_{start}_{end}_48hrs.csv`.

---

## How the 48-hour LEAR pipeline works

1) **Build training/test windows** from the dataset returned above using `epftoolbox.data.read_data`.  
2) For each **forecast issue time** (every 24h):
   - **Recalibrate LEAR** on a rolling calibration window and **forecast the next 24h** (`D+1`). This uses `model.recalibrate_and_forecast_next_day(...)`. The predicted prices for `D+1` are stored and also written to disk.  
   - **Forecast `D+2` (hours 24–47)** by rebuilding the feature set for the following day and **injecting the D+1 predictions** as the most recent prices. Concatenate D+1 + D+2 for a full **48-hour** vector and save.  
3) **Probabilistic bands (P10/P90)** are obtained by computing **rolling residual quantiles** per lead-time over a user-defined window (e.g., 60 days) and **shifting** them onto the point forecast (`mean ± quantile residual`).  
4) The function returns **lists of lists** (length = number of issues) in SHIPP-friendly format:  
   `forecast_mean, forecast_p10, forecast_p90, actual_price` (each `[ [48 values], ... ]`). The notebook validates these shapes and demonstrates plotting.

---

## Example: end-to-end usage (Python)

```python
from pathlib import Path
from epftoolbox.models import LEAR
from epftoolbox.data import read_data

from entsoe_price_data import get_entsoe_price_data
from lear_48hrs import get_forecast_issue_lear

# --- Configuration
api_key = "<YOUR_ENTSOE_API_KEY>"
country_code = "NL"
calibration_window = 728          # hours
rolling_window_days = 60
begin_test_date = "2018-01-01 00:00"
end_test_date   = "2018-03-31 23:00"

path_datasets_folder = Path("..") / "datasets"
path_recalibration_folder = Path("..") / "experimental_files"
path_recalibration_folder.mkdir(parents=True, exist_ok=True)

# --- 1) Fetch data & build dataset
dataset_stem = get_entsoe_price_data(
    api_key=api_key,
    country_code=country_code,
    calibration_window=calibration_window,
    rolling_window_days=rolling_window_days,
    begin_test_date=pd.to_datetime(begin_test_date),
    end_test_date=pd.to_datetime(end_test_date),
)

# --- 2) Read train/test frames for epftoolbox
df_train, df_test = read_data(
    dataset=dataset_stem,
    path=path_datasets_folder,
    begin_test_date=begin_test_date,
    end_test_date=end_test_date
)

# --- 3) Run LEAR 48h forecasts with bands
model = LEAR(calibration_window=calibration_window)
forecast_mean, forecast_p10, forecast_p90, actual_price = get_forecast_issue_lear(
    model=model,
    calibration_window=calibration_window,
    path_recalibration_folder=str(path_recalibration_folder),
    path_datasets_folder=str(path_datasets_folder),
    df_train_full=df_train,
    df_test_full=df_test,
    rolling_window_days=rolling_window_days
)
```

- The **dataset creation** step, including timezone handling and CSV naming, matches `entsoe_price_data.py`.  
- The **forecast call** implements the two-step 48-hour logic and returns SHIPP-compatible arrays with optional P10/P90.

---

## Visualizing results

The notebook demonstrates two plotting approaches:

- **Matplotlib** panel of issues vs. observations.  
- Optional **interactive Plotly** via `animated_plot.plot_interactive_forecasts(forecast_mean, forecast_p10, forecast_p90, actual_price, dt_issue=24, n_for=48)`.

---

## Notes & assumptions

- **Timezones & DST:** ENTSO-E data are requested in a Europe/Amsterdam timezone and then localized to naïve timestamps after resampling to hourly; **forward-fill** is used to ensure a complete hourly index through DST transitions.  
- **Output granularity:** Forecast issues are **every 24 hours**; each issue predicts **48 hours** ahead (hours 0–47).  
- **File outputs:** Intermediate D+1 and final 48-hour CSVs are stored under `experimental_files/` with filenames keyed by **country code** and **date span**.  
- **Shape checks:** The notebook contains asserts that verify SHIPP’s expected `[issues][1][48]` shape (and similar for bands and actuals).

---

## Quick start summary

1. Clone/download `epftoolbox`.  
2. Place this **`SHIPP Integration`** folder under `epftoolbox/examples/`.  
3. Create/activate the conda env (see `environment.yml`) and `pip install -e .` from the repo root.  
4. Get an ENTSO-E API key and set it in your environment or pass it directly.  
5. Run the notebook or a small driver script following the example above to generate the dataset and 48-hour forecasts.
