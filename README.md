# 48h Forecast of DA Prices - LEAR model integration into SHIPP optimisation framework

The folder provided in this repository integrates **electricity price forecasting** into the SHIPP workflow by using the **LEAR** (Lasso Estimated AutoRegressive) model from [`epftoolbox`](https://github.com/jeslago/epftoolbox). It includes:

- A data retriever for ENTSO-E day-ahead prices and system forecasts,
- A 48-hour ahead forecasting model with daily recalibration produceing **mean** and **P10/P90** bands,
- Checks for forecast output compatibility with the wider framework and an interactive visualisation tool.

> **Placement:** the folder named **`SHIPP Integration`** was created to be **dropped into the `examples/` directory** of the `epftoolbox` repository. After downloading or cloning `epftoolbox` (as described in its documentation), copy or move this folder to:
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
  Initiates the **48-hour rolling forecast** with LEAR. For each issue date:
  1) **Recalibrates** LEAR and forecasts **D+1 (hours 0–23)**;  
  2) Uses those D+1 results to forecast **D+2 (hours 24–47)**;  
  3) Builds a long table of forecasts vs. actuals to compute **rolling residual quantiles** per lead-time and derive **P10/P90** bands;  
  4) Saves intermediate **D+1** CSVs and the **48-hour** CSVs in `experimental_files/`.  
  Returns four lists (per issue): **mean forecast**, **P10**, **P90**, and **actual prices**.

- **`price_forecast.py` / notebook**  
  A reference notebook that shows how to replace a dummy AR model with the LEAR-based pipeline above, how to read datasets, run the 48-hour forecast, and plot results (including an optional interactive Plotly view via `animated_plot.plot_interactive_forecasts`).

## Required credentials (ENTSO-E)

You need an **ENTSO-E API key** to download data. Pass it directly to the function or expose it as an environment variable and read it in your launcher script.

---

## On the LEAR model and the 48-hour extension
Originally proposed by Uniejewski, Ziel, and Weron 2016, the Lasso Estimated AutoRegressive model is a linear ARX specification estimated with LASSO. It is essentially an autoregressive model with exogenous inputs (ARX) where a large number of potential regressors are considered, such as past hourly prices (lags), past daily patterns, and contemporaneous or lagged exogenous variables (load, wind forecasts, etc.). For each delivery hour h, a separate equation is fitted,

$${p_{t,h} = \alpha_h + \sum_{\ell} \phi_{h,\ell} \, p_{t-\ell,h} + \sum_{k} \gamma_{h,k} \, d_{k,t} + \sum_{j} \beta_{h,j} \, x_{t-j} + \varepsilon_{t,h}}$$  

where $p_{t-\ell,h}$ are price lags (intra-day/week), $d_{k,t}$ are calendar/seasonal dummies, and $x_{t-j}$ are exogenous inputs (i.e., load, wind forecasts).

The standard use-case for LEAR (as in Lago, De Ridder, and De Schutter 2021) is to forecast the next 24 hours (day-ahead) given data up to the issue time. We extend this to a 48-hour horizon by a two-step sequential approach. The process is as follows:
- Step 1 (Day+1 forecast): At the given issue time, train the LEAR model on the 2-year window up to present, then predict the next 24 hourly prices (for the next day). This yields a day-ahead price forecast vector for hours 0,23 of tomorrow. In our implementation, we recalibrate the model for each day’s forecast issuance to ensure it adapts to the latest data (this daily retraining strategy follows
Lago, De Ridder, and De Schutter’s recommendation for robust performance).
- Step 2 (Day+2 forecast): To forecast hours 24,47 (the second day ahead), we leverage the results of Step 1. We append the day+1 forecasted prices to the historical dataset as if they were “pseudo-observations” for that day. In other words, when predicting the second day, we assume the first day’s forecast is accurate (or at least the best available information for those hours). We then recalibrate or reuse the model to predict an additional 24 hours beyond the first forecast. This yields the prices for day+2 (hours 24,47 ahead).
  
Beyond point forecasts, we quantify uncertainty by constructing prediction intervals for each lead time, implementing a quantile estimation approach using the distribution of recent forecast errors (residuals). For each forecast issue (after we have actuals), we compute the error = actual price - forecast price for every hour lead (0 to 47). Over a rolling window (e.g. the past 60 days of forecast issues ), we gather the residuals for each specific lead horizon. Assuming that past errors are indicative of future uncertainty, we estimate the 10th and 90th percentiles of the residual distribution for each lead time. If
fewer than a minimum number of residual points are available (we require at least 30 data points to ensure a stable quantile estimate), we default to no interval for that lead at that time. We construct prediction bands as

$${P10_{t+h} = \hat{p}_{t+h} + q_{0.10}(e_{t+h})}$$  

$${P90_{t+h} = \hat{p}_{t+h} + q_{0.90}(e_{t+h})}$$  

In this context, “rolling” does not refer to re-training the point-forecast model day by day, rather, it describes how the uncertainty bands are constructed. For each forecast-issue date, once the corresponding actuals are available, the model takes a standard 60-day look-back of residuals (actual - point forecast) for each lead time (0–47) and derives the empirical 10th and 90th percentiles. These lead-specific quantiles are then added to the current point forecast to produce the P10 and P90 bands; as new days arrive, the 60-day window advances, so the bands adapt to the most recent error behaviour. 

---

## Example

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
