def get_forecast_issue_lear(year,
                            forecast_days,
                            model, 
                            calibration_window,
                            path_recalibration_folder, 
                            path_datasets_folder,
                            rolling_window_days,
                            api_key, 
                            country_code,
                            forecast_horizon
                            ):
    """
    Issue a 48-hour LEAR forecast with quantile bands (P10, P90).

    Parameters
    ----------
    data_price : list
        Historical hourly electricity prices.
    init_index : int
        Index at which to issue the forecast.
    model : epftoolbox.models.LEAR
        Initialized LEAR model.
    calibration_window : int
        Size of the rolling calibration window.
    df_train_full : pd.DataFrame
        Full training set from epftoolbox.
    df_test_full : pd.DataFrame
        Full test set from epftoolbox.
    forecast_horizon : int
        Length of forecast (default is 48 hours).
    rolling_window_days : int
        Lookback window to estimate residual quantiles.

    Returns
    -------
    list of 3 lists:
        - forecast (mean)
        - forecast P10 (10th percentile)
        - forecast P90 (90th percentile)
    """

    import pandas as pd
    import numpy as np

    from epftoolbox.data import read_data
    from epftoolbox.evaluation import MAE, sMAPE
    from epftoolbox.models import LEAR
    from entsoe_price_data import get_entsoe_price_data
    from datetime import datetime, timedelta
    from pathlib import Path

    # Forecast start & end date
    true_forecast_start_date = datetime(year, 1, 1)
    begin_test_date = true_forecast_start_date - timedelta(days=rolling_window_days)
    end_test_date = true_forecast_start_date + timedelta(days=forecast_days - 1)

    #Get the dataset
    dataset = get_entsoe_price_data(api_key, country_code, 
                                    calibration_window, rolling_window_days,
                                    begin_test_date, end_test_date)

    # Load input data for the forecast
    df_train_full, df_test_full = read_data(dataset=dataset,
                                path=path_datasets_folder,
                                begin_test_date=begin_test_date,
                                end_test_date=end_test_date)

    # --------------------------------------------------------------------
    # REAL VALUE EXTRACTION FOR VERIFICATION
    # --------------------------------------------------------------------

    # Initialize placeholder for 48h rolling forecasts and real data
    forecast = pd.DataFrame(index=df_test_full.index[::24], columns=['h' + str(k) for k in range(48)])
    real_values = []
    forecast_index = []

    # Manually slice 48-hour real price windows every 24 hours
    for i in range(0, len(df_test_full) - 47, 24):
        window = df_test_full['Price'].iloc[i:i+48].values
        if len(window) == 48:
            real_values.append(window)
            forecast_index.append(df_test_full.index[i])

    real_values = pd.DataFrame(real_values, index=forecast_index, columns=['h' + str(k) for k in range(48)])
    forecast_dates = forecast.index

    # --------------------------------------------------------------------
    # STEP 1: FORECAST HOURS 0–23 (DAY +1)
    # --------------------------------------------------------------------

    # Initialize model and forecast DataFrame
    forecast_d1 = pd.DataFrame(index=forecast_dates, columns=[f'h{k}' for k in range(24)])
    start_fst = pd.to_datetime(begin_test_date).strftime("%d%m%Y")
    end_fst = pd.to_datetime(end_test_date).strftime("%d%m%Y")
    filename_d1 = f"{country_code}_{start_fst}_{end_fst}_d1.csv"
    output_path_d1 = Path(path_recalibration_folder) / filename_d1

    for date in forecast_dates:
    # Prepare input: combine train and up-to-date test data
        data_available = pd.concat([df_train_full, df_test_full.loc[:date + pd.Timedelta(hours=23)]])

        # Mask the next 24h (to be predicted)
        data_available.loc[date:date + pd.Timedelta(hours=23), 'Price'] = np.NaN

        # Forecast next 24 hours using LEAR
        Yp_d1 = model.recalibrate_and_forecast_next_day(df=data_available,
                                                        next_day_date=date,
                                                        calibration_window=calibration_window)
        forecast_d1.loc[date] = Yp_d1.flatten()

        # Save interim results
        forecast_d1.to_csv(output_path_d1)
    
    # --------------------------------------------------------------------
    # STEP 2: FORECAST HOURS 24–47 (DAY +2) USING STEP 1 RESULTS
    # --------------------------------------------------------------------

    forecast_d1 = pd.read_csv(output_path_d1, index_col=0, parse_dates=True)
    forecast_d2 = pd.DataFrame(index=forecast_dates, columns=[f'h{k}' for k in range(24)])
    forecast_48h = pd.DataFrame(index=forecast_dates, columns=[f'h{k}' for k in range(48)])
    filename_48h = f"{country_code}_{start_fst}_{end_fst}_48hrs.csv"
    output_path_48h = Path(path_recalibration_folder) / filename_48h

    for date in forecast_dates:
        data_available = pd.concat([df_train_full, df_test_full.loc[:date + pd.Timedelta(hours=47)]])
        d1_index = pd.date_range(date, date + pd.Timedelta(hours=23), freq='h')

        # Replace day +1 prices with step-1 forecasts
        data_available.loc[d1_index, 'Price'] = forecast_d1.loc[date].values

        # Extend historical window to ensure context for D+2
        df_test_expanded = data_available.loc[date - pd.Timedelta(days=10):date + pd.Timedelta(days=2)]

        try:
            _, _, X_d2 = model._build_and_split_XYs(df_train=data_available,
                                                    df_test=df_test_expanded,
                                                    date_test=date + pd.Timedelta(days=1))
            Yp_d2 = model.predict(X_d2)
            forecast_d2.loc[date] = Yp_d2.flatten()

            # Concatenate 48h forecast
            forecast_48h.loc[date] = np.concatenate([forecast_d1.loc[date].values, Yp_d2.flatten()])

        except Exception as e:
            print(f"Skipping {date} due to: {e}")
            continue

    forecast_48h.to_csv(output_path_48h)


    # --------------------------------------------------------------------
    # FORECAST TO LONG FORMAT FOR ERROR ANALYSIS
    # --------------------------------------------------------------------

    df_long = forecast_48h.copy()
    df_long.index.name = 'forecast_issue_time'  # Ensure index has the correct name
    df_long = df_long.reset_index()  # Now this becomes a column

    records = []
    for _, row in df_long.iterrows():
        issue_time = row['forecast_issue_time']
        for h in range(48):
            records.append({
                'forecast_issue_time': issue_time,
                'target_time': issue_time + pd.Timedelta(hours=h),
                'lead_time_hr': h,
                'forecast_price': row[f'h{h}']
            })

    df_leadtime = pd.DataFrame(records)

    # Add actual prices to compute errors
    actual_prices = df_test_full['Price'].copy().reset_index()
    actual_prices.rename(columns={'Date': 'target_time', 'Price': 'actual_price'}, inplace=True)
    df_leadtime = df_leadtime.merge(actual_prices, on='target_time', how='left')

    # --------------------------------------------------------------------
    # PROBABILISTIC FORECASTING (QUANTILES & SCORING)
    # --------------------------------------------------------------------

    # Compute rolling residuals
    df_leadtime['residual'] = df_leadtime['actual_price'] - df_leadtime['forecast_price']
    start_prob_date = true_forecast_start_date
    quantile_records = []
    forecast_dates = sorted(df_leadtime['forecast_issue_time'].unique())

    # Rolling quantile computation per lead time
    for current_date in forecast_dates:
        if current_date < start_prob_date:
            continue
        window_start = current_date - pd.Timedelta(days=rolling_window_days)
        residuals_window = df_leadtime[
            (df_leadtime['forecast_issue_time'] >= window_start) &
            (df_leadtime['forecast_issue_time'] < current_date)
        ]

        quantiles = (
            residuals_window
            .groupby('lead_time_hr')['residual']
            .apply(lambda x: x.quantile([0.1, 0.9]) if len(x) >= 30 else pd.Series([np.nan, np.nan], index=[0.1, 0.9]))
            .unstack()
        )
        quantiles.columns = ['q10', 'q90']

        current_forecast = df_leadtime[df_leadtime['forecast_issue_time'] == current_date].copy()
        current_with_quantiles = current_forecast.merge(quantiles, on='lead_time_hr', how='left')
        current_with_quantiles['P10'] = current_with_quantiles['forecast_price'] + current_with_quantiles['q10']
        current_with_quantiles['P90'] = current_with_quantiles['forecast_price'] + current_with_quantiles['q90']
        quantile_records.append(current_with_quantiles)

    df_prob_forecast = pd.concat(quantile_records, ignore_index=True)

    # Group by forecast_issue_time and select relevant columns for output
    grouped = df_prob_forecast.groupby('forecast_issue_time')

    forecast_mean = [ [group['forecast_price'].tolist()] for _, group in grouped ]
    forecast_p10  = [ [group['P10'].tolist()] for _, group in grouped ]
    forecast_p90  = [ [group['P90'].tolist()] for _, group in grouped ]
    actual_price = [[group['actual_price'].tolist()] for _, group in grouped]

    return forecast_mean, forecast_p10, forecast_p90, actual_price