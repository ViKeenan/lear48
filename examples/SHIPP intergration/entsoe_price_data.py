from entsoe import EntsoePandasClient
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

def get_entsoe_price_data(api_key, country_code, 
                          calibration_window, rolling_window_days,
                          begin_test_date, end_test_date):
    
    """
    Retrieve day-ahead prices, grid load, and wind forecasts from ENTSO-E and save as a CSV file.
    
    Parameters:
    -----------
    api_key : str
        Your ENTSO-E API key.
    country_code : str
        Country code in ENTSO-E format (e.g., 'NL' for Netherlands).
    start_str : str
        Start date in "YYYY-MM-DD HH:MM" format.
    end_str : str
        End date in "YYYY-MM-DD HH:MM" format.
        
    Returns:
    --------
    str
        The name of the saved file (as 'file').
    """

    #Setup Client and Time
    client = EntsoePandasClient(api_key=api_key)

    # Data extraction period needs padding: calibration + rolling + 1 day for end date
    total_padding = calibration_window + rolling_window_days + 1
    data_start = begin_test_date - timedelta(days=total_padding)
    data_end = end_test_date + timedelta(days=1)  # To get values until 23:00

    # Format for Entsoe
    start_str = data_start.strftime("%Y-%m-%d %H:%M")
    end_str = data_end.strftime("%Y-%m-%d %H:%M")

    start = pd.Timestamp(start_str, tz='Europe/Amsterdam')
    end = pd.Timestamp(end_str, tz='Europe/Amsterdam')

    # Query data
    day_ahead_price = client.query_day_ahead_prices(country_code, start=start, end=end)
    load_forecast = client.query_load_forecast(country_code, start=start, end=end)
    wind_forecast = client.query_wind_and_solar_forecast(country_code, start=start, end=end)

    #Combine data and uniform (Wind Data needs to be 1D)
    wind_forecast_series = wind_forecast['Wind Onshore'] + wind_forecast['Wind Offshore']
    load_forecast_series = load_forecast.iloc[:, 0]

    df = pd.DataFrame({
        'Price': day_ahead_price.squeeze(),
        'Grid load forecast': load_forecast_series,
        'Wind power forecast': wind_forecast_series,
    }).reset_index().rename(columns={'index': 'Date'})

    df = df.resample('h', on='Date').mean()

    #Reset Index and ensure Date is in correct format and sorted
    if df.index.name == 'Date' or df.index.name is not None:
        df = df.reset_index()
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.drop_duplicates('Date', keep='first')
    df = df.set_index('Date')

    #Forward fill missing hours (accounting for time zones and changes in DST)
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
    df = df.reindex(full_range)
    df = df.ffill()
    df.index.name = 'Date'
    df = df.reset_index()

    # Build filename based on country code and dates
    start_fmt = pd.to_datetime(start_str).strftime("%d%m%Y")
    end_fmt = pd.to_datetime(end_str).strftime("%d%m%Y")
    filename = f"{country_code}_{start_fmt}_{end_fmt}.csv"
    output_path = Path("..") / "datasets" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    return output_path.stem

