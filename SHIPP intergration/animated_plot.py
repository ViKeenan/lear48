import pandas as pd
import plotly.graph_objects as go

def plot_interactive_forecasts(forecast_mean, forecast_p10, forecast_p90, actual_price, dt_issue=24, n_for=48):
    """
    Plot interactive forecast vs. observation using Plotly, with P10/P90 bands.

    Parameters:
    - forecast_mean: list of forecast vectors in format [[ [...48 values...] ], ...]
    - forecast_p10: same structure as forecast_mean
    - forecast_p90: same structure as forecast_mean
    - actual_price: list of actual prices (from df_test['Price'].tolist())
    - dt_issue: hours between forecast issues (default = 24)
    - n_for: forecast horizon (default = 48)
    """
    nt = len(forecast_mean)
    fig = go.Figure()

    for i in range(nt):
        x_vals = list(range(i * dt_issue, i * dt_issue + n_for))

        # Extract forecast components
        y_forecast = forecast_mean[i][0]
        y_p10 = forecast_p10[i][0]
        y_p90 = forecast_p90[i][0]
        y_observed = actual_price[i * dt_issue : i * dt_issue + n_for]

        # Forecast mean
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_forecast,
            mode='lines',
            name='Forecast',
            visible=(i == 0),
            line=dict(color='blue')
        ))

        # P10
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_p10,
            mode='lines',
            name='P10',
            visible=(i == 0),
            line=dict(color='gray', dash='dot')
        ))

        # P90
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_p90,
            mode='lines',
            name='P90',
            visible=(i == 0),
            line=dict(color='gray', dash='dot')
        ))

        # Observed
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_observed,
            mode='lines',
            name='Observed',
            visible=(i == 0),
            line=dict(color='black', dash='dash')
        ))

    # Slider to toggle forecast issues
    steps = []
    for i in range(nt):
        visible = [False] * (4 * nt)
        visible[4*i + 0] = True  # mean
        visible[4*i + 1] = True  # p10
        visible[4*i + 2] = True  # p90
        visible[4*i + 3] = True  # observed

        step = dict(
            method="update",
            args=[{"visible": visible}],
            label=f"Issue {i}"
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Forecast Issue: "},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title="Interactive Forecast Viewer with P10/P90 Quantiles",
        xaxis_title="Time [h]",
        yaxis_title="Price [EUR/MWh]",
        height=600,
        width=950
    )

    fig.show()
