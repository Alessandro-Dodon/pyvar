import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import plotly.graph_objects as go



# Volatility based VaR
def var_garch(returns, confidence_level, p=1, q=1):
    """
    Fit a GARCH(p,q) model and compute empirical daily VaR.

    Parameters:
    - returns: pd.Series (unscaled)
    - confidence_level: float
    - p, q: GARCH orders (default 1,1)

    Returns:
    - result_data: pd.DataFrame (unscaled)
    - next_day_var: float (unscaled)
    """
    returns_scaled = returns * 100  # scale for stability

    garch_model = arch_model(returns_scaled, vol="Garch", p=p, q=q)
    garch_fit = garch_model.fit(disp="off")

    volatility = garch_fit.conditional_volatility / 100  # rescale
    innovations = returns / volatility  # use original returns for innovations
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    next_day_vol = (garch_fit.forecast(horizon=1).variance.values[-1][0] ** 0.5) / 100
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data.dropna(), next_day_var



def var_arch(returns, confidence_level, p=1):
    """
    Compute VaR using ARCH(p) model.

    Parameters:
    - returns: pd.Series (unscaled)
    - confidence_level: float
    - p: ARCH order (default 1)

    Returns:
    - result_data: pd.DataFrame (unscaled)
    - next_day_var: float (unscaled)
    """
    returns_scaled = returns * 100

    model = arch_model(returns_scaled, vol="ARCH", p=p)
    fit = model.fit(disp="off")

    volatility = fit.conditional_volatility / 100
    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    next_day_vol = (fit.forecast(horizon=1).variance.values[-1][0] ** 0.5) / 100
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data.dropna(), next_day_var



def var_ewma(returns, confidence_level, decay_factor=0.94):
    """
    Compute VaR using EWMA volatility model.

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - decay_factor: float (lambda), default 0.94

    Returns:
    - result_data: pd.DataFrame
    - next_day_var: float
    """
    squared = returns ** 2
    ewma_var = squared.ewm(alpha=1 - decay_factor).mean()
    volatility = np.sqrt(ewma_var)

    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]
    result_data.dropna(inplace=True)

    next_day_vol = volatility.iloc[-1]
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data, next_day_var



def var_moving_average(returns, confidence_level, window=20):
    """
    Compute VaR using rolling standard deviation (moving average volatility).

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - window: int, size of rolling window (default 20)

    Returns:
    - result_data: pd.DataFrame
    - next_day_var: float
    """
    volatility = returns.rolling(window=window).std()
    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]
    result_data.dropna(inplace=True)

    next_day_vol = volatility.iloc[-1]
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data, next_day_var



# VaR Backtesting (General)
def backtest_var(data, confidence_level):
    """
    Backtest VaR by counting violations.

    Parameters:
    - data: pd.DataFrame with 'Returns', 'VaR', 'VaR Violation'
    - confidence_level: float

    Returns:
    - total_violations: int
    - violation_rate: float
    """
    violations = data["VaR Violation"]
    total_violations = violations.sum()
    total_days = len(violations)
    violation_rate = 100 * total_violations / total_days

    return total_violations, violation_rate



def subset_backtest_var(data, start_date, end_date):
    """
    Count VaR violations and violation rate within a specific date subset.

    Parameters:
    - data: pd.DataFrame with 'VaR Violation' column and a datetime index
    - start_date: str or pd.Timestamp, e.g. "2019-01-01"
    - end_date: str or pd.Timestamp, e.g. "2020-01-01"

    Returns:
    - total_violations: int
    - violation_rate: float (%)
    """
    if "VaR Violation" not in data.columns:
        raise ValueError("Data must contain 'VaR Violation' column.")
    
    subset = data.loc[start_date:end_date]
    total_violations = subset["VaR Violation"].sum()
    total_days = len(subset)
    
    if total_days == 0:
        raise ValueError("Subset is empty. Check your date range.")

    violation_rate = 100 * total_violations / total_days
    return total_violations, violation_rate



# Expected Shortfall (Volatility)
def compute_expected_shortfall_volatility(data, confidence_level, subset=None):
    """
    Compute Expected Shortfall (ES) based on empirical innovations.

    Parameters:
    - data: pd.DataFrame with 'Innovations' and 'Volatility'
    - confidence_level: float
    - subset: tuple (start_date, end_date) or None

    Returns:
    - data: pd.DataFrame with new 'ES' column (full period)
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")
    
    subset_data = data.copy()
    if subset is not None:
        subset_data = data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"].dropna(), 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    data["ES"] = -data["Volatility"] * tail_mean
    return data



# Interactive plots
def interactive_plot_var(data, subset=None):
    """
    Interactive version of plot_var using Plotly with white background,
    black border, cleaner hover formatting (VaR as positive), and optional subset.

    Parameters:
    - data: pd.DataFrame with 'Returns', 'VaR', 'VaR Violation'
    - subset: tuple (start_date, end_date) or None
    """
    if subset is not None:
        data = data.loc[subset[0]:subset[1]]

    violations = data["VaR Violation"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=100 * data["Returns"],
        mode="lines",
        name="Log Returns",
        line=dict(color="blue", width=0.8),
        hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=-100 * data["VaR"],
        mode="lines",
        name="VaR Level",
        line=dict(color="black", width=0.8),
        hovertemplate="Date: %{x}<br>VaR: %{customdata:.2f}%",
        customdata=100 * data["VaR"].abs()
    ))

    fig.add_trace(go.Scatter(
        x=data.index[violations],
        y=100 * data["Returns"][violations],
        mode="markers",
        name="VaR Violation",
        marker=dict(color="red", symbol="circle-open", size=8),
        hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
    ))

    fig.update_layout(
        title="Backtesting Value-at-Risk",
        yaxis_title="Returns (%)",
        hovermode="x unified",
        height=500,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )

    return fig



def interactive_plot_es(data, subset=None):
    """
    Interactive version of plot_es using Plotly with white background,
    black border, zero line, and cleaner hover formatting (VaR/ES as positive).
    """
    if subset is not None:
        data = data.loc[subset[0]:subset[1]]

    violations = data["VaR Violation"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=data.index,
        y=100 * data["Returns"],
        name="Log Returns",
        marker_color="blue",
        hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=-100 * data["VaR"],
        mode="lines",
        name="VaR",
        line=dict(color="black", width=0.8),
        hovertemplate="Date: %{x}<br>VaR: %{customdata:.2f}%",
        customdata=100 * data["VaR"].abs()
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=-100 * data["ES"],
        mode="lines",
        name="ES",
        line=dict(color="red", dash="dash", width=0.8),
        hovertemplate="Date: %{x}<br>ES: %{customdata:.2f}%",
        customdata=100 * data["ES"].abs()
    ))

    fig.add_trace(go.Scatter(
        x=data.index[violations],
        y=100 * data["Returns"][violations],
        mode="markers",
        name="VaR Violation",
        marker=dict(color="red", symbol="circle-open", size=8),
        hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
    ))

    fig.add_shape(
        type="line",
        x0=data.index.min(), x1=data.index.max(),
        y0=0, y1=0,
        line=dict(color="black", width=0.5),
        xref="x", yref="y"
    )

    fig.update_layout(
        title="Backtesting Expected Shortfall",
        yaxis_title="Returns (%)",
        hovermode="x unified",
        height=500,
        width=1000,
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )

    return fig



def interactive_plot_volatility(volatility_series, subset=None):
    """
    Interactive plot of conditional volatility with white background and black border.

    Parameters:
    - volatility_series: pd.Series with datetime index and volatility values
    - subset: tuple (start_date, end_date) or None

    Returns:
    - fig: plotly.graph_objects.Figure
    """
    if subset is not None:
        volatility_series = volatility_series.loc[subset[0]:subset[1]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility_series.index,
        y=100 * volatility_series,  # Convert to percentage
        mode="lines",
        name="Volatility",
        line=dict(color="blue", width=0.8),
        hovertemplate="Date: %{x}<br>Volatility: %{y:.2f}%"
    ))

    fig.update_layout(
        title="Volatility Estimate",
        yaxis_title="Volatility (%)",
        hovermode="x unified",
        height=500,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )

    return fig