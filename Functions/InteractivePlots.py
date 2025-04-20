import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import itertools



# VaR Plot Backtesting
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



# ES Plot Backtesting
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



# Volatility Plot
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



# Asset Color Map (for portfolio visualization)
def get_asset_color_map(assets):
    """
    Generate a consistent color map for any list of assets using a cycling color palette.

    Parameters
    ----------
    assets : list-like
        List of asset names (strings).

    Returns
    -------
    dict
        Dictionary mapping each asset to a unique color.
    """
    base_colors = px.colors.qualitative.Plotly
    color_cycle = itertools.cycle(base_colors)
    return {asset: next(color_cycle) for asset in assets}



# VaR and UVaR Plot (for portfolio visualization)
def plot_var_series(var_series, uvar_series):
    """
    Plot Asset-Normal VaR and Undiversified VaR over time.

    Parameters
    ----------
    var_series : pd.Series
        Time series of diversified VaR (Asset-Normal VaR).
    uvar_series : pd.Series
        Time series of Undiversified VaR (ρ = 1 case).

    Returns
    -------
    None
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=var_series.index,
        y=var_series,
        name='VaR',
        line=dict(color='blue'),
        hovertemplate="Date: %{x}<br>VaR = %{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=uvar_series.index,
        y=uvar_series,
        name='Undiversified VaR',
        line=dict(color='red'),
        hovertemplate="Date: %{x}<br>Undiversified VaR = %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title='Asset-Normal VaR vs. Undiversified VaR',
        xaxis_title='Date',
        yaxis_title='VaR (monetary units)',
        template='simple_white',
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )
    fig.show()



# Risk Contribution Pie Chart (for portfolio visualization)
def plot_risk_contribution_pie(component_df):
    """
    Plot an interactive pie chart of average absolute Component VaR per asset.

    Parameters
    ----------
    component_df : pd.DataFrame
        Time series of Component VaR values (T x N).

    Returns
    -------
    None
    """
    average_contributions = component_df.abs().mean()
    total = average_contributions.sum()

    if total == 0:
        raise ValueError("All component VaR contributions are zero; cannot plot pie chart.")

    asset_colors = get_asset_color_map(average_contributions.index)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=average_contributions.index,
                values=average_contributions,
                hovertemplate="%{label}<br>Average Contribution = %{value:.3f} (%{percent})<extra></extra>",
                textinfo='label+percent',
                marker=dict(
                    colors=[asset_colors[asset] for asset in average_contributions.index],
                    line=dict(color='black', width=1)
                ),
                hole=0.3
            )
        ]
    )

    fig.update_layout(
        title='Average Component VaR Contribution by Asset',
        template='simple_white',
        margin=dict(l=60, r=60, t=50, b=50)
    )

    fig.show()



# Risk Contribution Over Time (for portfolio visualization)
def plot_risk_contribution_lines(component_df):
    """
    Plot Component VaR over time for each asset using consistent colors.

    Parameters
    ----------
    component_df : pd.DataFrame
        Time series of Component VaR values (T x N).

    Returns
    -------
    None
    """
    asset_colors = get_asset_color_map(component_df.columns)

    fig = go.Figure()

    for asset in component_df.columns:
        fig.add_trace(go.Scatter(
            x=component_df.index,
            y=component_df[asset],
            mode="lines",
            name=asset,
            line=dict(color=asset_colors[asset]),
            hovertemplate="Date: %{x}<br>" + asset + " VaR = %{y:.3f}<extra></extra>"
        ))

    fig.update_layout(
        title="Component VaR by Asset Over Time",
        yaxis_title="Component VaR (monetary units)",
        xaxis_title="Date",
        template="simple_white",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        margin=dict(l=60, r=60, t=50, b=50)
    )

    fig.show()
