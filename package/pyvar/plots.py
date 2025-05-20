"""
Interactive Visualization Module for Risk and Portfolio Analysis
----------------------------------------------------------------

Provides plotting utilities for risk metrics, portfolio contributions, 
and correlation structures using Plotly and Matplotlib. All Plotly-based 
plots follow a consistent white-background aesthetic with black-bordered 
axes and support both interactive display and static image export.

This module is designed to be best used in Jupyter notebooks or similar environments.

Static images can be exported using `output_path`. If no path is provided,
a high-resolution inline image is displayed.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- plot_backtest: Backtesting VaR and ES with return violations
- plot_volatility: Line plot of conditional volatility
- plot_var_series: Diversified vs. undiversified VaR comparison
- plot_risk_contribution_bar: Average Component VaR bar chart
- plot_risk_contribution_lines: Component VaR evolution by asset
- plot_correlation_matrix: Heatmap of asset return correlations
- plot_simulated_distribution: Histogram + KDE of simulated P&L with VaR/ES (static only)
- plot_simulated_paths: Simulated portfolio value trajectories (static only)
- get_asset_color_map: Generate consistent color assignment for assets
- display_high_dpi_inline: Utility for displaying PNG images inline
"""

# TODO: double check all names/ titles

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import itertools
import kaleido  # Required for static image export via Plotly
from IPython.display import display, HTML
from plotly.io import to_image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import seaborn as sns


#----------------------------------------------------------
# Display helper 
#----------------------------------------------------------
def display_high_dpi_inline(png_bytes, width):
    """
    Main
    ----
    Display a high-resolution PNG image inline in a notebook.
    Encodes the image in base64 and renders it with a specified width.
    This is a support function.

    Parameters
    ----------
    png_bytes : bytes
        PNG image in byte format.
    width : int
        Width in pixels for display.

    Returns
    -------
    IPython.display.HTML
        HTML image element for inline display.
    """
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return HTML(f'<img src="data:image/png;base64,{encoded}" style="width:{width}px;"/>')


#----------------------------------------------------------
# Backtesting Plot for VaR and ES 
#----------------------------------------------------------
def plot_backtest(data, subset=None, interactive=True, output_path=None):
    """
    Main
    ----
    Generate a backtesting plot for Value-at-Risk (VaR) and optionally Expected Shortfall (ES).

    Displays returns, VaR threshold, and VaR violations. Adds ES line if present.
    Automatically switches between bar and line plot depending on time span.
    Can show plot interactively or export as static image.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'Returns', 'VaR', and 'VaR Violation'. Optionally 'ES'.
    subset : tuple, optional
        Start and end date for subsetting the time window.
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PDF if interactive is False.

    Returns
    -------
    plotly.graph_objs.Figure
        Generated plotly figure.
    """
    if subset is not None:
        data = data.loc[subset[0]:subset[1]]

    use_bars = len(data) <= 504
    violations = data["VaR Violation"]
    has_es = "ES" in data.columns

    # Set title dynamically
    title = "Backtesting Value-at-Risk and Expected Shortfall" if has_es else "Backtesting Value-at-Risk"

    fig = go.Figure()

    # Plot returns
    if use_bars:
        fig.add_trace(go.Bar(
            x=data.index,
            y=100 * data["Returns"],
            name="Returns",
            marker=dict(color="blue", line=dict(color="black", width=0)),
            opacity=1.0,
            hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=100 * data["Returns"],
            mode="lines",
            name="Returns",
            line=dict(color="blue", width=0.8),
            hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
        ))

    # Plot VaR line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=-100 * data["VaR"],
        mode="lines",
        name="VaR",
        line=dict(color="black", width=0.8),
        hovertemplate="Date: %{x}<br>VaR: %{customdata:.2f}%",
        customdata=100 * data["VaR"].abs()
    ))

    # Plot ES line if present
    if has_es:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=-100 * data["ES"],
            mode="lines",
            name="ES",
            line=dict(color="red", dash="dash", width=0.8),
            hovertemplate="Date: %{x}<br>ES: %{customdata:.2f}%",
            customdata=100 * data["ES"].abs()
        ))

    # Plot violations
    fig.add_trace(go.Scatter(
        x=data.index[violations],
        y=100 * data["Returns"][violations],
        mode="markers",
        name="VaR Violation",
        marker=dict(color="red", symbol="circle-open", size=8),
        hovertemplate="Date: %{x}<br>Return: %{y:.2f}%"
    ))

    # Add zero line only if bars are used
    if use_bars:
        fig.add_shape(
            type="line",
            x0=data.index.min(), x1=data.index.max(),
            y0=0, y1=0,
            line=dict(color="black", width=0.5),
            xref="x", yref="y"
        )

        # Layout
    fig.update_layout(
        title=title,
        yaxis_title="Returns (%)",
        hovermode="x unified",
        height=500,
        width=1000,
        barmode="overlay",
        bargap=0.025,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            range=[data.index.min(), data.index.max()]  # 🔧 Ensures full-width alignment
        ),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )

    # Export
    width = fig.layout.width or 1000
    height = fig.layout.height or 500
    scale = 4

    if interactive:
        fig.show()
        return fig
    elif output_path:
        fig.write_image(output_path, format="pdf", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))

    return fig


#----------------------------------------------------------
# Volatility Plot
#----------------------------------------------------------
def plot_volatility(volatility_series, subset=None, interactive=True, output_path=None):
    """
    Main
    ----
    Plot conditional volatility over time.

    Generates an interactive line plot of volatility estimates (in %). 
    Supports subsetting, export to PDF, or inline display.

    Parameters
    ----------
    volatility_series : pd.Series
        Time series of volatility estimates in decimal format (e.g., 0.02 = 2%).
    subset : tuple, optional
        Start and end date as (start_date, end_date) to zoom in.
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PDF if interactive is False.

    Returns
    -------
    plotly.graph_objs.Figure
        Plotly figure of volatility over time.
    """
    if subset is not None:
        volatility_series = volatility_series.loc[subset[0]:subset[1]]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=volatility_series.index,
        y=100 * volatility_series,
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

    width = fig.layout.width or 1000
    height = fig.layout.height or 500
    scale = 4

    if interactive:
        fig.show()
        return fig

    elif output_path:
        fig.write_image(output_path, format="pdf", width=width, height=height, scale=scale)

    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


#----------------------------------------------------------
# Asset Color Map (for portfolio visualization)
#----------------------------------------------------------
def get_asset_color_map(assets):
    """
    Main
    ----
    Generate consistent colors for asset-level visualizations.
    Assigns a unique color to each asset using Plotly's qualitative palette, 
    cycling through it as needed. Useful for consistent coloring in portfolio plots.
    This is a support function.

    Parameters
    ----------
    assets : list-like
        List of asset names (strings).

    Returns
    -------
    dict
        Dictionary mapping each asset to a color string.
    """
    base_colors = px.colors.qualitative.Plotly
    color_cycle = itertools.cycle(base_colors)
    return {asset: next(color_cycle) for asset in assets}


#----------------------------------------------------------
# VaR and UVaR Plot 
#----------------------------------------------------------
def plot_var_series(var_df, interactive=True, output_path=None):
    """
    Main
    ----
    Plot diversified vs. undiversified Value-at-Risk (VaR) over time.

    Displays a line chart comparing diversified VaR (accounting for correlation)
    and undiversified VaR (assuming full asset correlation). Useful for visualizing
    the diversification benefit in monetary units.

    Parameters
    ----------
    var_df : pd.DataFrame
        Must contain the columns:
        - 'Diversified_VaR': Time series of portfolio VaR
        - 'Undiversified_VaR': VaR assuming ρ = 1 for all assets
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PDF if interactive is False.

    Returns
    -------
    None
        Displays or saves the plot; does not return a figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=var_df.index,
        y=var_df["Diversified_VaR"],
        name='VaR',
        line=dict(color='blue'),
        hovertemplate="Date: %{x}<br>VaR = %{y:.2f}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=var_df.index,
        y=var_df["Undiversified_VaR"],
        name='Undiversified VaR',
        line=dict(color='red'),
        hovertemplate="Date: %{x}<br>Undiversified VaR = %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title='Asset-Normal VaR vs. Undiversified VaR Over Time',
        xaxis_title='Date',
        yaxis_title='VaR (monetary units)',
        template='simple_white',
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True)
    )

    # Ensure consistent width and height
    if fig.layout.width is None:
        fig.update_layout(width=1000)
    if fig.layout.height is None:
        fig.update_layout(height=500)

    width = fig.layout.width
    height = fig.layout.height
    scale = 4

    if interactive:
        fig.show()
    elif output_path:
        fig.write_image(output_path, format="pdf", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


# ----------------------------------------------------------
# Risk Contribution Bar Chart 
# ----------------------------------------------------------
def plot_risk_contribution_bar(component_df, interactive=True, output_path=None):
    """
    Main
    ----
    Plot average absolute Component VaR contributions by asset.

    Displays a horizontal bar chart showing the average absolute 
    risk contribution of each asset to portfolio VaR. Useful for 
    visualizing the relative importance of individual assets.

    Parameters
    ----------
    component_df : pd.DataFrame
        Time series of Component VaR values (T × N).
        Each column corresponds to an asset.
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PDF if interactive is False.

    Returns
    -------
    plotly.graph_objs.Figure
        Interactive Plotly figure object.
    
    Raises
    ------
    ValueError
        If all component VaR contributions are zero.
    """
    average_contributions = component_df.abs().mean()
    total = average_contributions.sum()

    if total == 0:
        raise ValueError("All component VaR contributions are zero; cannot plot bar chart.")

    percentages = 100 * average_contributions / total
    sorted_assets = average_contributions.sort_values().index
    asset_colors = get_asset_color_map(sorted_assets)

    fig = go.Figure(
        data=[
            go.Bar(
                x=average_contributions[sorted_assets].values,
                y=sorted_assets,
                orientation="h",
                marker=dict(
                    color=[asset_colors[asset] for asset in sorted_assets],
                    line=dict(color='black', width=1)
                ),
                hovertemplate="%{y}<br>Average Absolute CVaR = %{x:.2f}<br>% Contribution = %{customdata:.2f}%<extra></extra>",
                customdata=percentages[sorted_assets].values,
                name="Average Component VaR"
            )
        ]
    )

    fig.update_layout(
        title="Average Absolute Component VaR by Asset",
        xaxis_title="Average Absolute CVaR",
        yaxis_title="Asset",
        template="simple_white",
        height=500,
        width=1000,
        margin=dict(l=80, r=40, t=50, b=50),
        showlegend=False
    )

    width = fig.layout.width or 1000
    height = fig.layout.height or 500
    scale = 4

    if interactive:
        fig.show()
        return  # Prevent double rendering
    elif output_path:
        fig.write_image(output_path, format="pdf", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


#----------------------------------------------------------
# Component VaR Over Time 
#----------------------------------------------------------
def plot_risk_contribution_lines(component_df, interactive=True, output_path=None):
    """
    Plot Component VaR contributions by asset over time.

    Displays a multi-line chart showing how each asset's Component VaR evolves. 
    Useful for tracking risk dynamics and identifying shifting contributions.

    Parameters
    ----------
    component_df : pd.DataFrame
        Time series of Component VaR values (T × N), one column per asset.
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PDF if interactive is False.

    Returns
    -------
    None
        Displays or saves the plot; does not return a figure.
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
            hovertemplate=f"Date: %{{x}}<br>{asset} CVaR = %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title="Component VaR by Asset Over Time",
        yaxis_title="Component VaR (monetary units)",
        xaxis_title="Date",
        template="simple_white",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, tickformat="%Y-%m-%d"),
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
        margin=dict(l=60, r=60, t=50, b=50),
        height=500,
        width=1000
    )

    width = fig.layout.width or 1000
    height = fig.layout.height or 500
    scale = 4

    if interactive:
        fig.show()
        return  # Prevent duplicate display
    elif output_path:
        fig.write_image(output_path, format="pdf", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


#----------------------------------------------------------
# Correlation Matrix Heatmap
#----------------------------------------------------------
def plot_correlation_matrix(position_data, interactive=True, output_path=None):
    """
    Main
    ----
    Plot a correlation heatmap of asset returns.

    Computes returns from monetary positions and visualizes the return correlation matrix 
    as a lower-triangular heatmap with a red-blue diverging color scale. Hover labels 
    show asset pairs and correlation values. Axis labels are hidden for a cleaner look.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Time series of monetary positions (T × N).
    interactive : bool, optional
        Whether to display the plot interactively. Default is True.
    output_path : str, optional
        File path to save static PNG if interactive is False.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive correlation matrix heatmap.

    Notes
    -----
    - This must be exported to PNG to preserve quality.
    - Always uses percentage returns (pct_change).
    - Masks the upper triangle to remove redundant values.
    """
    position_data = pd.DataFrame(position_data).dropna()

    # Always use percentage returns
    returns = position_data.pct_change().dropna()

    corr_matrix = returns.corr()
    asset_names = corr_matrix.columns.tolist()

    mask = np.tril(np.ones(corr_matrix.shape, dtype=bool))
    masked_corr = np.where(mask, corr_matrix.values, np.nan)

    hover_text = [
        [f"{asset_i} & {asset_j}<br>Correlation = {corr_matrix.iloc[i, j]:.2f}" if mask[i, j] else ""
         for j, asset_j in enumerate(asset_names)]
        for i, asset_i in enumerate(asset_names)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=masked_corr,
        x=asset_names,
        y=asset_names,
        text=hover_text,
        hoverinfo='text',
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        showscale=True,
        colorbar=dict(title='Correlation', thickness=15, len=0.75)
    ))

    fig.update_layout(
        title="Correlation Matrix (Percentage Returns)",
        width=900,
        height=800,
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=80, b=80),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed")
    )

    # White gridlines
    for i in range(len(asset_names)):
        fig.add_shape(type="line", x0=-0.5, x1=len(asset_names) - 0.5,
                      y0=i - 0.5, y1=i - 0.5, line=dict(color="white", width=1),
                      xref="x", yref="y")
        fig.add_shape(type="line", x0=i - 0.5, x1=i - 0.5,
                      y0=-0.5, y1=len(asset_names) - 0.5, line=dict(color="white", width=1),
                      xref="x", yref="y")

    width = fig.layout.width or 900
    height = fig.layout.height or 800
    scale = 4

    if interactive:
        fig.show()
        return
    elif output_path:
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


# ----------------------------------------------------------
# Simulated P&L Distribution Plot with KDE (Static Only)
# ----------------------------------------------------------
def plot_simulated_distribution(profit_and_loss, var, es, confidence_level=0.99, output_path=None):
    """
    Main
    ----
    Plot histogram and KDE of simulated Profit & Loss (P&L) with VaR and ES markers.

    Suitable for visualizing risk from Monte Carlo, bootstrapped, or historical
    simulations. Combines histogram and smoothed KDE, with vertical lines marking
    Value-at-Risk and Expected Shortfall.

    This is a static only plot.

    Parameters
    ----------
    profit_and_loss : np.ndarray or pd.Series
        Simulated profit and loss values.
    var : float
        Value-at-Risk estimate.
    es : float
        Expected Shortfall estimate.
    confidence_level : float, optional
        Confidence level for VaR and ES (e.g., 0.99). Default is 0.99.
    output_path : str, optional
        File path to export PDF. If None, shows the plot inline.
    """
    plt.figure(figsize=(10, 5))

    sns.histplot(profit_and_loss, bins=80, kde=True, stat="density",
                 color="lightblue", edgecolor="black", linewidth=0.5)

    plt.axvline(-var, color="black", linestyle="-", linewidth=1.5,
                label=f"VaR ({int(confidence_level * 100)}%)")
    plt.axvline(-es, color="red", linestyle="--", linewidth=1.5,
                label=f"ES ({int(confidence_level * 100)}%)")

    plt.title("Simulated Portfolio P&L Distribution", loc="left", fontsize=13, fontweight="medium")
    plt.legend(loc="upper left", frameon=True, edgecolor="black")
    plt.xlabel("Profit / Loss")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, format="pdf")
        plt.close()
    else:
        buf = BytesIO()
        plt.savefig(buf, format="png", dpi=300)
        buf.seek(0)
        display(display_high_dpi_inline(buf.read(), width=1000))
        plt.close()


# ----------------------------------------------------------
# Simulated Portfolio Path Plot (Static Only)
# ----------------------------------------------------------
def plot_simulated_paths(portfolio_paths, output_path=None):
    """
    Main
    ----
    Plot simulated portfolio value trajectories over time.

    Displays individual paths generated from a Monte Carlo simulation of
    portfolio values. Useful for assessing the dispersion and range of
    simulated future outcomes.

    This is a static only plot.

    Parameters
    ----------
    portfolio_paths : np.ndarray
        Simulated paths of portfolio values (T × N), where T is time steps
        and N is number of simulation paths.
    output_path : str, optional
        File path to export PDF. If None, shows the plot inline.
    """
    num_days, num_paths = portfolio_paths.shape
    sample_paths = min(num_paths, 2500)
    x_axis = np.arange(num_days)

    plt.figure(figsize=(10, 5))
    for i in range(sample_paths):
        plt.plot(x_axis, portfolio_paths[:, i], alpha=0.4)

    plt.xlim(0, num_days - 1)
    plt.title(f"Simulated Portfolio Value Trajectories over {num_days - 1} Days", loc="left", fontsize=13, fontweight="medium")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf")
        plt.close()
    else:
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        display(display_high_dpi_inline(buf.read(), width=1000))
        plt.close()