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

#################################################
# Note: double check all names/ titles
#################################################

#----------------------------------------------------------
# VaR Plot Backtesting
#----------------------------------------------------------
def interactive_plot_var(data, subset=None):
    """
    Interactive VaR Backtest Plot.

    Create an interactive Plotly figure showing returns, VaR level, and VaR violations.

    Parameters:
    - data (pd.DataFrame):
        Must contain 'Returns', 'VaR', and 'VaR Violation' columns.
    - subset (tuple, optional):
        (start_date, end_date) to zoom into a specific time range.

    Returns:
    - plotly.graph_objs.Figure:
        Interactive backtest plot with clean formatting.

    Notes:
    - VaR is plotted as a negative threshold line.
    - Violations are shown as red open circles.
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


#----------------------------------------------------------
# ES Plot Backtesting
#----------------------------------------------------------
def interactive_plot_es(data, subset=None):
    """
    Interactive ES Backtest Plot.

    Create an interactive Plotly figure showing returns, VaR, ES level, and VaR violations.

    Parameters:
    - data (pd.DataFrame):
        Must contain 'Returns', 'VaR', 'ES', and 'VaR Violation' columns.
    - subset (tuple, optional):
        (start_date, end_date) to zoom into a specific time range.

    Returns:
    - plotly.graph_objs.Figure:
        Interactive backtest plot with clean formatting.

    Notes:
    - VaR is plotted as a solid black line.
    - ES is plotted as a dashed red line.
    - Violations are shown as red open circles.
    - A zero-return line is added for reference.
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


#----------------------------------------------------------
# Volatility Plot
#----------------------------------------------------------
def interactive_plot_volatility(volatility_series, subset=None):
    """
    Interactive Volatility Plot.

    Create an interactive Plotly figure showing conditional volatility over time.

    Parameters:
    - volatility_series (pd.Series):
        Time series of volatility estimates (decimal format).

    - subset (tuple, optional):
        (start_date, end_date) to zoom into a specific time range.

    Returns:
    - plotly.graph_objs.Figure:
        Interactive volatility plot with clean formatting.

    Notes:
    - Volatility is displayed as a blue line (in %).
    - The background is white with black axis borders.
    - Hover labels show volatility in percentage terms.
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


#----------------------------------------------------------
# Asset Color Map (for portfolio visualization)
#----------------------------------------------------------
def get_asset_color_map(assets):
    """
    Asset Color Mapping.

    Generate a consistent color assignment for a list of assets, cycling through a predefined color palette.

    Parameters:
    - assets (list-like):
        List of asset names (strings).

    Returns:
    - dict:
        Dictionary mapping each asset to a unique color.

    Notes:
    - Colors are drawn cyclically from Plotly's qualitative palette.
    - Ensures consistent asset coloring across plots.
    """
    base_colors = px.colors.qualitative.Plotly
    color_cycle = itertools.cycle(base_colors)
    return {asset: next(color_cycle) for asset in assets}


#----------------------------------------------------------
# VaR and UVaR Plot 
#----------------------------------------------------------
def interactive_plot_var_series(var_df):
    """
    Asset-Normal vs. Undiversified VaR Plot.

    Create an interactive Plotly line chart comparing diversified (Asset-Normal) VaR and 
    Undiversified VaR (ρ = 1) over time.

    Parameters:
    - var_series (pd.Series):
        Time series of diversified Value-at-Risk (VaR).

    - uvar_series (pd.Series):
        Time series of undiversified VaR (full asset correlation assumed).

    Returns:
    - None:
        Displays an interactive plot with both series for comparison.

    Notes:
    - Useful for visualizing the impact of diversification over time.
    - Outputs VaR in monetary units.
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
    return fig


# ----------------------------------------------------------
# Risk Contribution Bar Chart 
# ----------------------------------------------------------
def interactive_plot_risk_contribution_bar(component_df):
    """
    Average Component VaR Contribution Bar Chart.

    Create an interactive horizontal bar chart showing the average absolute Component VaR 
    contribution of each asset to total portfolio risk.

    Parameters:
    - component_df (pd.DataFrame):
        Time series of Component VaR values (shape: T × N).

    Returns:
    - fig (plotly.graph_objs.Figure):
        Interactive bar chart figure.
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
        margin=dict(l=80, r=40, t=50, b=50),
        showlegend=False
    )

    return fig


#----------------------------------------------------------
# Component VaR Over Time 
#----------------------------------------------------------
def interactive_plot_risk_contribution_lines(component_df):
    """
    Component VaR Over Time Line Plot.

    Create an interactive Plotly line chart showing the evolution of each asset's Component VaR 
    contribution over time.

    Parameters:
    - component_df (pd.DataFrame):
        Time series of Component VaR values (shape: T × N).

    Returns:
    - None:
        Displays an interactive line plot with one line per asset.

    Notes:
    - Colors are consistently assigned using a fixed palette.
    - Helps visualize changing risk contributions across time.
    - Hover templates display asset name and Component VaR in monetary units.
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
            hovertemplate="Date: %{x}<br>" + asset + " CVaR = %{y:.2f}<extra></extra>"
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
        yaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True, tickformat="%Y-%m-%d"),
        margin=dict(l=60, r=60, t=50, b=50)
    )

    return fig


#----------------------------------------------------------
# Correlation Matrix Heatmap
#----------------------------------------------------------
def interactive_plot_correlation_matrix(position_data):
    """
    Interactive Correlation Matrix Heatmap.

    Create an interactive heatmap visualization of the asset return correlations, 
    showing only the lower triangle of the matrix with a diverging red-blue color scheme.

    Description:
    - Computes returns from monetary positions:
        - Log returns if all positions are positive.
        - Percentage returns if any position is negative (e.g., due to short selling).
    - Masks the upper triangle to reduce redundancy.
    - White gridlines and no visible axes for a clean heatmap appearance.

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary positions (shape: T × N).

    Returns:
    - fig (plotly.graph_objects.Figure):
        Interactive heatmap figure ready for display.

    Notes:
    - Correlations range from -1 (perfect negative) to +1 (perfect positive).
    - Hover tooltips show asset pairs and their correlation values.
    """
    position_data = pd.DataFrame(position_data).dropna()

    # Use log returns if all values are positive
    if (position_data <= 0).any().any():
        returns = position_data.pct_change().dropna()
    else:
        returns = np.log(position_data / position_data.shift(1)).dropna()

    corr_matrix = returns.corr()
    asset_names = corr_matrix.columns.tolist()

    # Mask upper triangle
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
        title="Correlation Matrix (Returns)",
        width=900,
        height=800,
        template="simple_white",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=80, r=80, t=80, b=80),  # extra margin to avoid axis bleeding
        xaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False
        ),
        yaxis=dict(
            visible=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            autorange="reversed"
        )
    )

    # White gridlines between cells
    for i in range(len(asset_names)):
        fig.add_shape(type="line",
                      x0=-0.5, x1=len(asset_names) - 0.5,
                      y0=i - 0.5, y1=i - 0.5,
                      line=dict(color="white", width=1),
                      xref="x", yref="y")
        fig.add_shape(type="line",
                      x0=i - 0.5, x1=i - 0.5,
                      y0=-0.5, y1=len(asset_names) - 0.5,
                      line=dict(color="white", width=1),
                      xref="x", yref="y")

    return fig


#----------------------------------------------------------
# General Plot Caller Function (Notebook + High-Res Export)
#----------------------------------------------------------
def display_high_dpi_inline(png_bytes, width):
    """
    Display a high-DPI PNG inline in Jupyter without losing resolution.
    """
    encoded = base64.b64encode(png_bytes).decode('utf-8')
    return HTML(f'<img src="data:image/png;base64,{encoded}" style="width:{width}px;"/>')


def plot_caller(plot_type, interactive=True, output_path=None, **kwargs):
    """
    General Plot Caller.

    Displays a Plotly plot interactively or as a static high-res PNG (inline or saved).

    Parameters:
    - plot_type (str): One of:
        'var', 'es', 'volatility', 'var_series', 
        'risk_bar', 'risk_lines', 'correlation'.

    - interactive (bool): If True, show interactive plot (default).
                          If False, render or save high-resolution static plot.

    - output_path (str or None): If set and interactive=False, save high-res PNG to this path.

    - **kwargs: Passed to the selected plotting function.

    Returns:
    - None (renders or saves the figure)
    """
    plot_functions = {
    "backtest_var": interactive_plot_var,
    "backtest_es": interactive_plot_es,
    "volatility": interactive_plot_volatility,
    "compare_var_series": interactive_plot_var_series,
    "component_var_bar": interactive_plot_risk_contribution_bar,
    "component_var_lines": interactive_plot_risk_contribution_lines,
    "correlation_matrix": interactive_plot_correlation_matrix
    }

    if plot_type not in plot_functions:
        raise ValueError(f"Invalid plot_type: '{plot_type}'. "
                         f"Choose from {list(plot_functions.keys())}.")

    fig = plot_functions[plot_type](**kwargs)

    # Use figure's own layout dimensions, fallback defaults if missing
    width = fig.layout.width or 1000
    height = fig.layout.height or 500
    scale = 4  # Very high resolution for export

    if interactive:
        fig.show()
    elif output_path:
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))

    return None
