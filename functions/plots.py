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
# Note: output path should work only with static images
#################################################

# TODO: if there is the output to the plot, make another plot with
# interactive = False or force False and export

#----------------------------------------------------------
# Display helper 
#----------------------------------------------------------
def display_high_dpi_inline(png_bytes, width):
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return HTML(f'<img src="data:image/png;base64,{encoded}" style="width:{width}px;"/>')


#----------------------------------------------------------
# Backtesting Plot for VaR (and optionally ES)
#----------------------------------------------------------
def plot_backtest(data, subset=None, interactive=True, output_path=None):
    """
    Adaptive backtest plot for Value-at-Risk and optionally Expected Shortfall.

    Automatically chooses bar or line plots for returns based on data length.
    Adds VaR line and violations. If 'ES' column exists, includes ES line.
    Zero line is shown only when bars are used.

    Parameters:
    - data (pd.DataFrame): Must contain 'Returns', 'VaR', and 'VaR Violation'. 
                           Optionally can contain 'ES'.
    - subset (tuple, optional): (start_date, end_date) for time window selection.
    - interactive (bool): Show interactive plot (True) or export static PNG (False).
    - output_path (str, optional): File path to save PNG if interactive=False.

    Returns:
    - fig (plotly.graph_objs.Figure): Plotly figure object.
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
        xaxis=dict(showline=True, linewidth=1, linecolor="black", mirror=True),
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
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))

    return fig


#----------------------------------------------------------
# Volatility Plot
#----------------------------------------------------------
def plot_volatility(volatility_series, subset=None, interactive=True, output_path=None):
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
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)

    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


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
def plot_var_series(var_df, interactive=True, output_path=None):
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
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


# ----------------------------------------------------------
# Risk Contribution Bar Chart 
# ----------------------------------------------------------
def plot_risk_contribution_bar(component_df, interactive=True, output_path=None):
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
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


#----------------------------------------------------------
# Component VaR Over Time 
#----------------------------------------------------------
def plot_risk_contribution_lines(component_df, interactive=True, output_path=None):
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
        fig.write_image(output_path, format="png", width=width, height=height, scale=scale)
    else:
        png_bytes = to_image(fig, format="png", width=width, height=height, scale=scale)
        display(display_high_dpi_inline(png_bytes, width))


#----------------------------------------------------------
# Correlation Matrix Heatmap
#----------------------------------------------------------
def plot_correlation_matrix(position_data, interactive=True, output_path=None):
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

    # Compute returns
    if (position_data <= 0).any().any():
        returns = position_data.pct_change().dropna()
    else:
        returns = np.log(position_data / position_data.shift(1)).dropna()

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
        title="Correlation Matrix (Returns)",
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


