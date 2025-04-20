import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings



# Asset Normal VaR and Undiversified VaR
def var_asset_normal(
    position_data,
    confidence_level,
    holding_period=1,
    undiversified=False
):
    """
    Compute Asset-Normal Parametric VaR using a time series of monetary positions.

    If `undiversified=True`, computes the Undiversified VaR assuming full correlation (ρ = 1),
    i.e., the sum of individual VaRs per asset.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Time series of monetary values invested in each asset (shape: T x N).
        Each row is a day, each column is an asset.
        Must contain at least two rows to compute the covariance matrix.

    confidence_level : float
        Confidence level for VaR, e.g., 0.99 for 99% VaR.

    holding_period : int, default = 1
        VaR horizon in days.

    undiversified : bool, default = False
        If True, compute Undiversified VaR (sum of individual VaRs under ρ = 1).

    Returns
    -------
    pd.Series
        Daily series of VaR estimates (diversified or undiversified), indexed by date.
    """
    # Convert to DataFrame if needed
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()

    sigma_matrix = returns_data.cov().values
    z_score = norm.ppf(1 - confidence_level)

    positions = position_data.loc[returns_data.index].values
    var_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)

        if undiversified:
            std_devs = np.sqrt(np.diag(sigma_matrix)).reshape(-1, 1)
            var_i = z_score * std_devs * x
            var_t = np.abs(var_i.sum()) * np.sqrt(holding_period)
        else:
            variance = float(x.T @ sigma_matrix @ x)
            std_dev = np.sqrt(variance)
            var_t = np.abs(z_score * std_dev * np.sqrt(holding_period))

        var_list.append(var_t)

    return pd.Series(var_list, index=returns_data.index)



# Marginal VaR using portfolio VaR and consistent scaling
def marginal_var(
    position_data,
    confidence_level,
    holding_period=1
):
    """
    Compute Marginal VaR (ΔVaR) for each asset in a portfolio over time,
    using the same covariance structure and portfolio VaR logic as var_asset_normal.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Time series of monetary holdings per asset (T x N).
        Each row is a day, each column is an asset.

    confidence_level : float
        Confidence level for VaR (e.g., 0.99).

    holding_period : int, default = 1
        VaR horizon in days.

    Returns
    -------
    pd.DataFrame
        Time series of Marginal VaRs per asset (T x N), in monetary units.
    """
    # Ensure proper format
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    # Get AN VaR from the existing function
    portfolio_var_series = var_asset_normal(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        undiversified=False
    )

    # Align data
    positions = position_data.loc[returns_data.index].values
    marginal_var_list = []

    for xt, var_t in zip(positions, portfolio_var_series.values):
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            delta_var = np.zeros_like(x.flatten())
        else:
            # Compute beta weights (Σx / x'Σx)
            beta_vector = (sigma_matrix @ x).flatten() / variance
            delta_var = var_t * beta_vector  # Monetary marginal VaR

        marginal_var_list.append(delta_var)

    return pd.DataFrame(
        marginal_var_list,
        index=returns_data.index,
        columns=position_data.columns
    )



# Component VaR (using marginal VaR)
def component_var(position_data, confidence_level, holding_period=1):
    """
    Compute Component VaR = x_i * ΔVaR_i for each asset and day.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Monetary positions (T x N)
    confidence_level : float
    holding_period : int

    Returns
    -------
    pd.DataFrame
        Component VaR (T x N) in monetary units
    """
    position_data = pd.DataFrame(position_data)
    marginal_df = marginal_var(position_data, confidence_level, holding_period)
    component_df = position_data.loc[marginal_df.index] * marginal_df
    return component_df



# Relative Component VaR
def relative_component_var(position_data, confidence_level, holding_period=1):
    """
    Compute Relative Component VaR = CVaR_i / VaR_total for each asset and day.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Monetary positions (T x N)
    confidence_level : float
    holding_period : int

    Returns
    -------
    pd.DataFrame
        Relative CVaR (T x N), values between 0 and 1
    """
    position_data = pd.DataFrame(position_data)
    
    # Get components
    cvar_df = component_var(position_data, confidence_level, holding_period)
    var_series = var_asset_normal(position_data, confidence_level, holding_period)
    
    # Divide each row by the total VaR of that day
    rcvar_df = cvar_df.div(var_series, axis=0)
    
    return rcvar_df



# Incremental VaR
def incremental_var(position_data, change_vector, confidence_level, holding_period=1):
    """
    Compute Incremental VaR for a specified change in the portfolio.

    Formula (first-order approx):
        IVaR = ΔVaR' × a

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Monetary positions (T x N)
    change_vector : list or np.ndarray of shape (N,)
        Change in holdings (same units as position_data)
    confidence_level : float
    holding_period : int

    Returns
    -------
    pd.Series
        Time series of Incremental VaR values (1 per day)
    """
    position_data = pd.DataFrame(position_data)
    marginal_df = marginal_var(position_data, confidence_level, holding_period)

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    ivar_series = marginal_df @ a
    return ivar_series



# Plotting functions
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import itertools



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
