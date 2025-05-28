"""
Analytic VaR Module
-------------------

Provides analytical tools to compute and decompose Value-at-Risk (VaR) and Expected Shortfall (ES)
under the assumption of normally distributed portfolio assets.

Uses the asset-normal approximation to derive closed-form expressions for portfolio risk,
based on positions and the covariance matrix of returns.

Supports full risk decomposition into marginal, component, relative, and incremental contributions.
Assumes a static covariance structure and a buy-and-hold strategy.

Risk estimates should be updated if portfolio weights change significantly.
Portfolio-level VaR and ES can also be obtained using the parametric method in basic_var.py.

Some functions are internally linked and call each other when needed, allowing users to directly 
invoke the desired VaR or ES function without computing the entire set of risk measures.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
---------
- asset_normal_var: Diversified and undiversified VaR with diversification benefit
- marginal_var / marginal_es: Marginal contributions to VaR or ES
- component_var / component_es: Component contributions (position × marginal)
- relative_component_var / relative_component_es: Percentage contribution to total VaR or ES
- incremental_var / incremental_es: Impact of hypothetical changes in position size
"""


#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


#----------------------------------------------------------
# Asset-Normal VaR with Diversification Benefit
#----------------------------------------------------------
def asset_normal_var(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate portfolio Value-at-Risk (VaR) using the asset-normal approach.

    Computes both diversified and undiversified portfolio VaR under the normality assumption. 
    Diversified VaR uses the full covariance matrix of asset returns, while undiversified VaR 
    assumes perfect positive correlation between all assets (ρ = 1). The diversification benefit 
    is the difference between the two.

    Parameters
    ----------
    position_data : pd.DataFrame 
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date, with columns:
        - 'Diversified_VaR': Portfolio VaR using correlations (monetary amount)
        - 'Undiversified_VaR': VaR assuming full correlation (monetary amount)
        - 'Diversification_Benefit': Difference between undiversified and diversified VaR

    Raises
    ------
    ValueError
        If fewer than two observations are available after cleaning.
    """
    position_data = pd.DataFrame(position_data).dropna()

    if position_data.shape[0] < 2:
        raise ValueError("At least two time steps are required.")

    returns = position_data.pct_change().dropna()
    cov_matrix = returns.cov().values
    z_score = norm.ppf(1 - confidence_level)
    scale = np.sqrt(holding_period)

    diversified = []
    undiversified = []

    for positions in position_data.loc[returns.index].values:
        x = positions.reshape(-1, 1)

        var_diversified = abs(z_score * np.sqrt(float(x.T @ cov_matrix @ x)) * scale)
        var_undiversified = abs(z_score * np.sum(np.sqrt(np.diag(cov_matrix)) * positions) * scale)

        diversified.append(var_diversified)
        undiversified.append(var_undiversified)

    result_data = pd.DataFrame({
        "Diversified_VaR": diversified,
        "Undiversified_VaR": undiversified,
        "Diversification_Benefit": np.array(undiversified) - np.array(diversified)
    }, index=returns.index)

    return result_data


#----------------------------------------------------------
# Marginal VaR (Delta VaR per Asset)
#----------------------------------------------------------
def marginal_var(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Marginal Value-at-Risk (VaR) for each asset in a portfolio.

    Computes the marginal contribution of each asset to the total diversified 
    portfolio VaR. 

    Parameters
    ----------
    position_data : pd.DataFrame 
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of marginal VaR values (T × N).

    Raises
    ------
    ValueError
        If fewer than two time steps are available.
    """
    position_data = pd.DataFrame(position_data).dropna()

    if position_data.shape[0] < 2:
        raise ValueError("At least two time steps are required.")

    returns = position_data.pct_change().dropna()
    cov_matrix = returns.cov().values
    positions = position_data.loc[returns.index].values

    # Compute diversified VaR for each day
    diversified_var_series = asset_normal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )["Diversified_VaR"].values

    marginal_var_list = []

    for x_t, var_t in zip(positions, diversified_var_series):
        x = x_t.reshape(-1, 1)
        variance = float(x.T @ cov_matrix @ x)

        if variance <= 1e-10:
            delta_var = np.zeros_like(x.flatten())
        else:
            beta_vector = (cov_matrix @ x).flatten() / variance
            delta_var = var_t * beta_vector

        marginal_var_list.append(delta_var)

    result_data = pd.DataFrame(
        marginal_var_list,
        index=returns.index,
        columns=position_data.columns
    )

    return result_data


#----------------------------------------------------------
# Component VaR (via marginal VaR)
#----------------------------------------------------------
def component_var(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Component Value-at-Risk (VaR) for each asset in a portfolio.

    Computes the contribution of each asset to total diversified VaR using 
    the following formula: Component VaR = position × marginal VaR.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of Component VaR values (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)

    # Compute Marginal VaR
    marginal_df = marginal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    # Align positions with marginal VaR index
    aligned_positions = position_data.loc[marginal_df.index]

    # Compute Component VaR
    component_df = aligned_positions * marginal_df

    return component_df


#----------------------------------------------------------
# Relative Component VaR
#----------------------------------------------------------
def relative_component_var(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Relative Component Value-at-Risk (VaR) for each asset.

    Computes the share of total diversified VaR contributed by each asset at 
    each time step. Row-wise values sum to 1.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of relative Component VaR values (T × N), expressed as decimals 
        representing percentage contributions (e.g., 0.25 = 25%).
    """
    position_data = pd.DataFrame(position_data)

    # Compute Component VaR and total portfolio VaR
    component_df = component_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )
    total_var_series = asset_normal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )["Diversified_VaR"]

    # Compute relative contributions
    relative_df = component_df.div(total_var_series, axis=0)

    return relative_df


#----------------------------------------------------------
# Incremental VaR
#----------------------------------------------------------
def incremental_var(position_data, change_vector, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Incremental Value-at-Risk (VaR) from a position change.

    Computes the daily impact on total diversified VaR from a proposed change 
    in asset positions using first-order approximation via Marginal VaR.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    change_vector : array-like
        Monetary change in asset positions (length N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.Series
        Time series of Incremental VaR values in monetary units.
    
    Raises
    ------
    ValueError
        If length of change_vector does not match number of assets.
    """
    position_data = pd.DataFrame(position_data)
    marginal_df = marginal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    incremental_var_series = marginal_df @ a
    return incremental_var_series


#----------------------------------------------------------
# Marginal Expected Shortfall (Delta ES per Asset)
#----------------------------------------------------------
def marginal_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Marginal Expected Shortfall (ES) for each asset in a portfolio.

    Computes the marginal contribution of each asset to the total diversified 
    portfolio Expected Shortfall. Assumes normally 
    distributed returns and a buy-and-hold portfolio strategy.

    Parameters
    ----------
    position_data : pd.DataFrame or np.ndarray
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of marginal ES values (T × N).

    Raises
    ------
    ValueError
        If fewer than two time steps are available.
    """
    position_data = pd.DataFrame(position_data).dropna()

    if position_data.shape[0] < 2:
        raise ValueError("At least two time steps are required.")

    returns = position_data.pct_change().dropna()
    cov_matrix = returns.cov().values
    positions = position_data.loc[returns.index].values

    z = norm.ppf(confidence_level)
    scaling = norm.pdf(z) / (1 - confidence_level)
    marginal_es_list = []

    for x_t in positions:
        x = x_t.reshape(-1, 1)
        variance = float(x.T @ cov_matrix @ x)

        if variance <= 1e-10:
            delta_es = np.zeros_like(x.flatten())
        else:
            std_dev = np.sqrt(variance)
            portfolio_es = scaling * std_dev * np.sqrt(holding_period)
            beta_vector = (cov_matrix @ x).flatten() / variance
            delta_es = portfolio_es * beta_vector

        marginal_es_list.append(delta_es)

    result_data = pd.DataFrame(
        marginal_es_list,
        index=returns.index,
        columns=position_data.columns
    )

    return result_data


#----------------------------------------------------------
# Component Expected Shortfall (ES)
#----------------------------------------------------------
def component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Component Expected Shortfall (ES) for each asset in a portfolio.

    Computes the contribution of each asset to total portfolio ES using 
    the following formula: Component ES = position × marginal ES.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of Component ES values (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)

    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    aligned_positions = position_data.loc[marginal_df.index]
    component_df = aligned_positions * marginal_df

    return component_df


#----------------------------------------------------------
# Relative Component Expected Shortfall (ES)
#----------------------------------------------------------
def relative_component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Relative Component Expected Shortfall (ES) for each asset.

    Computes the share of total portfolio ES contributed by each asset at 
    each time step. Row-wise values sum to 1.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.DataFrame
        Time series of relative Component ES values (T × N), expressed as decimals 
        representing percentage contributions (e.g., 0.25 = 25%).
    """
    position_data = pd.DataFrame(position_data)

    component_df = component_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    total_es = component_df.sum(axis=1)
    relative_df = component_df.div(total_es, axis=0)

    return relative_df


#----------------------------------------------------------
# Incremental Expected Shortfall (ES)
#----------------------------------------------------------
def incremental_es(position_data, change_vector, confidence_level=0.99, holding_period=1):
    """
    Main
    ----
    Estimate Incremental Expected Shortfall (ES) from a position change.

    Computes the daily change in portfolio ES due to a change in asset positions 
    using first-order approximation via Marginal ES.

    Parameters
    ----------
    position_data : pd.DataFrame
        Monetary positions over time (T × N).
    change_vector : array-like
        Monetary change in asset positions (length N).
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.
    holding_period : int, optional
        Holding period in days. Default is 1.

    Returns
    -------
    pd.Series
        Time series of Incremental ES values in monetary units.

    Raises
    ------
    ValueError
        If length of change_vector does not match number of assets.
    """
    position_data = pd.DataFrame(position_data)

    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    incremental_series = marginal_df @ a
    return incremental_series
