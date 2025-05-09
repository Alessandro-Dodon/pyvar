#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
from IPython.display import display

#################################################
# Note: double check all formulas and inputs (x)
#################################################
# Note2: backtesting VaR AN? good idea or not?
#################################################

#----------------------------------------------------------
# Asset-Normal VaR with Diversification Benefit
#----------------------------------------------------------
def var_asset_normal(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Value-at-Risk (VaR) using the asset-normal approach with diversification.

    Computes diversified and undiversified portfolio VaR under the normality assumption.
    Returns a DataFrame with time series of both measures and the diversification benefit.

    Parameters:
    - position_data (pd.DataFrame or np.ndarray): Monetary positions (T × N).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): VaR horizon in days (default = 1).

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Diversified_VaR' (decimal)
        - 'Undiversified_VaR' (decimal)
        - 'Diversification_Benefit' (decimal)
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

        var_div = abs(z_score * np.sqrt(float(x.T @ cov_matrix @ x)) * scale)
        var_undiv = abs(z_score * np.sum(np.sqrt(np.diag(cov_matrix)) * positions) * scale)

        diversified.append(var_div)
        undiversified.append(var_undiv)

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
    Estimate Marginal Value-at-Risk (VaR) for each asset using the asset-normal approach.

    Computes time-varying marginal contributions of each asset to the total portfolio VaR 
    based on the full covariance matrix and a buy-and-hold portfolio strategy.

    Parameters:
    - position_data (pd.DataFrame or np.ndarray): Monetary positions (T × N).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): VaR horizon in days (default = 1).

    Returns:
    - result_data (pd.DataFrame): Marginal VaR contributions (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data).dropna()

    if position_data.shape[0] < 2:
        raise ValueError("At least two time steps are required.")

    returns = position_data.pct_change().dropna()
    cov_matrix = returns.cov().values
    positions = position_data.loc[returns.index].values

    # Compute diversified VaR for each day
    diversified_var_series = var_asset_normal(
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
    Estimate Component Value-at-Risk (VaR) using marginal VaR decomposition.

    Computes the contribution of each asset to the total diversified portfolio VaR 
    using Euler decomposition. This equals the product of each asset's position 
    and its marginal VaR.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of Component VaRs (T × N), in monetary units.
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
    Estimate Relative Component Value-at-Risk (VaR).

    Computes the proportionate contribution of each asset to the total diversified portfolio VaR,
    based on the Euler decomposition. Values sum to 1 across assets at each time step.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of Relative Component VaRs (T × N), as percentages of total VaR.
    """
    position_data = pd.DataFrame(position_data)

    # Compute Component VaR and total portfolio VaR
    component_df = component_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )
    total_var_series = var_asset_normal(
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
    Estimate Incremental Value-at-Risk (VaR).

    Computes the impact on total portfolio VaR from a change in asset positions using Marginal VaR.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - change_vector (array-like): Change in positions (length N).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.Series: Time series of Incremental VaR (1 per day), in monetary units.
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

    return marginal_df @ a

