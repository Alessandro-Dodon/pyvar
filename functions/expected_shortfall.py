#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd
import warnings # for new ES functions
from IPython.display import display # for new ES functions

#################################################
# Note: double check all formulas 
#       and ES for vol should also
#       return the ES for the next day? 
#################################################
# Note2: for the portfolio metrics merge
#         the ES and VaR functions into one?
#################################################




#----------------------------------------------------------
# Marginal Expected Shortfall (ES)
#----------------------------------------------------------
def marginal_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Marginal Expected Shortfall (ES) for each asset in a portfolio.

    Computes marginal ES using Euler decomposition under a normal distribution 
    assumption. Reflects each asset's sensitivity to portfolio tail risk.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of marginal ES (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)
    if position_data.shape[0] < 2:
        warnings.warn("At least two rows of data are required.")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    alpha = confidence_level
    z = norm.ppf(alpha)
    es_scaling = norm.pdf(z) / (1 - alpha)

    positions = position_data.loc[returns_data.index].values
    marginal_es_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            marginal = np.zeros_like(x.flatten())
        else:
            std_dev = np.sqrt(variance)
            portfolio_es = es_scaling * std_dev * np.sqrt(holding_period)
            beta_vector = (sigma_matrix @ x).flatten() / variance
            marginal = portfolio_es * beta_vector

        marginal_es_list.append(marginal)

    return pd.DataFrame(
        marginal_es_list,
        index=returns_data.index.strftime("%Y-%m-%d"),
        columns=position_data.columns
    )


#----------------------------------------------------------
# Component Expected Shortfall (ES)
#----------------------------------------------------------
def component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Component Expected Shortfall (ES) for each asset in a portfolio.

    Computes component ES using Euler decomposition: 
    marginal ES × monetary position.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of component ES (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)

    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    if isinstance(position_data.index[0], pd.Timestamp):
        position_data.index = position_data.index.strftime("%Y-%m-%d")

    aligned_positions = position_data.loc[marginal_df.index]
    component_df = aligned_positions * marginal_df

    return component_df


#----------------------------------------------------------
# Relative Component Expected Shortfall (ES)
#----------------------------------------------------------
def relative_component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Relative Component Expected Shortfall (ES) for each asset.

    Computes the share of total ES each asset contributes at each time step:
    relative ES = component ES / total ES.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of relative component ES (T × N), values sum to 1.
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
    Estimate Incremental Expected Shortfall (ES) for a portfolio.

    Computes how a change in positions affects total ES using Marginal ES 
    and a first-order approximation.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions (T × N).
    - change_vector (list or np.ndarray): Change in positions (length N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Horizon in days (default: 1).

    Returns:
    - pd.Series: Time series of Incremental ES values in monetary units.
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

    ies_series = marginal_df @ a
    return ies_series
