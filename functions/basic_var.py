#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd

#################################################
# Note: double check all formulas 
#################################################

# TODO: violations is not needed! can be put into the graph directly!

#----------------------------------------------------------
# Historical VaR (Non-Parametric)
#----------------------------------------------------------
def var_historical(returns, confidence_level=0.99, holding_period=1, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using historical (non-parametric) simulation.

    Computes VaR directly from the empirical distribution of past returns without 
    assuming a specific distribution. The result is scaled for the holding period 
    using the square-root-of-time rule.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): VaR horizon in days (default: 1).
    - wealth (float, optional): Portfolio value. If set, VaR outputs are in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'VaR' (estimated constant VaR, decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (optional, if wealth is provided)
    """
    var_cutoff = np.percentile(returns.dropna(), 100 * (1 - confidence_level))
    scaled_var = np.sqrt(holding_period) * var_cutoff
    var_series = pd.Series(-scaled_var, index=returns.index)

    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data.dropna()


#----------------------------------------------------------
# Parametric VaR 
#----------------------------------------------------------
def var_parametric(returns, confidence_level=0.99, holding_period=1, distribution="normal", wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a parametric distribution.

    Fits a specified distribution (Normal, Student-t, or GED) to the return series, 
    then computes VaR as the corresponding quantile scaled by volatility and the 
    square-root-of-time adjustment.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - holding_period (int): VaR horizon in days (default: 1).
    - distribution (str): Distribution assumed for returns ("normal", "t", or "ged").
    - wealth (float, optional): Portfolio value. If set, VaR outputs are in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'VaR' (estimated constant VaR, decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (optional, if wealth is provided)

    Raises:
    - ValueError: If an unsupported distribution is passed.
    """
    returns_clean = returns.dropna()

    match distribution:
        case "normal":
            std_dev = returns_clean.std()
            quantile = norm.ppf(1 - confidence_level)
            scaled_std = std_dev * np.sqrt(holding_period)
        case "t":
            df, loc, scale = t.fit(returns_clean)
            quantile = t.ppf(1 - confidence_level, df)
            scaled_std = scale * np.sqrt(holding_period)
        case "ged":
            beta, loc, scale = gennorm.fit(returns_clean)
            quantile = gennorm.ppf(1 - confidence_level, beta)
            scaled_std = scale * np.sqrt(holding_period)
        case _:
            raise ValueError("Supported distributions: 'normal', 't', 'ged'")

    var_value = -quantile * scaled_std
    var_series = pd.Series(var_value, index=returns.index)
    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data.dropna()