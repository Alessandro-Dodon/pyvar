"""
Extreme Value Theory (EVT) Module for VaR and ES
------------------------------------------------

This module implements Extreme Value Theory (EVT) risk measures using the 
Peaks Over Threshold (POT) method and the Generalized Pareto Distribution (GPD). 
It provides functions to estimate Value-at-Risk (VaR) and Expected Shortfall (ES) 
based on fitted tail parameters from historical return data.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- fit_evt_parameters: Tail fitting using POT and GPD
- evt_var: EVT-based Value-at-Risk estimation
- evt_es: EVT-based Expected Shortfall estimation

Notes
-----
- All returns are assumed to be daily and in decimal format (e.g., 0.01 = 1%).
- Input series are not automatically cleaned — NaNs are preserved.
  Users are responsible for handling missing values.
- Warnings are issued if NaNs are detected, but no data is dropped by default.
"""

# TODO: double check all formulas 

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import genpareto
import warnings


#----------------------------------------------------------
# EVT Parameter Estimation (Peaks Over Threshold)
#----------------------------------------------------------
def fit_evt_parameters(returns, threshold_percentile=97.5):
    """
    Fit a Generalized Pareto Distribution (GPD) to the tail of the return distribution.

    Applies the Peaks Over Threshold (POT) method to model the extreme losses using GPD.
    This function estimates the shape and scale parameters of the GPD based on the 
    excesses above a high threshold.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    threshold_percentile : float, optional
        Quantile used to define the tail threshold. Default is 97.5.

    Returns
    -------
    params : dict
        Dictionary containing:
        - 'xi': shape parameter of the GPD
        - 'beta': scale parameter of the GPD
        - 'threshold_u': threshold for exceedances
        - 'num_exceedances': number of exceedances above threshold
        - 'total_observations': total number of non-NaN return observations
        - 'losses': transformed loss series (i.e., -returns)

    Raises
    ------
    Warning
        If NaNs are detected in the input series.
    """
    if pd.Series(returns).isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    returns = pd.Series(returns)
    losses = -returns

    threshold_u = np.percentile(losses, threshold_percentile)
    exceedances = losses[losses > threshold_u] - threshold_u

    xi_hat, loc_hat, beta_hat = genpareto.fit(exceedances, floc=0)

    return {
        "xi": xi_hat,
        "beta": beta_hat,
        "threshold_u": threshold_u,
        "num_exceedances": len(exceedances),
        "total_observations": len(losses),
        "losses": losses
    }


#----------------------------------------------------------
# EVT Value-at-Risk (VaR)
#----------------------------------------------------------
def evt_var(returns, confidence_level=0.99, threshold_percentile=97.5, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using Extreme Value Theory (EVT).

    Computes the VaR from a Generalized Pareto Distribution fitted to the tail 
    of the loss distribution using the Peaks Over Threshold (POT) method.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    threshold_percentile : float, optional
        Percentile threshold to define the tail (e.g., 97.5). Default is 97.5.
    wealth : float, optional
        Portfolio value. If provided, VaR is also returned in monetary units.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with columns:
        - 'Returns': original return series
        - 'VaR': constant EVT-based VaR (decimal loss)
        - 'VaR Violation': boolean indicator for VaR breach
        - 'VaR_monetary': optional, VaR scaled by wealth
    """
    params = fit_evt_parameters(returns, threshold_percentile)
    xi = params["xi"]
    beta = params["beta"]
    u = params["threshold_u"]
    n = params["num_exceedances"]
    N = params["total_observations"]
    losses = params["losses"]

    var_evt = u + (beta / xi) * ((N / n * (1 - confidence_level)) ** (-xi) - 1)
    var_series = pd.Series(var_evt, index=losses.index)
    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data


#----------------------------------------------------------
# EVT Expected Shortfall (ES)
#----------------------------------------------------------
def evt_es(result_data, threshold_percentile=97.5, wealth=None):
    """
    Estimate Expected Shortfall (ES) using Extreme Value Theory (EVT).

    Computes the ES from a Generalized Pareto Distribution fitted to the tail 
    of the loss distribution using the Peaks Over Threshold (POT) method.

    This function assumes that result_data was generated by evt_var(...) and
    contains precomputed returns and EVT-based VaR.

    Parameters
    ----------
    result_data : pd.DataFrame
        DataFrame returned by evt_var(), must contain:
        - 'Returns': original return series
        - 'VaR': constant EVT-based VaR
    threshold_percentile : float, optional
        Percentile threshold to define the tail (e.g., 97.5). Default is 97.5.
    wealth : float, optional
        Portfolio value. If provided, ES is also returned in monetary units.

    Returns
    -------
    result_data : pd.DataFrame
        Updated DataFrame with:
        - 'ES': constant EVT-based Expected Shortfall
        - 'ES_monetary': optional, ES scaled by wealth

    Raises
    ------
    KeyError
        If required columns ('Returns', 'VaR') are missing from input.
    """
    if "Returns" not in result_data or "VaR" not in result_data:
        raise KeyError("Input DataFrame must contain 'Returns' and 'VaR' columns from evt_var().")

    returns = result_data["Returns"]
    var_evt = result_data["VaR"].iloc[0]  # Constant across all rows

    params = fit_evt_parameters(returns, threshold_percentile)
    xi = params["xi"]
    beta = params["beta"]
    u = params["threshold_u"]

    es_evt = (var_evt + (beta - xi * u)) / (1 - xi)
    es_series = pd.Series(es_evt, index=returns.index)
    result_data["ES"] = es_series

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    return result_data
