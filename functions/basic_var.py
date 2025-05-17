"""
Basic VaR and Expected Shortfall Estimation Module
--------------------------------------------

Provides functions to compute Value-at-Risk (VaR) and Expected Shortfall (ES)
using both non-parametric (historical) and parametric methods (Normal and Student-t distributions).

Authors
------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- historical_var: Historical simulation-based VaR
- historical_es: Historical ES based on tail mean
- parametric_var: Parametric VaR using Normal or Student-t distributions
- parametric_es: Parametric ES using Normal or Student-t distributions

Notes
-----
- All returns are assumed to be daily and in decimal format (e.g., 0.01 = 1%).
"""

# TODO: violations is not needed! can be put into the graph directly!
# TODO: double check all formulas 

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t
import numpy as np
import pandas as pd


#----------------------------------------------------------
# Historical VaR (Non-Parametric)
#----------------------------------------------------------
def historical_var(returns, confidence_level=0.99, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using historical (non-parametric) simulation.

    Computes the VaR from the empirical distribution of past returns without assuming
    a specific distributional form. The method is based purely on observed daily returns.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    wealth : float, optional
        Portfolio value in monetary units. If provided, a monetary VaR is also returned.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'VaR': constant VaR (decimal loss)
        - 'VaR Violation': True if loss exceeded VaR on a given day
        - 'VaR_monetary': optional, VaR scaled by wealth if provided
    """
    var_cutoff = np.percentile(returns, 100 * (1 - confidence_level))
    var_series = pd.Series(-var_cutoff, index=returns.index)

    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data


#----------------------------------------------------------
# Historical Expected Shortfall (Tail Mean)
#----------------------------------------------------------
def historical_es(result_data, wealth=None): 
    """
    Estimate Expected Shortfall (ES) using historical returns below the VaR threshold.

    Computes ES by averaging the returns that fall below the negative VaR level.
    Assumes that the VaR column is already computed and constant over time.
    The ES is reported as a constant loss level and optionally scaled by portfolio wealth.

    Parameters
    ----------
    result_data : pd.DataFrame
       DataFrame returned by the VaR estimation function (historical_var),
       to which the ES estimate will be added.
    wealth : float, optional
        Portfolio value in monetary units. If provided, ES is also returned in monetary terms.

    Returns
    -------
    result_data : pd.DataFrame
        Extended DataFrame with:
        - 'ES': constant Expected Shortfall (decimal loss)
        - 'ES_monetary': optional, ES scaled by wealth if provided
    """
    var_threshold = result_data["VaR"].iloc[0]
    tail_returns = result_data["Returns"][result_data["Returns"] < -var_threshold]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = tail_returns.mean()

    es_series = pd.Series(-es_value, index=result_data.index)
    result_data["ES"] = es_series

    if wealth is not None:
        result_data["ES_monetary"] = es_series * wealth

    return result_data


# ----------------------------------------------------------
# Parametric VaR (Normal or Student-t)
# ----------------------------------------------------------
def parametric_var(returns, confidence_level=0.99, distribution="normal", wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a parametric distribution.

    Fits a Normal or Student-t distribution to the return series and computes
    1-day VaR as the left-tail quantile. VaR is scaled to return volatility 
    and optionally converted to monetary loss if portfolio value is provided.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    distribution : {"normal", "t"}, optional
        Distribution to fit for quantile estimation. Default is "normal".
    wealth : float, optional
        Portfolio value in monetary units. If provided, VaR is also returned in monetary terms.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'VaR': estimated 1-day VaR (decimal loss)
        - 'VaR Violation': boolean flag for when returns exceed VaR
        - 'VaR_monetary': optional, monetary VaR if wealth is provided

    Raises
    ------
    ValueError
        If an unsupported distribution is specified.

    Notes
    -----
    If VaR is required for a longer horizon (e.g., h days),
    scale the reported VaR by √h.
    """
    match distribution:
        case "normal":
            std_dev = returns.std()
            quantile = norm.ppf(1 - confidence_level)
            scaled_std = std_dev
        case "t":
            df, loc, scale = t.fit(returns)
            quantile = t.ppf(1 - confidence_level, df)
            scaled_std = scale
        case _:
            raise ValueError("Supported distributions: 'normal', 't'")

    var_value = -quantile * scaled_std
    var_series = pd.Series(var_value, index=returns.index)
    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data


# ----------------------------------------------------------
# Parametric Expected Shortfall (Normal or Student-t)
# ----------------------------------------------------------
def parametric_es(result_data, confidence_level, distribution="normal", wealth=None):
    """
    Estimate Expected Shortfall (ES) using a parametric distribution.

    Computes 1-day ES as the conditional expectation of losses beyond the VaR threshold.
    Supports both Normal and Student-t distributions. Assumes the input DataFrame
    includes the return series (column 'Returns').

    Parameters
    ----------
    result_data : pd.DataFrame
        DataFrame returned by the VaR estimation function (parametric_var),
        containing the 'Returns' column.
    confidence_level : float
        Confidence level for ES (e.g., 0.99).
    distribution : {"normal", "t"}, optional
        Distribution to fit for tail expectation. Default is "normal".
    wealth : float, optional
        Portfolio value in monetary units. If provided, ES is also returned in monetary terms.

    Returns
    -------
    result_data : pd.DataFrame
        Updated DataFrame with:
        - 'ES': constant 1-day Expected Shortfall (decimal loss)
        - 'ES_monetary': optional, monetary ES if wealth is provided

    Raises
    ------
    ValueError
        If an unsupported distribution is specified.

    Notes
    -----
    If ES is needed over a longer horizon (e.g., h days),
    scale the reported ES by √h.
    """
    returns = result_data["Returns"]
    alpha = confidence_level

    if distribution == "normal":
        std_dev = returns.std()
        z = norm.ppf(alpha)
        es_value = std_dev * norm.pdf(z) / (1 - alpha)

    elif distribution == "t":
        df, loc, scale = t.fit(returns)
        t_alpha = t.ppf(alpha, df)
        pdf_val = t.pdf(t_alpha, df)
        factor = (df + t_alpha**2) / (df - 1)
        es_value = scale * pdf_val * factor / (1 - alpha)

    else:
        raise ValueError("Supported distributions: 'normal', 't'")

    result_data["ES"] = pd.Series(es_value, index=result_data.index)

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    return result_data
