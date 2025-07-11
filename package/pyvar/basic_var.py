"""
Basic VaR and ES Estimation Module
----------------------------------

Provides functions to compute Value-at-Risk (VaR) and Expected Shortfall (ES) using the most
basic methods, both non-parametric (historical) and parametric methods (Normal and Student-t distributions).

All the following methods require simplifying assumptions, like iid returns, and don't consider
time-varying volatility or correlations. 

Authors
------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- historical_var: Historical or non-parametric VaR
- historical_es: Historical ES based on empirical tail mean
- parametric_var: Parametric VaR using Normal or Student-t distributions
- parametric_es: Parametric ES using Normal or Student-t distributions
- hybrid_var: Hybrid VaR using Exponentially Weighted Historical method
- hybrid_es: Hybrid ES using Exponentially Weighted Historical method
- cornish_fisher_var: Cornish-Fisher adjusted VaR based on Normal quantile
"""


#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


#----------------------------------------------------------
# Historical VaR (Non-Parametric)
#----------------------------------------------------------
def historical_var(returns, confidence_level=0.99, wealth=None):
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using the historical (non-parametric) method.

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

    Notes
    -----
    - This function estimates 1-day VaR. For other horizons like weekly or monthly, different data must be used.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")

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
    Main
    ----
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

    Notes
    -----
    - This function estimates 1-day VaR. For other horizons like weekly or monthly, different data must be used.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")

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
# Hybrid VaR (Exponentially Weighted Historical)
# ----------------------------------------------------------
def hybrid_var(returns, confidence_level=0.99, lambda_decay=0.94, wealth=None):
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using the Hybrid (Exponentially Weighted Historical) method.

    This method applies exponentially decaying weights to past returns to form a weighted empirical 
    distribution and compute the corresponding quantile (VaR).

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    lambda_decay : float, optional
        Exponential decay factor in (0, 1). Default is 0.94.
    wealth : float, optional
        Portfolio value in monetary units. If provided, monetary VaR is returned.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with:
        - 'Returns': original return series
        - 'VaR': weighted historical VaR (decimal loss)
        - 'VaR Violation': boolean flag for violations
        - 'VaR_monetary': optional, monetary VaR if wealth is provided

    Raises
    ------
    ValueError
        If lambda is not in (0, 1).

    Notes
    -----
    - This is a 1-day VaR estimate. For longer horizons, returns must be rescaled accordingly.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")

    if not 0 < lambda_decay < 1:
        raise ValueError("lambda_decay must be between 0 and 1")

    n = len(returns)
    decay_exponents = np.arange(n)
    alpha0 = (1 - lambda_decay) / (lambda_decay * (1 - lambda_decay**n))
    weights = alpha0 * lambda_decay ** decay_exponents
    weights /= weights.sum()

    sorted_indices = np.argsort(returns.values)
    sorted_returns = returns.values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)

    cutoff_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
    var_value = -sorted_returns[cutoff_index] if cutoff_index < n else -sorted_returns[-1]
    var_series = pd.Series(var_value, index=returns.index)

    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series,
        "VaR Violation": returns < -var_series
    })

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data


# ----------------------------------------------------------
# Hybrid Expected Shortfall (Tail Weighted Mean)
# ----------------------------------------------------------
def hybrid_es(result_data, lambda_decay=0.94, wealth=None):
    """
    Estimate 1-day Expected Shortfall (ES) using EWHS method below the Hybrid VaR level.

    Parameters
    ----------
    result_data : pd.DataFrame
        Must contain 'Returns' and constant 'VaR' column (from hybrid_var).
    lambda_decay : float, optional
        Exponential decay factor. Default is 0.94.
    wealth : float, optional
        If provided, ES is scaled by portfolio value.

    Returns
    -------
    result_data : pd.DataFrame
        With added columns 'ES' and optionally 'ES_monetary'.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
    if "VaR" not in result_data.columns:
        raise ValueError("Missing 'VaR' column — run hybrid_var first.")

    if not 0 < lambda_decay < 1:
        raise ValueError("lambda_decay must be between 0 and 1")

    returns = result_data["Returns"]
    var_threshold = result_data["VaR"].iloc[0]

    n = len(returns)
    decay_exponents = np.arange(n)
    alpha0 = (1 - lambda_decay) / (lambda_decay * (1 - lambda_decay**n))
    weights = alpha0 * lambda_decay ** decay_exponents
    weights /= weights.sum()

    tail_mask = returns < -var_threshold
    tail_returns = returns.values[tail_mask]
    tail_weights = weights[tail_mask.values]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = -np.sum(tail_returns * tail_weights) / tail_weights.sum()

    result_data["ES"] = pd.Series(es_value, index=returns.index)

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    return result_data


# ----------------------------------------------------------
# Parametric VaR (Normal or Student-t)
# ----------------------------------------------------------
def parametric_var(returns, confidence_level=0.99, distribution="normal", wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a parametric distribution.

    Fits a Normal or Student-t distribution to the return series and computes
    1-day VaR as the left-tail quantile. VaR is optionally converted to monetary 
    loss if portfolio value is provided.

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
    - This function estimates 1-day VaR. For other horizons like weekly or monthly, scale the reported VaR by √h.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
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
    - This function estimates 1-day VaR. For other horizons like weekly or monthly, scale the reported VaR by √h.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
    returns = result_data["Returns"]

    if distribution == "normal":
        std_dev = returns.std()
        z = norm.ppf(confidence_level)
        es_value = std_dev * norm.pdf(z) / (1 - confidence_level)

    elif distribution == "t":
        df, loc, scale = t.fit(returns)
        t_quantile = t.ppf(confidence_level, df)
        pdf_val = t.pdf(t_quantile, df)
        factor = (df + t_quantile**2) / (df - 1)
        es_value = scale * pdf_val * factor / (1 - confidence_level)

    else:
        raise ValueError("Supported distributions: 'normal', 't'")

    result_data["ES"] = pd.Series(es_value, index=result_data.index)

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    return result_data


# ----------------------------------------------------------
# Cornish-Fisher VaR (Adjusted Normal Quantile)
# ----------------------------------------------------------
def cornish_fisher_var(returns, confidence_level=0.99, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using the Cornish-Fisher expansion.

    Adjusts the normal quantile based on sample skewness and excess kurtosis
    to better capture the shape of the return distribution.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    wealth : float, optional
        Portfolio value in monetary units. If provided, monetary VaR is also returned.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with:
        - 'Returns': original return series
        - 'VaR': Cornish-Fisher adjusted 1-day VaR (decimal loss)
        - 'VaR Violation': boolean flag for violations
        - 'VaR_monetary': optional, monetary VaR if wealth is provided

    Notes
    -----
    - Uses skewness and kurtosis to correct the standard normal quantile.
    - This method does not provide a consistent ES estimate.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
    std_dev = returns.std(ddof=1)
    skewness = skew(returns)
    excess_kurtosis = kurtosis(returns, fisher=True)
    z = norm.ppf(1 - confidence_level)

    # Cornish-Fisher expansion for adjusted quantile
    z_cf = (
        z
        + (1/6) * (z**2 - 1) * skewness
        + (1/24) * (z**3 - 3 * z) * excess_kurtosis
        - (1/36) * (2 * z**3 - 5 * z) * (skewness**2)
    )

    var_value = -z_cf * std_dev
    var_series = pd.Series(var_value, index=returns.index)

    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth

    return result_data