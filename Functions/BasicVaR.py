#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd

#################################################
# Note: double check all formulas 
#################################################

#----------------------------------------------------------
# Historical VaR (Non-Parametric)
#----------------------------------------------------------
def var_historical(returns, confidence_level, holding_period=1):
    """
    Historical VaR Estimation (Non-Parametric).

    Estimate Value-at-Risk (VaR) directly from the empirical distribution of returns without assuming any parametric model.

    Description:
    - VaR is computed as the empirical quantile at the specified confidence level.
    - Scaling for multi-day holding periods is done by multiplying by sqrt(holding_period).

    Formulas:
    - Historical VaR:
        VaR = - Quantile(returns, 1 - confidence_level) × sqrt(holding_period)

    Parameters:
    - returns (pd.Series):
        Time series of returns (decimal format, e.g., 0.01 = 1%).

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional):
        Holding period in days (default = 1).

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'VaR': estimated constant VaR series
        - 'VaR Violation': boolean flag where return < -VaR

    - var_estimate (float):
        Estimated Historical VaR (positive % loss magnitude).

    Notes:
    - Assumes i.i.d. returns when scaling by sqrt(holding_period).
    - The VaR estimate is reported as an absolute positive percentage.
    """
    var_cutoff = np.percentile(returns.dropna(), 100 * (1 - confidence_level))
    scaled_var = np.sqrt(holding_period) * var_cutoff

    var_series = pd.Series(-scaled_var, index=returns.index)
    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    var_estimate = 100 * abs(scaled_var)
    return result_data.dropna(), var_estimate


#----------------------------------------------------------
# Parametric VaR (i.i.d. assumption)
#----------------------------------------------------------
def var_parametric_iid(returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Parametric i.i.d. VaR Estimation.

    Estimate Value-at-Risk (VaR) under the assumption that returns are i.i.d. 
    and follow a specified parametric distribution (Normal, Student-t, or GED).

    Description:
    - Fits the chosen distribution to historical returns.
    - Computes VaR using the corresponding quantile.
    - Scales volatility for multi-day holding periods by sqrt(holding_period).

    Formulas:
    - Parametric VaR:
        VaR = - Quantile(distribution, 1 - confidence_level) × σ × sqrt(holding_period)
    where:
        - σ: standard deviation or scale parameter depending on distribution

    Parameters:
    - returns (pd.Series):
        Time series of returns (decimal format, e.g., 0.01 = 1%).

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional):
        Holding period in days (default = 1).

    - distribution (str, optional):
        Distribution assumed for returns ("normal", "t", or "ged").

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'VaR': estimated constant VaR series
        - 'VaR Violation': boolean flag where return < -VaR

    - var_estimate (float):
        Estimated Parametric VaR (positive % loss magnitude).

    Notes:
    - For Student-t and GED, scale and shape parameters are estimated by MLE.
    - Assumes returns are independent and identically distributed (i.i.d.).
    """
    returns_clean = returns.dropna()

    if distribution == "normal":
        std_dev = returns_clean.std()
        quantile = norm.ppf(1 - confidence_level)
        scaled_std = std_dev * np.sqrt(holding_period)

    elif distribution == "t":
        df, loc, scale = t.fit(returns_clean)
        quantile = t.ppf(1 - confidence_level, df)
        scaled_std = scale * np.sqrt(holding_period)

    elif distribution == "ged":
        beta, loc, scale = gennorm.fit(returns_clean)
        quantile = gennorm.ppf(1 - confidence_level, beta)
        scaled_std = scale * np.sqrt(holding_period)

    else:
        raise ValueError("Supported distributions: 'normal', 't', 'ged'")

    var_value = -quantile * scaled_std

    var_series = pd.Series(var_value, index=returns.index)
    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    var_estimate = 100 * abs(var_value)
    return result_data.dropna(), var_estimate