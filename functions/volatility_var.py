"""
Volatility-Based VaR and Expected Shortfall Estimation Module
-------------------------------------------------------------

Provides functions to compute Value-at-Risk (VaR) and Expected Shortfall (ES)
using volatility models such as GARCH, ARCH, EWMA, and Moving Average. Includes
semi-parametric approaches based on empirical quantiles of standardized residuals.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- forecast_garch_variance: Forecast conditional variance using GARCH(1,1)
- forecast_garch_var: Forecast VaR using empirical quantiles and GARCH variance
- var_garch: In-sample and out-of-sample VaR from GARCH-family models
- var_arch: VaR using ARCH(p) with empirical quantiles
- var_ewma: VaR using exponentially weighted moving average volatility
- var_moving_average: VaR using rolling standard deviation
- es_volatility: Expected Shortfall using standardized residuals and volatility estimates

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
from arch import arch_model
from scipy.stats import norm, t, gennorm
import warnings


#----------------------------------------------------------
# GARCH(1,1) Forecast Variance
#----------------------------------------------------------
def forecast_garch_variance(returns, steps_ahead=10, cumulative=False):
    """
    Forecast future conditional variance using a GARCH(1,1) model.

    Uses the analytical closed-form solution under the assumption that returns follow
    a GARCH(1,1) process with normal innovations.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal form.
    steps_ahead : int
        Forecast horizon in days.
    cumulative : bool
        If True, return cumulative variance over the forecast horizon.

    Returns
    -------
    float
        Forecasted variance (squared returns).

    Raises
    ------
    ValueError
        If the fitted model is unstable (alpha + beta ≥ 1).
    Warning
        If NaNs are detected in the return series. 
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")

    omega = fit.params["omega"]
    alpha = fit.params["alpha[1]"]
    beta = fit.params["beta[1]"]
    phi = alpha + beta

    if phi >= 1:
        raise ValueError("Unstable GARCH model: alpha + beta must be < 1.")

    sigma2_t = fit.conditional_volatility.iloc[-1] ** 2
    long_run_var = omega / (1 - phi)

    if not cumulative:
        return long_run_var + (phi**steps_ahead) * (sigma2_t - long_run_var)

    term1 = long_run_var * (steps_ahead - 1 - phi * (1 - phi**(steps_ahead - 1)) / (1 - phi))
    term2 = ((1 - phi**steps_ahead) / (1 - phi)) * sigma2_t
    return term1 + term2


#----------------------------------------------------------
# GARCH(1,1) Forecast Value-at-Risk
#----------------------------------------------------------
def forecast_garch_var(returns, steps_ahead=10, confidence_level=0.99, cumulative=False, wealth=None):
    """
    Forecast empirical Value-at-Risk (VaR) using a GARCH(1,1) model and empirical quantiles.

    Computes variance forecast using a GARCH(1,1) model and scales volatility by
    the empirical quantile of standardized residuals.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal form.
    steps_ahead : int
        Forecast horizon in days.
    confidence_level : float
        Confidence level for VaR (e.g., 0.99).
    cumulative : bool
        If True, use cumulative variance over the horizon.
    wealth : float, optional
        If provided, scales the result into monetary loss units.

    Returns
    -------
    float
        Forecasted empirical VaR (decimal loss or monetary loss).

    Raises
    ------
    Warning
        If NaNs are detected in the return series.
    ValueError
        If the GARCH model is unstable (handled internally by forecast_garch_variance).
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")

    variance = forecast_garch_variance(returns, steps_ahead, cumulative)

    residuals = returns / fit.conditional_volatility
    empirical_z = np.percentile(residuals, 100 * (1 - confidence_level))

    var = -empirical_z * np.sqrt(variance)
    return var * wealth if wealth else var


#----------------------------------------------------------
# Garch VaR
#----------------------------------------------------------
def var_garch(returns, confidence_level=0.99, p=1, q=1, model="GARCH", distribution="normal", wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric GARCH-family model with flexible specification.

    This function fits a GARCH-type volatility model to a return series using maximum likelihood, 
    where the user can choose both the volatility specification (e.g., GARCH, EGARCH, APARCH) and 
    the innovation distribution (e.g., Normal, Student-t, GED, Skewed-t). The fitted model is used 
    to filter conditional volatilities and standardized residuals.

    VaR is then computed semi-parametrically by applying empirical quantiles to the standardized 
    residuals, allowing for flexible tail behavior without fully relying on a parametric distribution.

    The function returns both a time series of in-sample VaR values and a one-step-ahead forecast.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    p : int, optional
        Lag order for the GARCH term. Default is 1.
    q : int, optional
        Lag order for the ARCH term. Default is 1.
    model : {"GARCH", "EGARCH", "GJR", "APARCH"}, optional
        Volatility model specification. Default is "GARCH".
    distribution : {"normal", "t", "ged", "skewt"}, optional
        Distribution for the innovations. Default is "normal".
    wealth : float, optional
        Portfolio value in monetary units. If provided, VaR is also returned in currency.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'Volatility': conditional standard deviation (in decimals)
        - 'Innovations': standardized residuals (ε / σ)
        - 'VaR': semi-parametric Value-at-Risk (decimal loss)
        - 'VaR Violation': boolean flag for VaR breaches
        - 'VaR_monetary': optional, monetary VaR if wealth is provided
    next_day_var : float
        One-step-ahead VaR forecast (decimal loss or monetary loss if wealth is set).

    Raises
    ------
    ValueError
        If an unsupported model or distribution is specified.
    Warning
        If NaNs are detected in the return series.
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    # Validate model and distribution
    model = model.upper()
    distribution = distribution.lower()
    valid_models = ["GARCH", "EGARCH", "GJR", "APARCH"]
    valid_dists = ["normal", "t", "ged", "skewt"]

    if model not in valid_models:
        raise ValueError(f"model must be one of {valid_models}")
    if distribution not in valid_dists:
        raise ValueError(f"distribution must be one of {valid_dists}")

    # Scale returns
    returns_scaled = returns * 100

    # Match-case logic for model specification
    match model:
        case "GARCH":
            model = arch_model(returns_scaled, vol="GARCH", p=p, q=q, dist=distribution)
        case "EGARCH":
            model = arch_model(returns_scaled, vol="EGARCH", p=p, q=q, dist=distribution)
        case "GJR":
            model = arch_model(returns_scaled, vol="GARCH", p=p, o=1, q=q, dist=distribution)
        case "APARCH":
            model = arch_model(returns_scaled, vol="APARCH", p=p, o=1, q=q, dist=distribution)
        case _:
            raise ValueError(f"Unsupported model: {model}")
        
    # Fit model
    fit = model.fit(disp="off")

    # Extract conditional volatility
    volatility = fit.conditional_volatility / 100

    # Compute standardized residuals
    innovations = returns / volatility
    quantile = np.percentile(innovations, 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    # Forecast next-day variance and compute 1-day ahead VaR
    forecast_var = fit.forecast(horizon=1).variance.values[-1][0]
    next_day_vol = np.sqrt(forecast_var) / 100
    next_day_var = abs(quantile * next_day_vol)

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        next_day_var *= wealth

    return result_data, next_day_var


#----------------------------------------------------------
# Arch VaR
#----------------------------------------------------------
def var_arch(returns, confidence_level=0.99, p=1, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric ARCH model.

    Fits an ARCH(p) model to the return series using maximum likelihood estimation,
    then computes Value-at-Risk (VaR) based on the empirical quantile of standardized residuals.
    This semi-parametric approach preserves flexibility in modeling the tails of the distribution
    while capturing volatility clustering through the ARCH filter.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    p : int, optional
        Lag order for the ARCH model. Default is 1.
    wealth : float, optional
        Portfolio value in monetary units. If provided, VaR is also returned in currency.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'Volatility': conditional standard deviation (in decimals)
        - 'Innovations': standardized residuals (ε / σ)
        - 'VaR': semi-parametric Value-at-Risk (decimal loss)
        - 'VaR Violation': boolean flag for VaR breaches
        - 'VaR_monetary': optional, monetary VaR if wealth is provided
    next_day_var : float
        One-step-ahead VaR forecast (decimal loss or monetary loss if wealth is set).

    Raises
    ------
    Warning
        If NaNs are detected in the return series.
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    returns_scaled = returns * 100

    model = arch_model(returns_scaled, vol="ARCH", p=p)
    fit = model.fit(disp="off")

    volatility = fit.conditional_volatility / 100
    innovations = returns / volatility
    quantile = np.percentile(innovations, 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    next_day_vol = (fit.forecast(horizon=1).variance.values[-1][0] ** 0.5) / 100
    next_day_var = abs(quantile * next_day_vol)

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        next_day_var *= wealth

    return result_data, next_day_var


#----------------------------------------------------------
# EWMA VaR
#----------------------------------------------------------
def var_ewma(returns, confidence_level=0.99, decay_factor=0.94, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric EWMA volatility model.

    Fits an Exponentially Weighted Moving Average (EWMA) model to estimate conditional volatility,
    then computes Value-at-Risk (VaR) using empirical quantiles of standardized residuals. The 
    final EWMA volatility value is used to forecast the next-day VaR.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    decay_factor : float, optional
        EWMA smoothing parameter (e.g., 0.94). Default is 0.94.
    wealth : float, optional
        Portfolio value in monetary units. If provided, VaR is also returned in currency.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'Volatility': EWMA volatility estimate (in decimals)
        - 'Innovations': standardized residuals (ε / σ)
        - 'VaR': semi-parametric Value-at-Risk (decimal loss)
        - 'VaR Violation': boolean flag for VaR breaches
        - 'VaR_monetary': optional, monetary VaR if wealth is provided
    next_day_var : float
        One-step-ahead VaR forecast (decimal loss or monetary loss if wealth is set).

    Raises
    ------
    Warning
        If NaNs are detected in the return series.
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    squared = returns ** 2
    ewma_var = squared.ewm(alpha=1 - decay_factor).mean()
    volatility = np.sqrt(ewma_var)

    innovations = returns / volatility
    quantile = np.percentile(innovations, 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]

    # Manual 1-step-ahead forecast for EWMA volatility
    last_return = returns.iloc[-1]
    last_vol = volatility.iloc[-1]
    last_var = last_vol ** 2
    next_var = decay_factor * last_var + (1 - decay_factor) * last_return**2
    next_day_vol = np.sqrt(next_var)
    next_day_var = abs(quantile * next_day_vol)

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        next_day_var *= wealth

    return result_data, next_day_var


#----------------------------------------------------------
# MA VaR
#----------------------------------------------------------
def var_moving_average(returns, confidence_level=0.99, window=20, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric moving average volatility model.

    Estimates volatility using a fixed-window moving average of squared returns, and computes
    Value-at-Risk (VaR) using the empirical quantile of standardized residuals. The most
    recent rolling volatility is used to forecast the next-day VaR.

    Parameters
    ----------
    returns : pd.Series
        Daily return series in decimal format (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    window : int, optional
        Size of the moving average window. Default is 20.
    wealth : float, optional
        Portfolio value in monetary units. If provided, VaR is also returned in currency.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with the following columns:
        - 'Returns': original return series
        - 'Volatility': rolling volatility estimate (in decimals)
        - 'Innovations': standardized residuals (ε / σ)
        - 'VaR': semi-parametric Value-at-Risk (decimal loss)
        - 'VaR Violation': boolean flag for VaR breaches
        - 'VaR_monetary': optional, monetary VaR if wealth is provided
    next_day_var : float
        One-step-ahead VaR forecast (decimal loss or monetary loss if wealth is set).

    Raises
    ------
    Warning
        If NaNs are detected in the return series.
    """
    if returns.isna().any():
        warnings.warn("NaNs detected in return series. Consider handling or dropping missing values.")

    volatility = returns.rolling(window=window).std()
    innovations = returns / volatility
    quantile = np.percentile(innovations, 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]

    # Manual 1-step-ahead forecast for MA volatility
    recent_returns = returns.iloc[-window:]
    next_day_vol = recent_returns.std()
    next_day_var = abs(quantile * next_day_vol)

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        next_day_var *= wealth

    return result_data, next_day_var


#----------------------------------------------------------
# Expected Shortfall Volatility
#----------------------------------------------------------
def es_volatility(data, confidence_level, subset=None, wealth=None):
    """
    Estimate Expected Shortfall (ES) using standardized residuals and model-implied volatility.

    This function computes dynamic Expected Shortfall (ES) by averaging standardized residuals 
    (innovations) that fall below a quantile threshold, then scales the tail mean by the conditional 
    volatility. It is designed for models that return both time-varying volatility and residuals.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing at least the columns:
        - 'Innovations': standardized residuals (ε / σ)
        - 'Volatility': conditional standard deviation (in decimals)
    confidence_level : float
        Confidence level for ES (e.g., 0.99).
    subset : tuple of str or pd.Timestamp, optional
        Optional date range (start, end) to compute tail mean on a subset of the data.
    wealth : float, optional
        Portfolio value in monetary units. If provided, ES is also returned in currency.

    Returns
    -------
    data : pd.DataFrame
        Original input DataFrame extended with:
        - 'ES': time-varying expected shortfall (decimal loss)
        - 'ES_monetary': optional, monetary ES if wealth is provided

    Raises
    ------
    ValueError
        If the required columns 'Innovations' or 'Volatility' are missing from the input.
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")

    subset_data = data if subset is None else data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"], 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    data["ES"] = -data["Volatility"] * tail_mean

    if wealth is not None:
        data["ES_monetary"] = data["ES"] * wealth

    return data
