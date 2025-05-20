"""
Volatility-Based VaR and Expected Shortfall Estimation Module
-------------------------------------------------------------

Provides functions to compute Value-at-Risk (VaR) and Expected Shortfall (ES)
using volatility models such as GARCH, ARCH, EWMA, and Moving Average. 
A semi-parametric approach is adopted consistently, based on empirical quantiles of standardized residuals.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- forecast_garch_variance: Forecast conditional variance using GARCH(1,1)
- forecast_garch_var: Estimates VaR using forecasted conditional variance
- garch_var: VaR from GARCH-family models with flexible distributions
- arch_var: VaR using ARCH(p) 
- ewma_var: VaR using exponentially weighted moving average volatility
- ma_var: VaR using rolling standard deviation
- volatility_es: Expected Shortfall using standardized residuals and volatility estimates
"""


#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t, gennorm


#----------------------------------------------------------
# GARCH(1,1) Forecast Variance
#----------------------------------------------------------
def forecast_garch_variance(returns, steps_ahead=10, cumulative=False):
    """
    Main
    ----
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

    Notes
    -----
    - Returns are scaled by 100 before fitting to improve numerical stability.
    """
    returns_scaled = returns * 100

    model = arch_model(returns_scaled, vol="GARCH", p=1, q=1, dist="normal")
    fit = model.fit(disp="off")

    omega_scaled = fit.params["omega"]
    alpha = fit.params["alpha[1]"]
    beta = fit.params["beta[1]"]
    phi = alpha + beta

    if phi >= 1:
        raise ValueError("Unstable GARCH model: alpha + beta must be < 1.")

    # Rescale omega to original units
    omega = omega_scaled / 10000

    sigma2_t_scaled = fit.conditional_volatility.iloc[-1] ** 2
    sigma2_t = sigma2_t_scaled / 10000

    long_run_var = omega / (1 - phi)

    if not cumulative:
        forecast = long_run_var + (phi**steps_ahead) * (sigma2_t - long_run_var)
        return forecast

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
    ValueError
        If the GARCH model is unstable (handled internally by forecast_garch_variance).

    Notes
    -----
    - Returns are scaled by 100 before fitting to improve numerical stability.
    """
    returns_scaled = returns * 100

    model = arch_model(returns_scaled, vol="GARCH", p=1, q=1, dist="normal")
    fit = model.fit(disp="off")

    variance = forecast_garch_variance(returns, steps_ahead, cumulative)

    residuals = returns_scaled / fit.conditional_volatility
    empirical_z = np.percentile(residuals, 100 * (1 - confidence_level))

    var = -empirical_z * np.sqrt(variance)
    return var * wealth if wealth else var


#----------------------------------------------------------
# Garch VaR
#----------------------------------------------------------
def garch_var(returns, confidence_level=0.99, p=1, q=1, model="GARCH", distribution="normal", wealth=None):
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using a flexible GARCH-family model.

    This function fits a GARCH-type volatility model to a return series using MLE.
    Users can choose the volatility model (e.g., GARCH, EGARCH, APARCH) and innovation
    distribution (e.g., Normal, Student-t, GED, Skewed-t). Conditional volatility is 
    extracted to compute standardized residuals and empirical quantile-based VaR.

    VaR is computed both in-sample and as a 1-day ahead forecast.

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

    Notes
    -----
    - Returns are scaled by 100 before fitting to improve numerical stability.
    """
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
def arch_var(returns, confidence_level=0.99, p=1, wealth=None):
    """
    Main
    ----
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

    Notes
    -----
    - Returns are scaled by 100 before fitting to improve numerical stability.
    """
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
def ewma_var(returns, confidence_level=0.99, decay_factor=0.94, wealth=None):
    """
    Main
    ----
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
    """
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
def ma_var(returns, confidence_level=0.99, window=20, wealth=None):
    """
    Main
    ----
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

    Notes
    -----
    - A very short window may lead to unstable VaR estimates.
    """
    volatility = returns.rolling(window=window).std()
    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]
    result_data.dropna(inplace=True)

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
def volatility_es(result_data, confidence_level, wealth=None):
    """
    Main
    ----
    Estimate Expected Shortfall (ES) using standardized residuals and model-implied volatility.

    This function computes dynamic Expected Shortfall (ES) by averaging standardized residuals 
    (innovations) that fall below a quantile threshold, then scales the tail mean by the conditional 
    volatility. It is designed for models that return both time-varying volatility and residuals.

    Parameters
    ----------
    result_data : pd.DataFrame
        DataFrame containing at least the columns:
        - 'Innovations': standardized residuals (ε / σ)
        - 'Volatility': conditional standard deviation (in decimals)
    confidence_level : float
        Confidence level for ES (e.g., 0.99).
    wealth : float, optional
        Portfolio value in monetary units. If provided, ES is also returned in currency.

    Returns
    -------
    result_data : pd.DataFrame
        Original input DataFrame extended with:
        - 'ES': time-varying expected shortfall (decimal loss)
        - 'ES_monetary': optional, monetary ES if wealth is provided

    Raises
    ------
    ValueError
        If the required columns 'Innovations' or 'Volatility' are missing from the input.
    """
    if "Innovations" not in result_data.columns or "Volatility" not in result_data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")

    threshold = np.percentile(result_data["Innovations"], 100 * (1 - confidence_level))
    tail_mean = result_data["Innovations"][result_data["Innovations"] < threshold].mean()

    result_data["ES"] = -result_data["Volatility"] * tail_mean

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    return result_data

