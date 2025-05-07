#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t, gennorm
import warnings

#################################################
# Note: double check all formulas 
#################################################

#----------------------------------------------------------
# Garch Forecast (Analytical Formula, for Variance or VaR)
#----------------------------------------------------------
def garch_forecast(
    returns,
    steps_ahead=10,
    cumulative=False,
    compute_var=False,
    confidence_level=0.99,
    wealth=None
):
    """
    Forecast future variance or Value-at-Risk (VaR) using a GARCH(1,1) model with normal innovations.

    This function fits a GARCH(1,1) model under the assumption of normally distributed residuals
    and computes variance forecasts using the closed-form analytical formula. It supports both
    step-ahead and cumulative variance forecasting over a user-defined horizon. Optionally, it
    converts the forecasted variance into a fully parametric VaR estimate using the Normal distribution.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - steps_ahead (int): Forecast horizon in days.
    - cumulative (bool): If True, returns cumulative variance over the forecast horizon.
    - compute_var (bool): If True, returns parametric VaR instead of variance.
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - wealth (float, optional): Portfolio value. If set, VaR is returned in monetary units.

    Returns:
    - float:
        - If compute_var=False: forecasted variance (in decimal squared returns).
        - If compute_var=True and wealth is None: parametric VaR (in decimal loss magnitude).
        - If compute_var=True and wealth is set: parametric VaR (in monetary units).

    Raises:
    - ValueError: If the model is unstable (α + β ≥ 1), or if `wealth` is set without `compute_var=True`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(returns, vol="GARCH", p=1, q=1, dist="normal")
        fit = model.fit(disp="off")

    omega = fit.params["omega"]
    alpha = fit.params["alpha[1]"]
    beta = fit.params["beta[1]"]
    phi = alpha + beta

    if phi >= 1:
        raise ValueError("Unstable GARCH: alpha + beta must be < 1 for variance forecast.")

    sigma2_t = fit.conditional_volatility.iloc[-1] ** 2
    var_long_run = omega / (1 - phi)

    if cumulative:
        denom = 1 - phi
        term1 = var_long_run * (steps_ahead - 1 - phi * (1 - phi**(steps_ahead - 1)) / denom)
        term2 = ((1 - phi**steps_ahead) / denom) * sigma2_t
        variance = term1 + term2
    else:
        variance = var_long_run + phi**steps_ahead * (sigma2_t - var_long_run)

    if wealth is not None and not compute_var:
        raise ValueError("Wealth can only be used when compute_var=True")

    if not compute_var:
        return variance  # plain decimal variance

    z = norm.ppf(1 - confidence_level)
    var_result = -z * np.sqrt(variance)

    if wealth is not None:
        return var_result * wealth

    return var_result


#----------------------------------------------------------
# Garch VaR
#----------------------------------------------------------
def var_garch(returns, confidence_level=0.99, p=1, q=1, vol_model="GARCH", distribution="normal", wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric GARCH-family model with flexible specification.

    This function fits a GARCH-type volatility model to a return series using maximum likelihood, 
    where the user can choose both the volatility specification (e.g., GARCH, EGARCH, APARCH) and 
    the innovation distribution (e.g., Normal, Student-t, GED, Skewed-t). The fitted model is used 
    to filter conditional volatilities and standardized residuals.

    VaR is then computed semi-parametrically by applying empirical quantiles to the standardized 
    residuals, allowing for flexible tail behavior without fully relying on a parametric distribution.

    The function returns both a time series of in-sample VaR values and a one-step-ahead forecast.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99 for 99%).
    - p (int): Lag order for the GARCH term (default: 1).
    - q (int): Lag order for the ARCH term (default: 1).
    - vol_model (str): One of {"GARCH", "EGARCH", "GJR", "APARCH"}.
    - distribution (str): One of {"normal", "t", "ged", "skewt"}.
    - wealth (float, optional): Portfolio value. If set, VaR outputs are in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'Volatility' (conditional standard deviation in decimals)
        - 'Innovations' (standardized residuals)
        - 'VaR' (decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (if wealth is provided, in currency units)
    - next_day_var (float): One-step-ahead forecasted VaR (decimal or currency if scaled by wealth)

    Raises:
    - ValueError: If an unsupported model or distribution is specified.
    """
    # Validate model and distribution
    vol_model = vol_model.upper()
    distribution = distribution.lower()
    valid_models = ["GARCH", "EGARCH", "GJR", "APARCH"]
    valid_dists = ["normal", "t", "ged", "skewt"]

    if vol_model not in valid_models:
        raise ValueError(f"vol_model must be one of {valid_models}")
    if distribution not in valid_dists:
        raise ValueError(f"distribution must be one of {valid_dists}")

    # Scale returns
    returns_scaled = returns * 100

    # Model-specific settings
    if vol_model == "GARCH":
        model = arch_model(returns_scaled, vol="Garch", p=p, q=q, dist=distribution)
    elif vol_model == "EGARCH":
        model = arch_model(returns_scaled, vol="EGARCH", p=p, q=q, dist=distribution)
    elif vol_model == "GJR":
        model = arch_model(returns_scaled, vol="GARCH", p=p, o=1, q=q, dist=distribution)
    elif vol_model == "APARCH":
        model = arch_model(returns_scaled, vol="APARCH", p=p, o=1, q=q, dist=distribution)

    # Fit model
    fit = model.fit(disp="off")

    # Extract conditional volatility
    volatility = fit.conditional_volatility / 100

    # Compute standardized residuals
    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
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

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# Arch VaR
#----------------------------------------------------------
def var_arch(returns, confidence_level=0.99, p=1, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric ARCH model.

    This function fits an ARCH(p) volatility model to a return series using maximum likelihood,
    then computes Value-at-Risk (VaR) using empirical quantiles of the standardized residuals.
    The approach allows for flexible tail estimation while using a parametric volatility filter.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99 for 99%).
    - p (int): Order of the ARCH model (default: 1).
    - wealth (float, optional): Portfolio value. If set, VaR outputs are returned in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'Volatility' (conditional standard deviation in decimals)
        - 'Innovations' (standardized residuals)
        - 'VaR' (semi-parametric VaR in decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (if wealth is provided, in currency units)
    - next_day_var (float): One-step-ahead forecasted VaR (decimal or monetary if wealth is set)
    """
    returns_scaled = returns * 100

    model = arch_model(returns_scaled, vol="ARCH", p=p)
    fit = model.fit(disp="off")

    volatility = fit.conditional_volatility / 100
    innovations = returns / volatility
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
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

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# EWMA VaR
#----------------------------------------------------------
def var_ewma(returns, confidence_level=0.99, decay_factor=0.94, wealth=None):
    """
    Estimate Value-at-Risk (VaR) using a semi-parametric EWMA volatility model.

    This function estimates conditional volatility using an Exponentially Weighted Moving Average (EWMA)
    model, then computes Value-at-Risk (VaR) using empirical quantiles of the standardized residuals.
    The final volatility value is used for the one-step-ahead VaR forecast.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - decay_factor (float): EWMA smoothing parameter (e.g., 0.94).
    - wealth (float, optional): Portfolio value. If set, VaR outputs are returned in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'Volatility' (EWMA standard deviation in decimals)
        - 'Innovations' (standardized residuals)
        - 'VaR' (semi-parametric VaR in decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (if wealth is provided, in currency units)
    - next_day_var (float): One-step-ahead forecasted VaR (decimal or monetary if wealth is set)
    """
    squared = returns ** 2
    ewma_var = squared.ewm(alpha=1 - decay_factor).mean()
    volatility = np.sqrt(ewma_var)

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

    next_day_vol = volatility.iloc[-1]
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

    This function estimates conditional volatility using a rolling standard deviation over a fixed window,
    then computes Value-at-Risk (VaR) using empirical quantiles of the standardized residuals.
    The final rolling volatility is used for the one-step-ahead VaR forecast.

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - window (int): Size of the moving average window (default: 20).
    - wealth (float, optional): Portfolio value. If set, VaR outputs are returned in monetary units.

    Returns:
    - result_data (pd.DataFrame): With columns:
        - 'Returns' (decimal)
        - 'Volatility' (rolling standard deviation in decimals)
        - 'Innovations' (standardized residuals)
        - 'VaR' (semi-parametric VaR in decimal loss magnitude)
        - 'VaR Violation' (bool)
        - 'VaR_monetary' (if wealth is provided, in currency units)
    - next_day_var (float): One-step-ahead forecasted VaR (decimal or monetary if wealth is set)
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

    next_day_vol = volatility.iloc[-1]
    next_day_var = abs(quantile * next_day_vol)

    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        next_day_var *= wealth

    return result_data, next_day_var


