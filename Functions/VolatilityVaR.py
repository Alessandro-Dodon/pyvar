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
#       double check caller logic and wealth scaling
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
    distribution="normal"
):
    """
    GARCH(1,1) Variance or Value-at-Risk Forecast.

    Forecasts future variance or Value-at-Risk (VaR) using an analytical formula from a fitted GARCH(1,1) model.

    Description:
    - Variance is forecasted recursively via:
        σ²_{t+h} = ω / (1 - φ) + φ^h × (σ²_t - ω / (1 - φ)), where φ = α + β.
    - Optionally, cumulative variance over the forecast horizon is returned.
    - If compute_var=True, transforms forecasted variance into a Value-at-Risk (VaR) estimate.

    Formulas:
    - Single-step variance:  
        σ²_{t+h} = ω / (1 - φ) + φ^h × (σ²_t - ω / (1 - φ))
    - Cumulative variance over h steps:  
        Var_cum = sum of σ²_{t+i} from i = 1 to h
    - VaR:
        VaR = -Quantile × √(Variance)

    Parameters:
    - returns (pd.Series): Daily return series in decimal format (e.g., 0.01 for 1%).
    - steps_ahead (int): Forecast horizon in days.
    - cumulative (bool): If True, returns cumulative variance over horizon.
    - compute_var (bool): If True, returns VaR instead of variance.
    - confidence_level (float): Confidence level for VaR (e.g., 0.99 for 99% VaR).
    - distribution (str): Distribution for standardized residuals: {"normal", "t", "ged"}.

    Returns:
    - float:
        - If compute_var=False: forecasted variance in decimal units (e.g., 0.0001).
        - If compute_var=True: forecasted VaR in decimal loss magnitude (e.g., 0.015 for 1.5%).

    Notes:
    - Assumes input returns are in decimals.
    - Output VaR is a positive scalar representing potential loss.
    - Stability condition φ = α + β < 1 is enforced.
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

    if not compute_var:
        return variance  # plain decimal variance

    # VaR calculation
    innovations = fit.resid / fit.conditional_volatility
    innovations = innovations.dropna()

    distribution = distribution.lower()
    if distribution == "normal":
        z = norm.ppf(1 - confidence_level)
    elif distribution == "t":
        df, loc, scale = t.fit(innovations)
        z = t.ppf(1 - confidence_level, df, loc=loc, scale=scale)
    elif distribution == "ged":
        beta_ged, loc, scale = gennorm.fit(innovations)
        z = gennorm.ppf(1 - confidence_level, beta_ged, loc=loc, scale=scale)
    else:
        raise ValueError("Supported distributions: 'normal', 't', 'ged'")

    return -z * np.sqrt(variance)  # VaR in decimals


#----------------------------------------------------------
# Garch VaR
#----------------------------------------------------------
def var_garch(returns, confidence_level=0.99, p=1, q=1, vol_model="GARCH", distribution="normal"):
    """
    GARCH-type Value-at-Risk (VaR) Estimation.

    Fits a GARCH-type volatility model to daily return data and estimates one-step-ahead and time series Value-at-Risk (VaR)
    using standardized residual quantiles.

    Model:
    - Standard GARCH(1,1) variance recursion:
        σ²ₜ = ω + α * ε²ₜ₋₁ + β * σ²ₜ₋₁
    - Supported extensions:
        - EGARCH: models log-volatility to capture asymmetry.
        - GJR-GARCH: includes a leverage effect via threshold terms.
        - APARCH: allows asymmetric power transformation of returns.

    Parameters:
    - returns (pd.Series): Daily returns in decimal format (e.g., 0.01 for 1 percent).
    - confidence_level (float): VaR confidence level (e.g., 0.99 for 99 percent).
    - p (int): GARCH lag order (default is 1).
    - q (int): ARCH lag order (default is 1).
    - vol_model (str): Volatility model, one of "GARCH", "EGARCH", "GJR", or "APARCH".
    - distribution (str): Distribution of standardized residuals, one of "normal", "t", "ged", or "skewt".

    Returns:
    - result_data (pd.DataFrame): DataFrame with columns:
        - 'Returns': original returns in decimals
        - 'Volatility': model-implied conditional standard deviation (in decimals)
        - 'Innovations': standardized residuals
        - 'VaR': estimated VaR at each time point (in decimals, positive loss magnitude)
        - 'VaR Violation': True where Return < -VaR
    - next_day_var (float): One-step-ahead VaR forecast (absolute value, in decimals)

    Notes:
    - Input returns are scaled to percentages internally for model fitting and then scaled back.
    - Output VaR values are in decimal units (e.g., 0.012 means a 1.2 percent potential loss).
    - The function drops rows with missing values before returning the result.
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

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# Arch VaR
#----------------------------------------------------------
def var_arch(returns, confidence_level=0.99, p=1):
    """
    ARCH-type Value-at-Risk (VaR) Estimation.

    Fits an ARCH(p) volatility model to daily return data and estimates Value-at-Risk (VaR)
    based on the empirical quantiles of standardized residuals.

    Model:
    - ARCH(p) variance recursion:
        σ²ₜ = ω + α₁ * ε²ₜ₋₁ + α₂ * ε²ₜ₋₂ + ... + αₚ * ε²ₜ₋ₚ

    Parameters:
    - returns (pd.Series): Daily returns in decimal format (e.g., 0.01 for 1 percent).
    - confidence_level (float): VaR confidence level (e.g., 0.99 for 99 percent).
    - p (int): ARCH order (default is 1).

    Returns:
    - result_data (pd.DataFrame): DataFrame with columns:
        - 'Returns': original return series (in decimals)
        - 'Volatility': model-implied conditional standard deviation (in decimals)
        - 'Innovations': standardized residuals
        - 'VaR': Value-at-Risk at each time point (positive loss magnitude, in decimals)
        - 'VaR Violation': Boolean flag indicating if return < -VaR
    - next_day_var (float): One-step-ahead VaR forecast (absolute value, in decimals)

    Notes:
    - Input returns are internally scaled to percentages during model fitting and scaled back in the output.
    - All volatility and VaR values are returned in decimal units (e.g., 0.012 means 1.2 percent potential loss).
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
    next_day_var =  abs(quantile * next_day_vol)

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# EWMA VaR
#----------------------------------------------------------
def var_ewma(returns, confidence_level=0.99, decay_factor=0.94):
    """
    EWMA-based Value-at-Risk (VaR) Estimation.

    Estimates daily Value-at-Risk (VaR) using an Exponentially Weighted Moving Average (EWMA) model 
    for volatility and empirical quantiles of standardized residuals.

    Model:
    - EWMA volatility recursion:
        σ²ₜ = λ * σ²ₜ₋₁ + (1 - λ) * ε²ₜ₋₁²
    where:
        - λ = decay_factor (e.g., 0.94)
        - εₜ = return at time t

    Parameters:
    - returns (pd.Series): Daily returns in decimal format (e.g., 0.01 for 1 percent).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - decay_factor (float): Smoothing parameter for EWMA (default is 0.94).

    Returns:
    - result_data (pd.DataFrame): DataFrame with:
        - 'Returns': input return series (in decimals)
        - 'Volatility': EWMA volatility estimate (in decimals)
        - 'Innovations': returns standardized by volatility
        - 'VaR': daily VaR estimates (positive loss magnitude, in decimals)
        - 'VaR Violation': Boolean flag where return < -VaR
    - next_day_var (float): One-step-ahead VaR forecast (decimal format)

    Notes:
    - All outputs are in decimal units. For example, a VaR of 0.012 means a 1.2 percent loss.
    - The last volatility value is used to compute the 1-day ahead VaR.
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
    next_day_var =  abs(quantile * next_day_vol)

    return result_data, next_day_var


#----------------------------------------------------------
# MA VaR
#----------------------------------------------------------
def var_moving_average(returns, confidence_level=0.99, window=20):
    """
    Moving Average-based Value-at-Risk (VaR) Estimation.

    Estimate daily Value-at-Risk (VaR) using a simple rolling window standard deviation model for volatility 
    and empirical quantiles of standardized residuals.

    Model:
    - Volatility is computed as the rolling standard deviation over a fixed-size window:
        σₜ = std(returnsₜ₋₍window₋₁₎ to returnsₜ)

    Parameters:
    - returns (pd.Series): Daily returns in decimal format (e.g., 0.01 for 1 percent).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - window (int): Window size for rolling volatility estimation (default = 20).

    Returns:
    - result_data (pd.DataFrame): DataFrame containing:
        - 'Returns': input return series (in decimals)
        - 'Volatility': rolling standard deviation (in decimals)
        - 'Innovations': standardized residuals (return divided by volatility)
        - 'VaR': estimated daily VaR (positive loss magnitude, in decimals)
        - 'VaR Violation': Boolean flag where return < -VaR
    - next_day_var (float): One-step-ahead VaR estimate based on the last available volatility (in decimals)

    Notes:
    - All outputs are expressed in decimal units. For example, a VaR of 0.012 indicates a 1.2 percent loss.
    - Volatility is backward-looking and constant across each rolling window.
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

    return result_data, next_day_var


#----------------------------------------------------------
# Wealth Scaling for VaR
#----------------------------------------------------------
def apply_wealth_scaling(result_data, wealth):
    """
    Apply wealth scaling to Value-at-Risk (VaR) estimates.

    Parameters:
    - result_data (pd.DataFrame): Output from a VaR model function. Must contain 'VaR' column (in decimals).
    - wealth (float or None): Portfolio value for converting VaR from percentage (decimal) to monetary units.

    Returns:
    - pd.DataFrame: DataFrame with a new 'VaR_monetary' column (if wealth is provided).

    Notes:
    - Does not modify the original 'VaR' column (remains in decimals, e.g., 0.01 = 1%).
    - Adds a new column 'VaR_monetary' only if wealth is specified.
    """
    if wealth is not None:
        if "VaR" in result_data.columns:
            result_data["VaR_monetary"] = result_data["VaR"] * wealth
    return result_data


#----------------------------------------------------------
# Unified Volatility-Based VaR Caller with Wealth Scaling
#----------------------------------------------------------
def compute_var_volatility(returns, confidence_level=0.99, method="garch", wealth=None, **kwargs):
    """
    Unified Value-at-Risk (VaR) Estimator Using Volatility Models.

    Dispatches the VaR estimation to a selected volatility-based model and optionally applies wealth scaling.

    Parameters:
    - returns (pd.Series): Return series in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR estimation (e.g., 0.99).
    - method (str): One of {"garch", "arch", "ewma", "ma", "forecast"}.
    - wealth (float or None): If provided, scales VaR results to monetary units.
    - **kwargs: Additional arguments for the chosen model.

    Returns:
    - result_data (pd.DataFrame or None): Contains 'VaR' in decimals, and 'VaR_monetary' if wealth is given.
    - next_day_var (float): One-step-ahead VaR estimate in decimal or monetary units.

    Notes:
    - For method='forecast', only a scalar is returned (not a full time series).
    - All VaR values are returned as positive loss magnitudes.
    - Wealth scaling does not alter the original 'VaR' values.
    """
    method = method.lower()

    model_functions = {
        "garch": var_garch,
        "arch": var_arch,
        "ewma": var_ewma,
        "ma": var_moving_average,
        "forecast": garch_forecast
    }

    if method not in model_functions:
        raise ValueError(f"'method' must be one of {list(model_functions.keys())}")

    # Special case for forecast: returns only scalar, not DataFrame
    if method == "forecast":
        var_forecast = garch_forecast(
            returns=returns,
            compute_var=True,
            confidence_level=confidence_level,
            **kwargs
        )
        if wealth is not None:
            var_forecast = var_forecast * wealth
        return None, var_forecast

    # Standard case: use model function
    model_func = model_functions[method]
    result_data, next_day_var = model_func(returns, confidence_level, **kwargs)

    result_data = apply_wealth_scaling(result_data, wealth)

    if wealth is not None:
        next_day_var = next_day_var * wealth

    return result_data, next_day_var
