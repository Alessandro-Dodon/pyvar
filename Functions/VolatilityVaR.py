#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t, gennorm
import warnings


#################################################
# Note: double check all formulas and scaling
#       add caller (meta) function?
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

    Forecast future variance or Value-at-Risk (VaR) using an analytical formula from a fitted GARCH(1,1) model.

    Description:
    - The variance is forecasted recursively following the GARCH(1,1) structure:
        σ²_{t+h} = ω / (1 - φ) + φ^h (σ²_t - ω / (1 - φ)), where φ = α + β.
    - Optionally, cumulative variance over multiple steps is computed.
    - If requested, the forecasted variance is transformed into a Value-at-Risk estimate.

    Formulas:
    - Non-cumulative variance:  
        σ²_{t+h} = ω / (1 - φ) + φ^h (σ²_t - ω / (1 - φ))
    - Cumulative variance:  
        Sum of σ²_{t+i} from i=1 to h
    - VaR:  
        VaR = -Quantile × √(Variance)

    Parameters:
    - returns (pd.Series): Daily return series (decimal format, e.g., 0.01 for 1%).
    - steps_ahead (int): Forecast horizon in days.
    - cumulative (bool): If True, returns cumulative variance across the horizon.
    - compute_var (bool): If True, returns Value-at-Risk instead of variance.
    - confidence_level (float): Confidence level for VaR computation (e.g., 0.99).
    - distribution (str): Distribution of standardized innovations: "normal", "t", or "ged".

    Returns:
    - float: Forecasted variance (decimal) or forecasted VaR (percentage).

    Notes:
    - Stability condition checked: α + β must be < 1.
    - VaR output is a positive loss magnitude.
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

    return 100 * -z * np.sqrt(variance)  # VaR in %


#----------------------------------------------------------
# Garch VaR
#----------------------------------------------------------
def var_garch(returns, confidence_level, p=1, q=1, vol_model="GARCH", distribution="normal"):
    """
    GARCH-type VaR Estimation.

    Fit a GARCH-type model to returns and estimate daily Value-at-Risk (VaR) using empirical quantiles of standardized residuals.

    Model:
    -----
    - Standard GARCH(1,1) recursion for variance:
        σ²ₜ = ω + α * ε²ₜ₋₁ + β * σ²ₜ₋₁
    - Extensions:
        - EGARCH: models log-volatility and captures asymmetry.
        - GJR-GARCH: introduces a leverage effect via threshold terms.
        - APARCH: generalizes to asymmetric power transformations of returns.

    Parameters:
    ----------
    - returns (pd.Series): Daily returns (decimal format, e.g., 0.01 for 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - p (int): GARCH lag order (default = 1).
    - q (int): ARCH lag order (default = 1).
    - vol_model (str): Volatility model ("GARCH", "EGARCH", "GJR", or "APARCH").
    - distribution (str): Innovation distribution ("normal", "t", "ged", or "skewt").

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'Volatility': conditional standard deviation
        - 'Innovations': standardized residuals
        - 'VaR': estimated VaR at each point in time
        - 'VaR Violation': flag where return < -VaR
    - next_day_var (float): 1-day ahead VaR (% absolute value).

    Notes:
    - Returns are internally scaled to percentages for fitting.
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
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# Arch VaR
#----------------------------------------------------------
def var_arch(returns, confidence_level, p=1):
    """
    ARCH-type VaR Estimation.

    Fit an ARCH(p) model to returns and estimate daily Value-at-Risk (VaR) using empirical quantiles of standardized residuals.

    Model:
    - ARCH(p) variance recursion:
        σ²ₜ = ω + α₁ * ε²ₜ₋₁ + α₂ * ε²ₜ₋₂ + ... + αₚ * ε²ₜ₋ₚ

    Parameters:
    - returns (pd.Series): Daily returns (decimal format, e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - p (int): ARCH order (default = 1).

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'Volatility': conditional standard deviation
        - 'Innovations': standardized residuals
        - 'VaR': estimated VaR at each point in time
        - 'VaR Violation': flag where return < -VaR
    - next_day_var (float): 1-day ahead VaR (% absolute value).

    Notes:
    - Returns are internally scaled to percentages for fitting.
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
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data.dropna(), next_day_var


#----------------------------------------------------------
# EWMA VaR
#----------------------------------------------------------
def var_ewma(returns, confidence_level, decay_factor=0.94):
    """
    EWMA-type VaR Estimation.

    Estimate daily Value-at-Risk (VaR) using the Exponentially Weighted Moving Average (EWMA) volatility model and empirical quantiles of standardized residuals.

    Model:
    - EWMA volatility recursion:
        σ²ₜ = λ * σ²ₜ₋₁ + (1 - λ) * ε²ₜ₋₁
    where:
        - λ = decay_factor (e.g., 0.94)
        - εₜ = return at time t

    Parameters:
    - returns (pd.Series): Daily returns (decimal format, e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - decay_factor (float): Smoothing parameter λ (default = 0.94).

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'Volatility': conditional standard deviation (EWMA)
        - 'Innovations': standardized residuals
        - 'VaR': estimated VaR at each time t
        - 'VaR Violation': flag where return < -VaR
    - next_day_var (float): 1-day ahead VaR (% absolute value).

    Notes:
    - Latest volatility estimate is used for next-day VaR.
    - Returns are internally assumed in decimal format (e.g., 0.01 = 1%).
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
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data, next_day_var


#----------------------------------------------------------
# MA VaR
#----------------------------------------------------------
def var_moving_average(returns, confidence_level, window=20):
    """
    Moving Average Volatility VaR Estimation.

    Estimate daily Value-at-Risk (VaR) using a simple rolling window standard deviation model for volatility and empirical quantiles of standardized residuals.

    Model:
    - Volatility is estimated as the rolling standard deviation over a fixed-size window:
        σₜ = std(returnsₜ₋₍window₋₁₎ to returnsₜ)

    Parameters:
    - returns (pd.Series): Daily returns (decimal format, e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - window (int): Size of the moving window for volatility estimation (default = 20).

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'Volatility': rolling standard deviation
        - 'Innovations': standardized residuals (return divided by volatility)
        - 'VaR': estimated VaR at each time t
        - 'VaR Violation': flag where return < -VaR
    - next_day_var (float): 1-day ahead VaR (% absolute value).

    Notes:
    - Latest rolling volatility is used for next-day VaR.
    - Returns are internally assumed in decimal format (e.g., 0.01 = 1%).
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
    next_day_var = 100 * abs(quantile * next_day_vol)

    return result_data, next_day_var


