import numpy as np
import pandas as pd
from arch import arch_model
from scipy.stats import norm, t, gennorm
import warnings

#################################################
# Note: double check all formulas and scaling
#################################################



# Garch Forecast (Analytical Formula, for Variance or VaR)
def garch_forecast(
    returns,
    steps_ahead=10,
    cumulative=False,
    compute_var=False,
    confidence_level=0.99,
    distribution="normal"
):
    """
    Forecast GARCH(1,1) variance or Value-at-Risk (VaR) steps_ahead into the future.

    Parameters
    ----------
    returns : pd.Series
        Daily returns in decimal format (e.g., 0.01 = 1%)
    steps_ahead : int
        Forecast horizon in days
    cumulative : bool
        If True, compute cumulative variance forecast (total variance over horizon)
    compute_var : bool
        If True, return VaR instead of variance
    confidence_level : float
        Confidence level for VaR computation (e.g., 0.99)
    distribution : str
        Distribution for standardized innovations: "normal", "t", or "ged"

    Returns
    -------
    float
        Forecasted variance (decimal) or VaR (percentage).
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



# Garch VaR
def var_garch(returns, confidence_level, p=1, q=1, vol_model="GARCH", distribution="normal"):
    """
    Fit a GARCH-type model and compute empirical daily VaR using standardized residuals.

    Parameters:
    - returns: pd.Series (unscaled daily returns, e.g., in decimal format)
    - confidence_level: float (e.g., 0.99 for 99% VaR)
    - p: int, GARCH lag order (default = 1)
    - q: int, ARCH lag order (default = 1)
    - vol_model: str (default = "GARCH")
        One of:
        - "GARCH": standard symmetric GARCH
        - "EGARCH": exponential GARCH (models log-volatility and asymmetry)
        - "GJR": threshold GARCH (captures leverage effect with a dummy term)
        - "APARCH": asymmetric power ARCH (generalized model with asymmetry + power)
    - distribution: str (default = "normal")
        One of:
        - "normal": standard Gaussian distribution
        - "t": Student's t distribution (fat tails)
        - "ged": Generalized Error Distribution
        - "skewt": Skewed Student's t (fat tails + skewness)

    Returns:
    - result_data: pd.DataFrame with:
        - 'Returns': original returns
        - 'Volatility': estimated conditional volatility
        - 'Innovations': standardized residuals
        - 'VaR': empirical VaR estimate at each time t
        - 'VaR Violation': boolean flag for return < -VaR
    - next_day_var: float, 1-day ahead VaR forecast (absolute % value)
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



# Arch VaR
def var_arch(returns, confidence_level, p=1):
    """
    Compute VaR using ARCH(p) model.

    Parameters:
    - returns: pd.Series (unscaled)
    - confidence_level: float
    - p: ARCH order (default 1)

    Returns:
    - result_data: pd.DataFrame (unscaled)
    - next_day_var: float (unscaled)
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



# EWMA VaR
def var_ewma(returns, confidence_level, decay_factor=0.94):
    """
    Compute VaR using EWMA volatility model.

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - decay_factor: float (lambda), default 0.94

    Returns:
    - result_data: pd.DataFrame
    - next_day_var: float
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



# MA VaR
def var_moving_average(returns, confidence_level, window=20):
    """
    Compute VaR using rolling standard deviation (moving average volatility).

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - window: int, size of rolling window (default 20)

    Returns:
    - result_data: pd.DataFrame
    - next_day_var: float
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


