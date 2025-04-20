import numpy as np
import pandas as pd
from arch import arch_model



# Garch VaR
def var_garch(returns, confidence_level, p=1, q=1):
    """
    Fit a GARCH(p,q) model and compute empirical daily VaR.

    Parameters:
    - returns: pd.Series (unscaled)
    - confidence_level: float
    - p, q: GARCH orders (default 1,1)

    Returns:
    - result_data: pd.DataFrame (unscaled)
    - next_day_var: float (unscaled)
    """
    returns_scaled = returns * 100  # scale for stability

    garch_model = arch_model(returns_scaled, vol="Garch", p=p, q=q)
    garch_fit = garch_model.fit(disp="off")

    volatility = garch_fit.conditional_volatility / 100  # rescale
    innovations = returns / volatility  # use original returns for innovations
    quantile = np.percentile(innovations.dropna(), 100 * (1 - confidence_level))
    var_series = -volatility * quantile

    result_data = pd.DataFrame({
        "Returns": returns,
        "Volatility": volatility,
        "Innovations": innovations,
        "VaR": var_series
    })
    result_data["VaR Violation"] = returns < -var_series

    next_day_vol = (garch_fit.forecast(horizon=1).variance.values[-1][0] ** 0.5) / 100
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



# Expected Shortfall (Volatility)
def compute_expected_shortfall_volatility(data, confidence_level, subset=None):
    """
    Compute Expected Shortfall (ES) based on empirical innovations.

    Parameters:
    - data: pd.DataFrame with 'Innovations' and 'Volatility'
    - confidence_level: float
    - subset: tuple (start_date, end_date) or None

    Returns:
    - data: pd.DataFrame with new 'ES' column (full period)
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")
    
    subset_data = data.copy()
    if subset is not None:
        subset_data = data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"].dropna(), 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    data["ES"] = -data["Volatility"] * tail_mean
    return data



