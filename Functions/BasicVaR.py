from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd



# Historical VaR (Non-Parametric)
def var_historical(returns, confidence_level, holding_period=1):
    """
    Compute Historical (Non-Parametric) VaR in %.

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - holding_period: int

    Returns:
    - result_data: pd.DataFrame with 'Returns', 'VaR', 'VaR Violation'
    - var_estimate: float (in %)
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



# Parametric VaR (i.i.d. assumption)
def var_parametric_iid(returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Compute Parametric i.i.d. VaR using Normal, Student-t, or GED distribution.

    Parameters:
    - returns: pd.Series
    - confidence_level: float
    - holding_period: int
    - distribution: str ("normal", "t", "ged")

    Returns:
    - result_data: pd.DataFrame with 'Returns', 'VaR', 'VaR Violation'
    - var_estimate: float (in %)
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



# Historical Expected Shortfall (Tail Mean)
def compute_es_historical(result_data, confidence_level):
    """
    Compute Historical Expected Shortfall (mean of returns below VaR).

    Parameters:
    - result_data: pd.DataFrame with 'Returns' and 'VaR'

    Returns:
    - result_data: pd.DataFrame with added 'ES' column (flat, negative)
    - es_estimate: float (in %)
    """
    var_threshold = result_data["VaR"].iloc[0]
    tail_returns = result_data["Returns"][result_data["Returns"] < -var_threshold]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = tail_returns.mean()

    result_data["ES"] = pd.Series(es_value * -1, index=result_data.index)
    es_estimate = 100 * result_data["ES"].iloc[0]
    return result_data, es_estimate



# Parametric Expected Shortfall (Normal and t only)
def compute_es_parametric(result_data, returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Compute Expected Shortfall (ES) for Parametric VaR model using Normal or Student-t.

    Parameters:
    - result_data: pd.DataFrame to update with 'ES' column
    - returns: pd.Series used to fit the distribution
    - confidence_level: float
    - holding_period: int
    - distribution: str ("normal", "t")

    Returns:
    - result_data: pd.DataFrame with added 'ES' column (flat, negative)
    - es_estimate: float (in %)
    """
    returns_clean = returns.dropna()
    alpha = confidence_level

    if distribution == "normal":
        std_dev = returns_clean.std()
        z = norm.ppf(alpha)
        es_raw = std_dev * norm.pdf(z) / (1 - alpha)
        es_value = es_raw * np.sqrt(holding_period)

    elif distribution == "t":
        df, loc, scale = t.fit(returns_clean)
        t_alpha = t.ppf(alpha, df)
        pdf_val = t.pdf(t_alpha, df)
        factor = (df + t_alpha**2) / (df - 1)
        es_raw = scale * pdf_val * factor / (1 - alpha)
        es_value = es_raw * np.sqrt(holding_period)

    else:
        raise ValueError("Supported distributions: 'normal', 't'")

    result_data["ES"] = pd.Series(es_value, index=result_data.index)
    es_estimate = 100 * es_value

    return result_data, es_estimate
