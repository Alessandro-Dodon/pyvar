from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd



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



