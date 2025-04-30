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
# Historical Expected Shortfall (Tail Mean)
#----------------------------------------------------------
def compute_es_historical(result_data, confidence_level):
    """
    Historical Expected Shortfall Estimation (Tail Mean Method).

    Compute Expected Shortfall (ES) non-parametrically as the average of returns 
    that fall below the Value-at-Risk (VaR) threshold.

    Description:
    - ES represents the expected loss given that the return exceeds the VaR threshold.
    - Computed by averaging all returns lower than -VaR (left tail of the distribution).

    Formula:
    - Expected Shortfall (ES):
        ES = - E[ Return | Return < -VaR ]

    Parameters:
    - result_data (pd.DataFrame):
        Must contain 'Returns' and 'VaR' columns.
        Assumes 'VaR' is constant over time (i.e., historical VaR).
        Returns and VaR must be in decimal format (e.g., 0.01 = 1%).

    - confidence_level (float):
        Confidence level for VaR/ES (e.g., 0.99 for 99%).

    Returns:
    - result_data (pd.DataFrame):
        DataFrame with a new column 'ES' (constant series, in decimals).

    - es_estimate (float):
        Scalar ES estimate (positive decimal loss magnitude, e.g., 0.015 for 1.5%).

    Notes:
    - ES is expressed as a positive number, representing the expected loss.
    - Returns and VaR must be expressed in decimal format throughout.
    """
    var_threshold = result_data["VaR"].iloc[0]
    tail_returns = result_data["Returns"][result_data["Returns"] < -var_threshold]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = tail_returns.mean()

    result_data["ES"] = pd.Series(es_value * -1, index=result_data.index)
    es_estimate = result_data["ES"].iloc[0]
    return result_data, es_estimate


#----------------------------------------------------------
# Parametric Expected Shortfall (Normal and t only)
#----------------------------------------------------------
def compute_es_parametric(result_data, returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Parametric Expected Shortfall Estimation (Normal or Student-t).

    Estimate Expected Shortfall (ES) analytically under the assumption 
    that returns follow either a Normal or Student-t distribution.

    Description:
    - ES represents the expected loss conditional on returns exceeding the VaR threshold.
    - Computed from the assumed distribution's PDF and quantile function.

    Formulas:
    - Normal:
        ES = σ × φ(z) / (1 - α)
    - Student-t:
        ES = scale × pdf(tₐ, df) × (df + tₐ²) / [(df - 1)(1 - α)]

    where:
        - σ = standard deviation of returns
        - φ(z) = standard normal PDF evaluated at z
        - tₐ = Student-t quantile at level α
        - pdf(tₐ, df) = Student-t PDF at tₐ
        - α = confidence level (e.g., 0.99)

    Parameters:
    - result_data (pd.DataFrame): DataFrame to update with an 'ES' column (same length as returns).
    - returns (pd.Series): Return series used to estimate parameters. Must be in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR/ES (e.g., 0.99 for 99%).
    - holding_period (int): Holding period in days (default = 1). Assumes i.i.d. scaling via sqrt(T).
    - distribution (str): Distribution to use: "normal" or "t" (Student-t).

    Returns:
    - result_data (pd.DataFrame): Updated with constant 'ES' column (in decimals, e.g., 0.012 = 1.2%).
    - es_estimate (float): Scalar ES value, in decimal format (positive loss magnitude).

    Notes:
    - ES is returned as a positive number (expected loss).
    - Returns and ES are expressed in decimal units.
    - The estimate is scaled for multi-day holding periods assuming i.i.d. returns.
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
    es_estimate = es_value

    return result_data, es_estimate


#----------------------------------------------------------
# Expected Shortfall (Volatility)
#----------------------------------------------------------
def compute_expected_shortfall_volatility(data, confidence_level, subset=None):
    """
    Volatility-Based Expected Shortfall Estimation (Empirical Innovations Method).

    Estimate Expected Shortfall (ES) dynamically over time using the empirical 
    average of standardized residuals (innovations) in the left tail, scaled by 
    the model-implied conditional volatility.

    Description:
    - ES is computed by averaging innovations below a quantile threshold, then scaling 
      this average (the tail mean) by the conditional volatility at each time point.
    - This allows capturing time-varying risk levels.

    Formulas:
    - Tail Mean:
        TailMean = mean(Innovationsₜ where Innovationsₜ < Quantile(Innovations, 1 - α))
    - Time-varying ES:
        ESₜ = - Volatilityₜ × TailMean

    Parameters:
    - data (pd.DataFrame): Must contain:
        - 'Innovations': standardized residuals (mean ≈ 0, std ≈ 1)
        - 'Volatility': conditional standard deviation (in decimal units)
    - confidence_level (float): Confidence level α for ES (e.g., 0.99 for 99%).
    - subset (tuple, optional): A date range (start_date, end_date) used to compute TailMean.

    Returns:
    - data (pd.DataFrame): Original DataFrame with an added 'ES' column (in decimals, e.g., 0.012 = 1.2%).

    Notes:
    - Returns and volatility are assumed in decimal format (e.g., 0.01 = 1%).
    - The ES column represents a positive loss magnitude at each time point.
    - The same tail mean is used for all rows unless a subset is specified.
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


#----------------------------------------------------------
# Wealth Scaling for ES
#----------------------------------------------------------
def apply_wealth_scaling_es(result_data, es_estimate=None, wealth=None):
    """
    Apply wealth scaling to Expected Shortfall (ES) values.

    Parameters:
    - result_data (pd.DataFrame): Output of an ES function, must include an 'ES' column (in decimals, e.g., 0.01 = 1%).
    - es_estimate (float or None): Scalar ES value to be scaled (in decimals).
    - wealth (float or None): Portfolio value used for converting ES from percentage (decimal) to monetary units.

    Returns:
    - result_data (pd.DataFrame): DataFrame with optional new column 'ES_monetary' (in monetary units).
    - es_estimate (float or None): Scaled scalar ES in monetary units (if provided), else None.

    Notes:
    - Does not overwrite the 'ES' column; only adds 'ES_monetary'.
    - Assumes all ES values are in decimal format (e.g., 0.01 = 1%).
    """
    if wealth is not None and "ES" in result_data.columns:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    if wealth is not None and es_estimate is not None:
        es_estimate = es_estimate * wealth

    return result_data, es_estimate


#----------------------------------------------------------
# Unified Expected Shortfall (ES) Caller with Wealth Scaling
#----------------------------------------------------------
def compute_expected_shortfall(method, result_data, confidence_level=0.99, wealth=None, **kwargs):
    """
    Unified Expected Shortfall (ES) Estimator with Optional Wealth Scaling.

    Dispatches ES computation to one of three methods:
    - 'historical': empirical tail average based on past returns.
    - 'parametric': closed-form ES using Normal or Student-t distribution.
    - 'volatility': time-varying ES from scaled innovation tails.

    Parameters:
    - method (str): One of {"historical", "parametric", "volatility"}.
    - result_data (pd.DataFrame): DataFrame with required columns depending on method:
        - 'Returns' and 'VaR' for 'historical'
        - 'Returns' for 'parametric'
        - 'Volatility' and 'Innovations' for 'volatility'
    - confidence_level (float): ES confidence level (e.g., 0.99).
    - wealth (float or None): If provided, scales ES results to monetary units.
    - **kwargs: Additional arguments:
        - 'returns' (pd.Series) for 'parametric'
        - 'holding_period' (int) for 'parametric' (default=1)
        - 'distribution' (str) for 'parametric' (default="normal")
        - 'subset' (tuple) for 'volatility' (optional date range)

    Returns:
    - result_data (pd.DataFrame): Contains 'ES' (in decimals) and optionally 'ES_monetary' (monetary units).
    - es_estimate (float or None): Scalar ES value if applicable; None for time-varying ES.

    Notes:
    - All ES values are computed in decimal format (e.g., 0.012 = 1.2%).
    - Wealth scaling affects both scalar ES and time series ES (if applicable).
    """

    method = method.lower()

    if method == "historical":
        result_data, es_estimate = compute_es_historical(result_data, confidence_level)

    elif method == "parametric":
        returns = kwargs.get("returns")
        if returns is None:
            raise ValueError("Parametric ES requires 'returns' (pd.Series).")
        holding_period = kwargs.get("holding_period", 1)
        distribution = kwargs.get("distribution", "normal")
        result_data, es_estimate = compute_es_parametric(
            result_data, returns, confidence_level, holding_period, distribution
        )

    elif method == "volatility":
        subset = kwargs.get("subset", None)
        result_data = compute_expected_shortfall_volatility(result_data, confidence_level, subset)
        es_estimate = None

    else:
        raise ValueError("Method must be one of: 'historical', 'parametric', or 'volatility'")

    # Apply wealth scaling
    result_data, es_estimate = apply_wealth_scaling_es(result_data, es_estimate, wealth)

    return result_data, es_estimate
