#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd

#################################################
# Note: double check all formulas 
#       add caller (meta) function?
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
    - ES represents the expected loss given that the loss exceeds VaR.
    - Computed by averaging all returns lower than the VaR estimate.

    Formulas:
    - Expected Shortfall (ES):
        ES = - E[ Return | Return < -VaR ]

    Parameters:
    - result_data (pd.DataFrame):
        Must contain 'Returns' and 'VaR' columns.
        'VaR' is assumed constant over time (as in historical VaR).

    - confidence_level (float):
        Confidence level for VaR/ES (e.g., 0.99 for 99% ES).

    Returns:
    - result_data (pd.DataFrame):
        Original DataFrame with an additional 'ES' column (constant over time).

    - es_estimate (float):
        Expected Shortfall estimate as a positive % value.

    Notes:
    - ES is computed based on the historical empirical distribution.
    - Returns must be in decimal format (e.g., 0.01 = 1%).
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


#----------------------------------------------------------
# Parametric Expected Shortfall (Normal and t only)
#----------------------------------------------------------
def compute_es_parametric(result_data, returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Parametric Expected Shortfall Estimation.

    Compute Expected Shortfall (ES) under the assumption that returns follow a Normal or Student-t distribution.

    Description:
    - ES represents the expected loss conditional on exceeding the VaR threshold.
    - Computed analytically from the probability density and cumulative distribution of the assumed model.

    Formulas:
    - Normal distribution:
        ES = σ × φ(z) / (1 - α)
    - Student-t distribution:
        ES = scale × pdf(tₐ, df) × (df + tₐ²) / [(df - 1)(1 - α)]

    where:
        - σ = standard deviation
        - φ(z) = standard normal pdf evaluated at quantile z
        - tₐ = Student-t quantile at level α
        - pdf(tₐ, df) = Student-t pdf evaluated at tₐ
        - α = confidence level (e.g., 0.99)

    Parameters:
    - result_data (pd.DataFrame):
        DataFrame to update with an 'ES' column.

    - returns (pd.Series):
        Return series used for distribution fitting.

    - confidence_level (float):
        Confidence level for VaR/ES (e.g., 0.99).

    - holding_period (int, optional):
        Holding period in days (default = 1).

    - distribution (str, optional):
        Distribution to assume: "normal" or "t" (Student-t).

    Returns:
    - result_data (pd.DataFrame):
        Original DataFrame updated with a constant 'ES' column (flat over time).

    - es_estimate (float):
        Estimated Expected Shortfall (positive % loss magnitude).

    Notes:
    - Assumes i.i.d. returns for scaling over multiple days.
    - Returns must be provided in decimal format (e.g., 0.01 = 1%).
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


#----------------------------------------------------------
# Expected Shortfall (Volatility)
#----------------------------------------------------------
def compute_expected_shortfall_volatility(data, confidence_level, subset=None):
    """
    Expected Shortfall Estimation Based on Empirical Innovations.

    Compute Expected Shortfall (ES) dynamically over time by scaling the empirical tail mean of standardized innovations.

    Description:
    - ES is computed from the average of innovations below the quantile corresponding to the confidence level.
    - The tail mean is then scaled by the conditional volatility at each point in time.

    Formulas:
    - Tail Mean:
        TailMean = mean(Innovationsₜ where Innovationsₜ < Quantile(Innovations, 1 - confidence_level))
    - Expected Shortfall at time t:
        ESₜ = - Volatilityₜ × TailMean

    Parameters:
    - data (pd.DataFrame):
        Time series containing 'Innovations' (standardized residuals) and 'Volatility' (conditional standard deviation).

    - confidence_level (float):
        Confidence level for ES estimation (e.g., 0.99).

    - subset (tuple, optional):
        (start_date, end_date) to restrict the innovation sample for TailMean computation.

    Returns:
    - data (pd.DataFrame):
        Original DataFrame updated with an 'ES' column.

    Notes:
    - Innovations are assumed standardized (mean 0, variance 1).
    - Scaling by volatility captures time-varying risk.
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



