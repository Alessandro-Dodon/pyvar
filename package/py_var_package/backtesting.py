#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import chi2

#################################################
# Note: double check
#################################################

# ----------------------------------------------------------
# Counting VaR Violations (General or Subset)
# ----------------------------------------------------------
def count_violations(data, start_date=None, end_date=None):
    """
    Count VaR violations over a full or partial time window.

    This function evaluates how often actual returns exceed the Value-at-Risk 
    threshold by summing 'VaR Violation' entries. It supports both full-sample 
    and subset-based backtesting for flexibility in analysis.

    Parameters:
    - data (pd.DataFrame): 
        DataFrame containing a binary column 'VaR Violation'.
        Index must be datetime-like if subsetting is used.
    - start_date (str or pd.Timestamp, optional): 
        Start date for subsetting (default is beginning of dataset).
    - end_date (str or pd.Timestamp, optional): 
        End date for subsetting (default is end of dataset).

    Returns:
    - total_violations (int): Number of VaR violations.
    - violation_rate (float): Violation rate as a percentage.

    Raises:
    - ValueError: If 'VaR Violation' column is missing or selected range is empty.
    """
    if "VaR Violation" not in data.columns:
        raise ValueError("Data must contain 'VaR Violation' column.")

    subset = data.copy()

    if start_date is not None or end_date is not None:
        subset = subset.loc[start_date:end_date]

    if subset.empty:
        raise ValueError("Subset is empty. Check your date range.")

    violations = subset["VaR Violation"]
    total_violations = violations.sum()
    total_days = len(violations)
    violation_rate = 100 * total_violations / total_days

    return total_violations, violation_rate


# ----------------------------------------------------------
# Kupiec Unconditional Coverage Test
# ----------------------------------------------------------
def kupiec_test(total_violations, total_days, confidence_level):
    """
    Kupiec's likelihood ratio test for unconditional coverage.

    This test evaluates whether the observed number of Value-at-Risk (VaR) 
    violations is consistent with the expected frequency implied by the confidence level. 
    It assumes violations follow an independent Bernoulli process.

    A rejection of the null suggests the model under- or overestimates risk — 
    i.e., the frequency of observed violations is statistically inconsistent with 
    the expected failure rate.

    Parameters:
    - total_violations (int): Observed number of VaR breaches.
    - total_days (int): Number of observations in the test period.
    - confidence_level (float): Confidence level used to compute VaR (e.g., 0.99).

    Returns:
    - dict: {
        'LR_uc': Likelihood ratio test statistic,
        'p_value': Associated p-value,
        'reject_null': True if test rejects null at 5% level
      }
    """
    if total_violations == 0 or total_violations == total_days:
        return {
            "LR_uc": np.nan,
            "p_value": np.nan,
            "reject_null": "undefined (perfect or null failure rate)"
        }

    failure_prob = 1 - confidence_level
    observed_prob = total_violations / total_days

    log_likelihood_null = (total_violations * np.log(failure_prob)
                          + (total_days - total_violations) * np.log(1 - failure_prob))

    log_likelihood_alt = (total_violations * np.log(observed_prob)
                         + (total_days - total_violations) * np.log(1 - observed_prob))

    LR_uc = -2 * (log_likelihood_null - log_likelihood_alt)
    p_value = 1 - chi2.cdf(LR_uc, df=1)
    reject = LR_uc > chi2.ppf(0.95, df=1)

    return {
        "LR_uc": LR_uc,
        "p_value": p_value,
        "reject_null": reject
    }


# ----------------------------------------------------------
# Christoffersen Independence Test
# ----------------------------------------------------------
def christoffersen_test(violations):
    """
    Christoffersen's likelihood ratio test for independence of exceptions.

    This test evaluates whether VaR violations are independently distributed over time 
    using a first-order Markov transition matrix. It checks for clustering of violations, 
    which may indicate that the risk model fails to capture time-varying volatility 
    or other forms of dependence.

    A rejection of the null suggests that violations are not independent — 
    typically interpreted as evidence of model misspecification or missing dynamics.

    Parameters:
    - violations (pd.DataFrame, pd.Series, list, or array):
        If DataFrame, must include a column named 'VaR Violation'.
        Otherwise, should be a 1D sequence of binary or boolean values.

    Returns:
    - dict: {
        'LR_c': Likelihood ratio test statistic,
        'p_value': Associated p-value,
        'reject_null': True if test rejects null at 5% level
      }

    Raises:
    - ValueError: If DataFrame lacks 'VaR Violation' column.
    """
    # Automatically extract from DataFrame if needed
    if isinstance(violations, pd.DataFrame):
        if "VaR Violation" not in violations.columns:
            raise ValueError("DataFrame must contain 'VaR Violation' column.")
        violations = violations["VaR Violation"]

    # Ensure binary 0/1 format
    violations = np.asarray(violations).astype(int)

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(violations)):
        prev, curr = violations[t - 1], violations[t]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_likelihood_null = ((n01 + n11) * np.log(pi) +
                           (n00 + n10) * np.log(1 - pi)) if 0 < pi < 1 else -np.inf

    log_likelihood_alt = 0
    if 0 < pi_0 < 1:
        log_likelihood_alt += n01 * np.log(pi_0) + n00 * np.log(1 - pi_0)
    if 0 < pi_1 < 1:
        log_likelihood_alt += n11 * np.log(pi_1) + n10 * np.log(1 - pi_1)

    LR_c = -2 * (log_likelihood_null - log_likelihood_alt)
    p_value = 1 - chi2.cdf(LR_c, df=1)
    reject = LR_c > chi2.ppf(0.95, df=1)

    return {
        "LR_c": LR_c,
        "p_value": p_value,
        "reject_null": reject
    }


# ----------------------------------------------------------
# Joint Test (Kupiec + Christoffersen)
# ----------------------------------------------------------
def joint_lr_test(LR_uc, LR_c):
    """
    Joint likelihood ratio test for coverage and independence of VaR violations.

    This test combines the Kupiec test for correct violation frequency (unconditional coverage) 
    and the Christoffersen test for independence of violations (no clustering). It provides 
    a unified assessment of whether a risk model is correctly calibrated in both dimensions.

    A rejection of the null indicates that either the number or the timing of exceptions — 
    or both — are statistically inconsistent with the model's assumptions.

    Parameters:
    - LR_uc (float): Likelihood ratio from the unconditional coverage test.
    - LR_c (float): Likelihood ratio from the independence test.

    Returns:
    - dict: {
        'LR_total': Combined LR statistic (χ² with 2 degrees of freedom),
        'p_value': Associated p-value,
        'reject_null': True if joint test rejects null at 5% level
      }
    """
    LR_total = LR_uc + LR_c
    p_value = 1 - chi2.cdf(LR_total, df=2)
    reject = LR_total > chi2.ppf(0.95, df=2)

    return {
        "LR_total": LR_total,
        "p_value": p_value,
        "reject_null": reject
    }
