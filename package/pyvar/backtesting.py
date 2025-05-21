"""
Backtesting Module
------------------

Implements statistical tests for validating Value-at-Risk (VaR) forecasts.

This module provides tools to count VaR violations and to formally evaluate
the statistical properties of those violations using likelihood ratio tests.
Specifically, it includes:

- The Kupiec test for unconditional coverage (correct frequency of exceptions)
- The Christoffersen test for independence (no clustering of violations)
- A joint test combining both criteria

These tools help assess whether a risk model is correctly calibrated in terms
of both the quantity and timing of its risk forecasts.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- count_violations: Compute number and rate of VaR violations in a time window
- kupiec_test: Likelihood ratio test for unconditional coverage
- christoffersen_test: Likelihood ratio test for independence of violations
- joint_lr_test: Combined test for both coverage and independence
"""

# TODO: check formulas

#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import chi2


# ----------------------------------------------------------
# Counting VaR Violations (General or Subset)
# ----------------------------------------------------------
def count_violations(result_data, start_date=None, end_date=None):
    """
    Main
    ----
    Count Value-at-Risk (VaR) violations over time.

    Computes how often actual portfolio returns exceed the estimated VaR threshold,
    using the binary 'VaR Violation' column. Supports full-sample or date-subset analysis.

    Parameters
    ----------
    result_data : pd.DataFrame
        DataFrame containing a binary column 'VaR Violation'.
        Index must be datetime-like if date subsetting is used.

    start_date : str or pd.Timestamp, optional
        Start date for subsetting. If None, uses full range.

    end_date : str or pd.Timestamp, optional
        End date for subsetting. If None, uses full range.

    Returns
    -------
    total_violations : int
        Number of VaR violations during the selected period.

    violation_rate : float
        Violation rate as a decimal (e.g., 0.02 = 2%).

    Raises
    ------
    ValueError
        If 'VaR Violation' column is missing or the selected subset is empty.
    """
    if "VaR Violation" not in result_data.columns:
        raise ValueError("Data must contain 'VaR Violation' column.")

    subset = result_data.copy()
    if start_date is not None or end_date is not None:
        subset = subset.loc[start_date:end_date]

    if subset.empty:
        raise ValueError("Subset is empty. Check your date range.")

    violations = subset["VaR Violation"]
    total_violations = violations.sum()
    total_days = len(violations)
    violation_rate = total_violations / total_days

    return total_violations, violation_rate


# ----------------------------------------------------------
# Kupiec Unconditional Coverage Test
# ----------------------------------------------------------
def kupiec_test(total_violations, total_days, confidence_level):
    """
    Main
    ----
    This is the Kupiec Unconditional Coverage Test.

    Evaluates whether the observed number of Value-at-Risk (VaR) violations is consistent 
    with the expected frequency implied by the chosen confidence level. Assumes i.i.d. 
    Bernoulli violations. Rejection implies miscalibrated VaR — either under- or 
    overestimating tail risk.

    Parameters
    ----------
    total_violations : int
        Observed number of VaR violations.

    total_days : int
        Total number of observations in the backtest period.

    confidence_level : float
        Confidence level used to compute VaR (e.g., 0.99).

    Returns
    -------
    dict
        Dictionary with:
        - 'LR_uc': Likelihood ratio test statistic
        - 'p_value': p-value under χ²(1)
        - 'reject_null': True if null is rejected at 5% level
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
    Main
    ----
    This is the Christoffersen Independence Test.

    Tests whether VaR violations are independent over time using a 2-state Markov 
    transition matrix. Rejection indicates clustering of exceptions, suggesting 
    model misspecification or missing dynamics (e.g., volatility persistence).

    Parameters
    ----------
    violations : array-like or pd.DataFrame
        Sequence of 0/1 indicators for VaR violations. If DataFrame, must contain 
        a column named 'VaR Violation'.

    Returns
    -------
    dict
        Dictionary with:
        - 'LR_c': Likelihood ratio test statistic
        - 'p_value': p-value under χ²(1)
        - 'reject_null': True if null is rejected at 5% level
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
    Main
    ----
    This is the Joint Likelihood Ratio Test (Kupiec + Christoffersen)

    Combines unconditional coverage and independence tests to assess whether a 
    VaR model produces both the correct number and timing of violations. 
    Rejection suggests the model fails on at least one of the two dimensions.

    Parameters
    ----------
    LR_uc : float
        Likelihood ratio from the unconditional coverage test (Kupiec).

    LR_c : float
        Likelihood ratio from the independence test (Christoffersen).

    Returns
    -------
    dict
        Dictionary with:
        - 'LR_total': Combined LR statistic (χ² with 2 df)
        - 'p_value': p-value under χ²(2)
        - 'reject_null': True if null is rejected at 5% level
    """
    LR_total = LR_uc + LR_c
    p_value = 1 - chi2.cdf(LR_total, df=2)
    reject = LR_total > chi2.ppf(0.95, df=2)

    return {
        "LR_total": LR_total,
        "p_value": p_value,
        "reject_null": reject
    }
