#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
import pandas as pd

#################################################
# Note: double check
#################################################

# ----------------------------------------------------------
# VaR Backtesting (General or Subset)
# ----------------------------------------------------------
def backtest_var(data, confidence_level, start_date=None, end_date=None):
    """
    VaR Backtesting (Full Sample or Subset Period).

    Count how many times the actual returns exceed (violate) the estimated VaR.

    Parameters:
    - data (pd.DataFrame):
        Must contain columns 'VaR Violation' and optionally 'Returns'.
        Index must be datetime-like for subset testing.

    - confidence_level (float):
        VaR confidence level (e.g., 0.99).

    - start_date (str or pd.Timestamp, optional):
        Start date for subset backtest (default is full sample).

    - end_date (str or pd.Timestamp, optional):
        End date for subset backtest (default is full sample).

    Returns:
    - total_violations (int):
        Number of violations in the selected period.

    - violation_rate (float):
        Percentage of days with violations in the selected period.

    Notes:
    - Raises an error if the subset is empty.
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
