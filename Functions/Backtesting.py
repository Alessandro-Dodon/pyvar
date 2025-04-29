#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
import pandas as pd

#################################################
# Note: double check/ they can be merged in 1 function!
#################################################

# ----------------------------------------------------------
# VaR Backtesting (General)
# ----------------------------------------------------------
def backtest_var(data, confidence_level):
    """
    VaR Backtesting (Full Sample).

    Count how many times the actual returns exceed (violate) the estimated VaR.

    Parameters:
    - data (pd.DataFrame):
        Must contain columns 'Returns', 'VaR', and 'VaR Violation'.
    - confidence_level (float):
        VaR confidence level (e.g., 0.99).

    Returns:
    - total_violations (int):
        Number of VaR violations observed.
    - violation_rate (float):
        Percentage of days with violations.
    """
    violations = data["VaR Violation"]
    total_violations = violations.sum()
    total_days = len(violations)
    violation_rate = 100 * total_violations / total_days

    return total_violations, violation_rate


# ----------------------------------------------------------
# VaR Backtesting (Subset Period)
# ----------------------------------------------------------
def subset_backtest_var(data, start_date, end_date):
    """
    VaR Backtesting (Subset Period).

    Count VaR violations and violation rate between two specific dates.

    Parameters:
    - data (pd.DataFrame):
        Must contain 'VaR Violation' column with datetime index.
    - start_date (str or pd.Timestamp):
        Start date of the subset (e.g., "2019-01-01").
    - end_date (str or pd.Timestamp):
        End date of the subset (e.g., "2020-01-01").

    Returns:
    - total_violations (int):
        Number of violations in the selected period.
    - violation_rate (float):
        Percentage of violations over the subset.

    Notes:
    - Raises an error if the subset is empty.
    """
    if "VaR Violation" not in data.columns:
        raise ValueError("Data must contain 'VaR Violation' column.")
    
    subset = data.loc[start_date:end_date]
    total_violations = subset["VaR Violation"].sum()
    total_days = len(subset)
    
    if total_days == 0:
        raise ValueError("Subset is empty. Check your date range.")

    violation_rate = 100 * total_violations / total_days
    return total_violations, violation_rate
