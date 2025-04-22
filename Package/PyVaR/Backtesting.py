import numpy as np
import pandas as pd



# VaR Backtesting (General)
def backtest_var(data, confidence_level):
    """
    Backtest VaR by counting violations.

    Parameters:
    - data: pd.DataFrame with 'Returns', 'VaR', 'VaR Violation'
    - confidence_level: float

    Returns:
    - total_violations: int
    - violation_rate: float
    """
    violations = data["VaR Violation"]
    total_violations = violations.sum()
    total_days = len(violations)
    violation_rate = 100 * total_violations / total_days

    return total_violations, violation_rate



def subset_backtest_var(data, start_date, end_date):
    """
    Count VaR violations and violation rate within a specific date subset.

    Parameters:
    - data: pd.DataFrame with 'VaR Violation' column and a datetime index
    - start_date: str or pd.Timestamp, e.g. "2019-01-01"
    - end_date: str or pd.Timestamp, e.g. "2020-01-01"

    Returns:
    - total_violations: int
    - violation_rate: float (%)
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
