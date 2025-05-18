"""
Price Data and Portfolio Construction Module
-----------------------------------------------

Provides utility functions to download and process financial time series data.
These functions are intended for use with the portfolio, factor model,
simulation, and time-varying correlation modules.

For single-asset analysis (e.g., VaR/ES with univariate models), these tools
are not required, since the input can be a single price or return series.

Each time we will assume daily VaR calculation, so all the data downloaded
with this module is daily.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- get_raw_prices: Download adjusted closing prices using yfinance
- convert_to_base: Convert raw prices to a common base currency
- create_portfolio: Convert prices into monetary exposures using share quantities
- summary_statistics: Compute return series, means, and covariance matrix
- validate_matrix: Run basic stability checks on matrices (e.g., prices, returns, positions)
"""

# TODO: better names for currency conversion function for the output

#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd
import numpy as np


#----------------------------------------------------------
# Checks for Financial Matrices (Shared Function)
#----------------------------------------------------------
def validate_matrix(matrix: pd.DataFrame, context: str = ""):
    """
    Perform basic structural and statistical checks on any matrix 
    used in financial modeling (e.g., prices, positions, returns).

    Parameters
    ----------
    matrix : pd.DataFrame
        Time-indexed matrix with assets as columns.
    context : str, optional
        Context label for warnings (e.g., 'raw prices', 'portfolio').

    Warns
    -----
    - If NaNs are present.
    - If sample size is less than number of columns.
    - If any column has near-zero variance.
    - If the covariance matrix is not positive semi-definite.
    """
    label = f"[{context}]" if context else ""

    if matrix.isnull().any().any():
        print(f"[warning] {label} NaNs detected — clean the data before analysis.")

    n_obs, n_assets = matrix.shape
    if n_obs < n_assets:
        print(f"[warning] {label} Fewer rows ({n_obs}) than columns ({n_assets}) — covariance may be unstable.")

    variances = matrix.var()
    near_zero = variances < 1e-10
    if near_zero.any():
        bad_assets = matrix.columns[near_zero].tolist()
        print(f"[warning] {label} Near-zero variance in: {bad_assets} — may cause instability.")

    cov = matrix.cov().values
    eigvals = np.linalg.eigvalsh(cov)
    if (eigvals < -1e-8).any():
        print(f"[warning] {label} Covariance matrix not PSD — negative eigenvalues detected.")


#----------------------------------------------------------
# Downloading Price Data
# ----------------------------------------------------------
def get_raw_prices(tickers, start="2024-01-01", end=None) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers using yfinance.

    Returns a DataFrame with tickers as columns and dates as index.
    Designed for multi-asset use cases and portfolio modeling.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols (e.g., ['AAPL', 'MSFT', 'BMW.DE']).
    start : str, optional
        Start date in 'YYYY-MM-DD' format. Default is '2024-01-01'.
    end : str, optional
        End date in 'YYYY-MM-DD' format. If None, fetches data up to the most recent available.

    Returns
    -------
    pd.DataFrame
        Adjusted closing prices with rows as dates and columns as tickers.

    Raises
    ------
    Warning
        If matrix structure is unstable or contains issues for downstream analysis.
    """
    prices = (
        yf.download(" ".join(tickers), start=start, end=end,
                    auto_adjust=True, progress=False)["Close"]
        .ffill()
    )

    validate_matrix(prices, context="raw prices")

    return prices


#----------------------------------------------------------
# Currency Conversion
# ----------------------------------------------------------
def convert_to_base(
    raw: pd.DataFrame,
    cur_map: dict = None,
    base: str = "EUR",
    show_currency_detection: bool = True
) -> pd.DataFrame:
    """
    Convert raw prices to a common base currency using FX rates from Yahoo Finance.

    Automatically scales minor units like GBp or ZAc and renames currencies to
    their major equivalents. FX rates are matched by date and applied element-wise.

    Parameters
    ----------
    raw : pd.DataFrame
        Raw price data (tickers as columns, dates as index).
    cur_map : dict, optional
        Mapping from tickers to their native currencies.
        If None, currencies are auto-detected via yfinance.
    base : str, optional
        Target base currency. Default is 'EUR'.
    show_currency_detection : bool, optional
        If True, prints currency detection and FX conversion steps.

    Returns
    -------
    pd.DataFrame
        Converted prices in base currency.

    Raises
    ------
    Warning
        If FX rates cannot be downloaded for some currencies or if currency detection fails.
    """
    # Detect currencies if not provided
    if cur_map is None:
        cur_map = {}
        for t in raw.columns:
            try:
                cur = yf.Ticker(t).fast_info.get("currency", "UNKNOWN") or "UNKNOWN"
            except Exception:
                cur = "UNKNOWN"
            cur_map[t] = cur
            if show_currency_detection:
                print(f"[currency detection] {t}: {cur}")

    # Determine which FX pairs are needed
    needed = {cur_map[t] for t in raw.columns if cur_map[t] not in {base, "UNKNOWN"}}
    fx_pairs = [f"{base}{cur}=X" for cur in needed]

    if fx_pairs:
        if show_currency_detection:
            print(f"[fx download] Downloading FX pairs: {', '.join(fx_pairs)}")
        fx = (
            yf.download(" ".join(fx_pairs), start=raw.index[0],
                        auto_adjust=True, progress=False)["Close"]
            .reindex(raw.index)
            .ffill()
        )
    else:
        fx = pd.DataFrame(index=raw.index)

    # Convert all tickers to base currency
    converted_prices = pd.DataFrame(index=raw.index)
    for t in raw.columns:
        p = raw[t].copy()
        cur = cur_map[t]

        if cur in {"GBp", "GBX", "ZAc"}:
            if show_currency_detection:
                print(f"[unit conversion] {t}: converting from {cur} to major unit")
            p *= 0.01
            cur = "GBP" if cur in {"GBp", "GBX"} else "ZAR"

        if cur not in {base, "UNKNOWN"}:
            pair = f"{base}{cur}=X"
            if pair not in fx.columns:
                if show_currency_detection:
                    print(f"[warning] {t}: FX pair {pair} not found — skipping")
                converted_prices[t] = p
                continue
            rate = fx[pair]
            p = p / rate
            if show_currency_detection:
                print(f"[conversion] {t}: {cur} → {base} via {pair}")

        converted_prices[t] = p

    return converted_prices 


#----------------------------------------------------------
# Create Portfolio from Share Quantities
# ----------------------------------------------------------
def create_portfolio(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """
    Convert asset prices to monetary portfolio exposures using share quantities.

    Tickers in `shares` must match the columns of the price DataFrame.
    Supports fractional and short positions.

    Parameters
    ----------
    prices : pd.DataFrame
        Price data (columns = tickers, index = dates).
    shares : pd.Series
        Shares held per ticker (index must match tickers in prices).

    Returns
    -------
    pd.DataFrame
        Monetary exposures for each asset over time.

    Raises
    ------
    ValueError
        If the tickers in `shares` do not match the price columns.
    ValueError
        If portfolio value is zero or negative on any day.
    Warning
        If matrix structure is unstable or contains issues for downstream analysis.
    """
    if set(prices.columns) != set(shares.index):
        raise ValueError("Mismatch between tickers in 'prices' and 'shares'. "
                         "They must match exactly (names only, order doesn't matter).")

    if len(shares) != prices.shape[1]:
        raise ValueError(f"'shares' length ({len(shares)}) does not match number of tickers in 'prices' ({prices.shape[1]})")

    shares_aligned = shares.reindex(prices.columns)
    positions = prices.multiply(shares_aligned, axis=1)

    portfolio_value = positions.sum(axis=1)
    if (portfolio_value <= 0).any():
        raise ValueError("Portfolio has zero or negative total value on some dates — adjust share allocations.")

    # Shared validation for statistical stability
    validate_matrix(positions, context="portfolio positions")

    return positions


#----------------------------------------------------------
# Summary Statistics
# ----------------------------------------------------------
def summary_statistics(matrix: pd.DataFrame): 
    """
    Compute daily returns, mean returns, and the return covariance matrix.

    This function works both with raw asset prices and with monetary positions
    obtained by multiplying prices by fixed share quantities. Since percentage 
    returns are unaffected by fixed multipliers, the output will be the same in 
    both cases if no trading occurs (i.e., shares are constant over time).

    This makes the function flexible: users can apply it after building a 
    monetary portfolio or after converting prices to a base currency, 
    without affecting the computed return series or statistics.

    Parameters
    ----------
    matrix : pd.DataFrame
        Price matrix or monetary position matrix with tickers as columns and dates as index.

    Returns
    -------
    returns : pd.DataFrame
        Daily percentage returns per asset.
    mean_returns : pd.Series
        Mean daily return per asset.
    covariance_matrix : pd.DataFrame
        Covariance matrix of the return series.

    Raises
    ------
    Warning
        If matrix structure is unstable or contains issues for downstream analysis.

    Notes
    -----
    - If shares are constant over time, returns and return-based statistics
    (means, covariance matrix) will be exactly the same as those computed 
    from raw prices.
    - If shares vary over time (e.g., due to trading or rebalancing),
    the return series will differ.
    """
    returns = matrix.pct_change().dropna()

    # Properly validate the returns matrix now
    validate_matrix(returns, context="summary statistics returns")

    mean_returns = returns.mean()
    covariance_matrix = returns.cov()

    return returns, mean_returns, covariance_matrix
