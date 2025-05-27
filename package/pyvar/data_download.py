"""
Price Data and Portfolio Construction Module
-----------------------------------------------

Provides utility functions to download and process financial time series data.
These functions are intended for use with the portfolio, factor model,
simulation, and time-varying correlation modules.

For single-asset analysis (e.g., VaR/ES with univariate models), these tools
are not required, since the input can be a single price or return series.

Each time we will assume daily VaR calculation, so all the data downloaded
with this module is daily. This module also partially helps in pre-processing
the data.

Notice that all the risk management measures adopted in the other modules
assume a "buy and hold" strategy". If shares change suddently, the risk measure
must be recalculated. 

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
- compute_returns: Compute percentage returns
- validate_matrix: Run basic stability checks on matrices (e.g., prices, returns, positions)
"""


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
    Main
    ----
    Perform basic structural and statistical checks on any matrix 
    used in financial modeling (e.g., prices, positions, returns).

    If you don't download the data using our functions, like using 
    your own csv file, you can use this function to check the data
    immediately. We don't reccomend however for our basic applications
    to use portfolios with overall negative values (complete shorts).
    Also, notice that a near zero value of portfolio (perfect hedge)
    may be a problem in some other functions. 

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

    n_observations, n_assets = matrix.shape 
    if n_observations < n_assets:
        print(f"[warning] {label} Fewer rows ({n_observations}) than columns ({n_assets}) — covariance may be unstable.")

    variances = matrix.var()
    near_zero = variances < 1e-10
    if near_zero.any():
        bad_assets = matrix.columns[near_zero].tolist()
        print(f"[warning] {label} Near-zero variance in: {bad_assets} — may cause instability.")

    cov = matrix.cov().values
    eigenvalues = np.linalg.eigvalsh(cov)
    if (eigenvalues < -1e-8).any():
        print(f"[warning] {label} Covariance matrix not PSD — negative eigenvalues detected.")


#----------------------------------------------------------
# Downloading Price Data
# ----------------------------------------------------------
def get_raw_prices(tickers, start="2024-01-01", end=None) -> pd.DataFrame:
    """
    Main
    ----
    Download daily adjusted closing prices for a list of tickers using yfinance.

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
    raw_prices = (
        yf.download(" ".join(tickers), start=start, end=end,
                    auto_adjust=True, progress=False)["Close"]
        .ffill()
    )

    validate_matrix(raw_prices, context="raw prices")

    return raw_prices


#----------------------------------------------------------
# Currency Conversion
# ----------------------------------------------------------
def convert_to_base(
    raw_prices: pd.DataFrame,
    currency_mapping: dict = None,
    base_currency: str = "EUR",
    show_currency_detection: bool = True
) -> pd.DataFrame:
    """
    Main
    ----
    Convert raw prices to a common base currency using FX rates from Yahoo Finance.

    Automatically scales minor units like GBp or ZAc and renames currencies to
    their major equivalents. FX rates are matched by date and applied element-wise.

    Parameters
    ----------
    raw_prices : pd.DataFrame
        Raw price data (tickers as columns, dates as index).
    currency_mapping : dict, optional
        Mapping from tickers to their native currencies.
        If None, currencies are auto-detected via yfinance.
    base_currency : str, optional
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
    if currency_mapping is None:
        currency_mapping = {}
        for ticker in raw_prices.columns:
            try:
                native_currency = yf.Ticker(ticker).fast_info.get("currency", "UNKNOWN") or "UNKNOWN"
            except Exception:
                native_currency = "UNKNOWN"
            currency_mapping[ticker] = native_currency
            if show_currency_detection:
                print(f"[currency detection] {ticker}: {native_currency}")

    # Determine which FX pairs are needed
    needed_currencies = {currency_mapping[ticker] for ticker in raw_prices.columns if currency_mapping[ticker] not in {base_currency, "UNKNOWN"}}
    fx_pair_list = [f"{base_currency}{currency}=X" for currency in needed_currencies]

    if fx_pair_list:
        if show_currency_detection:
            print(f"[fx download] Downloading FX pairs: {', '.join(fx_pair_list)}")
        fx_rates = (
            yf.download(" ".join(fx_pair_list), start=raw_prices.index[0],
                        auto_adjust=True, progress=False)["Close"]
            .reindex(raw_prices.index)
            .ffill()
        )
    else:
        fx_rates = pd.DataFrame(index=raw_prices.index)

    # Convert all tickers to base currency
    converted_prices = pd.DataFrame(index=raw_prices.index)
    for ticker in raw_prices.columns:
        my_price = raw_prices[ticker].copy()
        native_currency = currency_mapping[ticker]

        if native_currency in {"GBp", "GBX", "ZAc"}:
            if show_currency_detection:
                print(f"[unit conversion] {ticker}: converting from {native_currency} to major unit")
            my_price *= 0.01
            native_currency = "GBP" if native_currency in {"GBp", "GBX"} else "ZAR"

        if native_currency not in {base_currency, "UNKNOWN"}:
            fx_pair = f"{base_currency}{native_currency}=X"
            if fx_pair not in fx_rates.columns:
                if show_currency_detection:
                    print(f"[warning] {ticker}: FX pair {fx_pair} not found — skipping")
                converted_prices[ticker] = my_price
                continue
            fx_rate = fx_rates[fx_pair]
            my_price = my_price / fx_rate
            if show_currency_detection:
                print(f"[conversion] {ticker}: {native_currency} → {base_currency} via {fx_pair}")

        converted_prices[ticker] = my_price

    return converted_prices


#----------------------------------------------------------
# Create Portfolio from Share Quantities
# ----------------------------------------------------------
def create_portfolio(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    """
    Main
    ----
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


# ----------------------------------------------------------
# Compute Returns
# ----------------------------------------------------------
def compute_returns(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Main
    ----
    Compute daily percentage returns from a price or monetary position matrix.

    This function supports both raw asset prices and fixed-share monetary positions.
    Since returns are invariant to constant multipliers, the output is valid in both cases
    if there is no trading over time.

    Parameters
    ----------
    matrix : pd.DataFrame
        Price matrix or monetary position matrix with tickers as columns and dates as index.

    Returns
    -------
    returns : pd.DataFrame
        Daily percentage returns per asset.

    Raises
    ------
    Warning
        If the matrix structure is unsuitable for return analysis.

    Notes
    -----
    - Constant share positions yield identical returns to raw prices.
    - Varying positions (e.g., due to trading) will produce different return series.
    """
    returns = matrix.pct_change().dropna()

    validate_matrix(returns, context="returns")

    return returns
