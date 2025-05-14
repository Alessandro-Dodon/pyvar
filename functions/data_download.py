"""
Price Data and Portfolio Construction Module
-----------------------------------------------

Provides utility functions to download and process financial time series data.
These functions are intended for use with the portfolio, factor model,
simulation, and time-varying correlation modules.

For single-asset analysis (e.g., VaR/ES with univariate models), these tools
are not required, since the input can be a single price or return series.

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
"""

# TODO: check FX conversion, I see minor differences from use to use

# TODO: better names for currency conversion function for the output

#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd


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
        If missing values (NaNs) are detected in the downloaded price data.
    """
    prices = (
        yf.download(" ".join(tickers), start=start, end=end,
                    auto_adjust=True, progress=False)["Close"]
        .ffill()
    )

    if prices.isnull().any().any():
        bad_tickers = prices.columns[prices.isnull().any()].tolist()
        print(f"[warning] Missing values detected in: {bad_tickers}")
        print("[info] Consider handling NaNs before using this data for risk or return analysis.")

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
    out = pd.DataFrame(index=raw.index)
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
                out[t] = p
                continue
            rate = fx[pair]
            p = p / rate
            if show_currency_detection:
                print(f"[conversion] {t}: {cur} → {base} via {pair}")

        out[t] = p

    return out # WHY THIS NAME


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

    return positions


 #----------------------------------------------------------
# Summary Statistics
# ----------------------------------------------------------
def summary_statistics(prices: pd.DataFrame): 
    """
    Compute daily returns, mean returns, and the return covariance matrix.

    Input can be either per-share prices or monetary portfolio exposures.
    Output type (percentage or monetary returns) depends on the input.

    Parameters
    ----------
    prices : pd.DataFrame
        Price matrix with tickers as columns and dates as index.

    Returns
    -------
    returns : pd.DataFrame
        Daily returns (percentage or monetary).
    mean_returns : pd.Series
        Mean return per asset.
    covariance_matrix : pd.DataFrame
        Covariance matrix of the return series.
    """
    returns = prices.pct_change().dropna()
    mean_returns = returns.mean()
    covariance_matrix = returns.cov()

    return returns, mean_returns, covariance_matrix
