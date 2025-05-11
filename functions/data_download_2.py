#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import yfinance as yf
import pandas as pd

####################################################
# Note: add a vector of shares owned for each Asset?
#       currency conversion works on the x df or not?
####################################################

#----------------------------------------------------------
# Downloading Price Data
# ----------------------------------------------------------
def get_raw_prices(tickers, start="2024-01-01") -> pd.DataFrame:
    '''
    Downloads raw (adjusted) closing prices for a list of tickers.

    Parameters:
    - tickers (list of str): List of ticker symbols (e.g. ['MSFT', 'AAPL', 'BMW.DE']).
    - start (str): Start date in 'YYYY-MM-DD' format.

    Returns:
    - pd.DataFrame: Adjusted closing prices (tickers as columns, dates as index).

    Notes:
    - Issues a warning if any missing values are detected in the returned data.
    - Users are advised to inspect or handle missing data before downstream analysis.
    '''
    prices = (
        yf.download(" ".join(tickers), start=start,
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
    '''
    Converts prices from their native currencies into a single base currency.

    Parameters:
    - raw (pd.DataFrame): DataFrame of raw prices with tickers as columns and dates as index.
    - cur_map (dict, optional): Mapping of tickers to their currencies (e.g. {'AAPL': 'USD'}).
                                If None, currencies are auto-detected using yfinance's fast_info.
    - base (str): Target base currency (e.g. 'EUR', 'USD'). Defaults to 'EUR'.
    - show_currency_detection (bool): If True, prints detected currencies and conversion steps.

    Returns:
    - pd.DataFrame: A DataFrame of prices converted to the base currency. Same shape and index as input.
    
    Notes:
    - FX rates are fetched from Yahoo Finance using synthetic tickers like 'EURUSD=X'.
    - Minor currency units like GBp or ZAc are automatically scaled by 0.01,
      and their codes are replaced with GBP or ZAR to ensure correct FX conversion.
    - Prices are converted using the daily closing FX rate for each corresponding date.
    '''
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

    return out


#----------------------------------------------------------
# Create Portfolio from Share Quantities
# ----------------------------------------------------------
def create_portfolio(prices: pd.DataFrame, shares: pd.Series) -> pd.DataFrame:
    '''
    Multiplies prices by the number of shares held to compute monetary positions.

    Converts a price time series into a portfolio of monetary exposures by multiplying 
    each asset's price by its corresponding share count. The result reflects the value 
    of the full portfolio over time.

    Parameters:
    - prices (pd.DataFrame): Price data (tickers = columns, dates = index).
    - shares (pd.Series): Number of shares per ticker (index = tickers).

    Returns:
    - pd.DataFrame: Monetary positions (same shape as `prices`).

    Raises:
    - ValueError: If tickers in `shares` do not match columns in `prices`.
    - ValueError: If total portfolio value is zero or negative on any day.

    Notes:
    - Requires that the number of tickers in `shares` exactly matches `prices.columns`.
    - Supports any order of tickers in `shares`, as long as names match.
    - Short positions (negative shares) and fractional shares are allowed.
    - Currency conversion should be applied *before* this function.
    '''
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
# Returns and Summary Statistics
# ----------------------------------------------------------
def compute_returns_stats(prices: pd.DataFrame):
    '''
    Computes daily returns, mean returns, and the covariance matrix for a price DataFrame.

    Parameters:
    - prices (pd.DataFrame): Price data with tickers as columns and dates as index, all in the same currency.

    Returns:
    - returns (pd.DataFrame): Daily percentage returns.
    - mean_returns (pd.Series): Mean return for each asset.
    - covariance_matrix (pd.DataFrame): Covariance matrix of the returns.

    Notes:
    - If input is raw prices from `get_raw_prices()`, the output reflects per-share returns.
    - If input is monetary positions from `create_portfolio()`, the output reflects portfolio-scale monetary returns.
    - Apply after currency conversion, if necessary.
    '''
    returns = prices.pct_change().dropna()
    return returns, returns.mean(), returns.cov()



