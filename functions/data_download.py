import yfinance as yf
import pandas as pd


def get_raw_prices(tickers, start="2024-01-01") -> pd.DataFrame:
    '''
    Downloads raw (unadjusted) closing prices for a list of tickers.

    Parameters:
    - tickers (list of str): List of ticker symbols (e.g. ['MSFT', 'AAPL', 'BMW.DE'])
    - start (str): Start date in 'YYYY-MM-DD' format. Defaults to '2024-01-01'.

    Returns:
    - pd.DataFrame: A DataFrame indexed by date, with tickers as columns and unadjusted closing prices as values.
    '''
    prices = (
        yf.download(" ".join(tickers), start=start,
                    auto_adjust=False, progress=False)["Close"]
        .ffill()
    )
    return prices


def convert_to_base(raw: pd.DataFrame, cur_map: dict = None, base: str = "EUR") -> pd.DataFrame:
    '''
    Converts prices from their native currencies into a single base currency.

    Parameters:
    - raw (pd.DataFrame): DataFrame of raw prices with tickers as columns and dates as index.
    - cur_map (dict, optional): Mapping of tickers to their currencies (e.g. {'AAPL': 'USD'}).
                                If None, currencies are auto-detected using yfinance's fast_info.
    - base (str): Target base currency (e.g. 'EUR', 'USD'). Defaults to 'EUR'.

    Returns:
    - pd.DataFrame: A DataFrame of prices converted to the base currency. Same shape and index as input.
    
    Notes:
    - FX rates are fetched from Yahoo Finance using synthetic tickers like 'EURUSD=X'.
    - Minor currency units like GBp or ZAc are automatically scaled by 0.01.
    - If a ticker's currency is unknown or the FX rate is unavailable, that column remains unconverted.
    '''
    if cur_map is None:
        cur_map = {}
        for t in raw.columns:
            try:
                cur = yf.Ticker(t).fast_info.get("currency", "UNKNOWN") or "UNKNOWN"
            except Exception:
                cur = "UNKNOWN"
            cur_map[t] = cur
            print(f"[currency detection] {t}: {cur}")

    needed = {cur_map[t] for t in raw.columns if cur_map[t] not in {base, "UNKNOWN"}}
    fx_pairs = [f"{base}{cur}=X" for cur in needed]

    if fx_pairs:
        fx = (
            yf.download(" ".join(fx_pairs), start=raw.index[0],
                        auto_adjust=True, progress=False)["Close"]
            .reindex(raw.index)
            .ffill()
        )
    else:
        fx = pd.DataFrame(index=raw.index)

    out = pd.DataFrame(index=raw.index)
    for t in raw.columns:
        p = raw[t].copy()
        cur = cur_map[t]
        if cur in {"GBp", "GBX", "ZAc"}:
            p *= 0.01
        if cur not in {base, "UNKNOWN"}:
            pair = f"{base}{cur}=X"
            rate = fx[pair]
            p = p / rate
        out[t] = p

    return out


def compute_returns_stats(prices: pd.DataFrame):
    '''
    Computes daily returns, mean returns, and the covariance matrix for a price DataFrame.

    Parameters:
    - prices (pd.DataFrame): Price data with tickers as columns and dates as index, all in the same currency.

    Returns:
    - returns (pd.DataFrame): Daily percentage returns.
    - mean_returns (pd.Series): Mean return for each asset.
    - covariance_matrix (pd.DataFrame): Covariance matrix of the returns.
    '''
    returns = prices.pct_change().dropna()
    return returns, returns.mean(), returns.cov()



raw = get_raw_prices(["MSFT", "NVDA", "ISP.MI", "NESN.SW"])
converted = convert_to_base(raw, base="EUR")
returns, mu, cov = compute_returns_stats(converted)
print(">>> Raw prices:")
print(raw.head())
print(">>> Converted to EUR:")
print(converted.head())
print(">>> Returns:")
print(returns.head())