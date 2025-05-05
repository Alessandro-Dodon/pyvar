import yfinance as yf
import pandas as pd

# Download raw closing prices for given tickers
def get_raw_prices(tickers, start="2024-01-01") -> pd.DataFrame:
    """
    Downloads raw (unadjusted) closing prices for specified tickers.
    """
    prices = (
        yf.download(" ".join(tickers), start=start,
                    auto_adjust=False, progress=False)["Close"]
        .ffill()
    )
    return prices

# Convert raw prices from various currencies to a single base currency
def convert_to_base(raw: pd.DataFrame, cur_map: dict, base: str = "EUR") -> pd.DataFrame:
    """
    Converts raw prices from multiple currencies into a single base currency.
    """
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
        if cur_map[t] in {"GBp", "GBX", "ZAc"}:
            p *= 0.01
        cur = cur_map[t]
        if cur not in {base, "UNKNOWN"}:
            pair = f"{base}{cur}=X"
            rate = fx[pair]
            p = p / rate
        out[t] = p
    return out

# Compute returns, mean returns, and covariance matrix
def compute_returns_stats(prices: pd.DataFrame):
    """
    Computes daily returns, mean returns, and covariance matrix.
    """
    returns = prices.pct_change().dropna()
    return returns, returns.mean(), returns.cov()
