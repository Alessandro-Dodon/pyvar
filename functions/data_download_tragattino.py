from curl_cffi import requests
import yfinance as yf
import pandas as pd

# Global session with browser impersonation to avoid 429 errors
session = requests.Session(impersonate="chrome")

def get_raw_prices(tickers, start="2024-01-01") -> pd.DataFrame:
    """
    Downloads raw (unadjusted) closing prices for specified tickers.
    
    Parameters:
    - tickers (list): List of ticker symbols
    - start (str): Start date (YYYY-MM-DD)
    
    Returns:
    - pd.DataFrame: DataFrame with closing prices (tickers as columns)
    """
    data = yf.download(" ".join(tickers), start=start,
                       auto_adjust=False, progress=False, session=session)
    prices = data["Close"].ffill()
    return prices



def convert_to_base(
    raw: pd.DataFrame,
    cur_map: dict = None,
    base: str = "EUR",
    session=None
) -> pd.DataFrame:
    """
    Converts raw prices from multiple currencies to a single base currency.
    Prints diagnostic info during conversion.

    Parameters:
    - raw (pd.DataFrame): Unadjusted prices with tickers as columns
    - cur_map (dict, optional): {ticker: currency}. If None, auto-detected.
    - base (str): Target base currency (default: 'EUR')
    - session (requests.Session, optional): Optional session for custom headers / impersonation

    Returns:
    - pd.DataFrame: Prices converted to base currency
    """

    # Detect currencies if not provided
    if cur_map is None:
        cur_map = {}
        for t in raw.columns:
            try:
                info = yf.Ticker(t, session=session).fast_info
                cur = info.get("currency", "UNKNOWN") or "UNKNOWN"
            except Exception:
                cur = "UNKNOWN"
            cur_map[t] = cur
            print(f"[currency detection] {t}: {cur}")

    # Determine which FX pairs we need
    needed = {cur_map[t] for t in raw.columns if cur_map[t] not in {base, "UNKNOWN"}}
    fx_pairs = [f"{base}{cur}=X" for cur in needed]

    # Download FX rates if needed
    if fx_pairs:
        print(f"[fx download] Downloading FX pairs: {', '.join(fx_pairs)}")
        fx = (
            yf.download(" ".join(fx_pairs), start=raw.index[0],
                        auto_adjust=True, progress=False, session=session)["Close"]
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
            print(f"[unit conversion] {t}: converting from minor units ({cur}) to major")
            p *= 0.01

        if cur == base:
            print(f"[conversion] {t}: already in {base}, no conversion needed")
        elif cur == "UNKNOWN":
            print(f"[warning] {t}: currency UNKNOWN — skipping conversion")
        else:
            pair = f"{base}{cur}=X"
            if pair not in fx.columns:
                print(f"[error] {t}: FX rate {pair} not found — skipping")
                continue
            rate = fx[pair]
            p = p / rate
            print(f"[conversion] {t}: converted from {cur} using {pair}")

        out[t] = p

    return out



def compute_returns_stats(prices: pd.DataFrame):
    """
    Computes daily returns, mean returns, and covariance matrix.

    Parameters:
    - prices (pd.DataFrame): Prices in base currency

    Returns:
    - returns (pd.DataFrame)
    - mean (pd.Series)
    - covariance (pd.DataFrame)
    """
    returns = prices.pct_change().dropna()
    return returns, returns.mean(), returns.cov()
