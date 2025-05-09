from curl_cffi import requests
import yfinance as yf
import pandas as pd

# Crea sessione anti-429
session = requests.Session(impersonate="chrome")

# Scarica prezzi raw (non aggiustati)
def get_raw_prices(tickers, start="2024-01-01") -> pd.DataFrame:
    """
    Downloads raw (unadjusted) closing prices for specified tickers.
    """
    data = yf.download(" ".join(tickers), start=start,
                       auto_adjust=False, progress=False, session=session)
    prices = data["Close"].ffill()
    return prices

# Conversione a valuta base
def convert_to_base(raw: pd.DataFrame, cur_map: dict, base: str = "EUR") -> pd.DataFrame:
    """
    Converts raw prices from multiple currencies into a single base currency.
    """
    needed = {cur_map[t] for t in raw.columns if cur_map[t] not in {base, "UNKNOWN"}}
    fx_pairs = [f"{base}{cur}=X" for cur in needed]
    
    if fx_pairs:
        fx = yf.download(" ".join(fx_pairs), start=raw.index[0],
                         auto_adjust=True, progress=False, session=session)["Close"]
        fx = fx.reindex(raw.index).ffill()
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

# Statistiche dei rendimenti
def compute_returns_stats(prices: pd.DataFrame):
    """
    Computes daily returns, mean returns, and covariance matrix.
    """
    returns = prices.pct_change().dropna()
    return returns, returns.mean(), returns.cov()



tickers = ["MSFT", "NVDA"]
currency_map = {"MSFT": "USD", "NVDA": "USD"}

raw_prices = get_raw_prices(tickers)
eur_prices = convert_to_base(raw_prices, currency_map, base="EUR")
returns, mean_ret, cov_mat = compute_returns_stats(eur_prices)

print(returns.head())
print(mean_ret)
print(cov_mat)
