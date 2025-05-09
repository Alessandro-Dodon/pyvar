import data_download_tragattino as dd
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

tickers = ["MSFT", "NVDA"]
#currency_map = {"MSFT": "USD", "NVDA": "USD"}

# 1. Scarica i prezzi grezzi
print(">>> Raw prices (unadjusted):")
raw = dd.get_raw_prices(tickers)
print(raw.head(), "\n")

# 2. Converte tutto in EUR
print(">>> Converted to base currency (EUR):")
converted = dd.convert_to_base(raw, base='EUR', session=session)
print(converted.head(), "\n")

# 3. Calcola rendimenti
print(">>> Returns, mean, covariance:")
returns, mean_ret, cov = dd.compute_returns_stats(converted)
print("Returns:\n", returns.head(), "\n")
print("Mean returns:\n", mean_ret, "\n")
print("Covariance matrix:\n", cov, "\n")