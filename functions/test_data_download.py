import data_download_tragattino as dd
from curl_cffi import requests
session = requests.Session(impersonate="chrome")

tickers = ["MSFT", "NVDA", "ISP.MI", "NESN.SW"]

# 1. Download prices in native currency
print(">>> Raw prices (native currency):")
raw = dd.get_raw_prices(tickers)
print(raw.head())
print("-"*50)

# 2. Converte tutto in EUR
print(">>> Converted to base currency (EUR):")
converted = dd.convert_to_base(raw, base='EUR', session=session)
print(converted.head())
print("-"*50)

# 3. Calcola rendimenti
print(">>> Returns, mean, covariance:")
returns, mean_ret, cov = dd.compute_returns_stats(converted)
print("Returns:\n", returns.head(), "\n")
print("Mean returns:\n", mean_ret, "\n")
print("Covariance matrix:\n", cov, "\n")