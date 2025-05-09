# test_data_download.py
import data_download_tragattino as dd

tickers = ["MSFT", "NVDA", "ISP.MI", "NESN.SW"]

print(">>> Raw prices:")
raw = dd.get_raw_prices(tickers)
print(raw.head())

print(">>> Converted to EUR:")
converted = dd.convert_to_base(raw, base="EUR")
print(converted.head())

print(">>> Returns:")
returns, mu, cov = dd.compute_returns_stats(converted)
print(returns.head())
print(mu)
print(cov)
