# ================================================================
# IMPORTS
# ================================================================
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay
from functions.data_download import (
    get_raw_prices,
    convert_to_base,
    create_portfolio,
    summary_statistics
)
from functions.simulations import black_scholes

# ================================================================
# 1) USER INPUT – Base currency, tickers, share quantities
# ================================================================
BASE = input("Base currency [EUR] : ").strip().upper() or "EUR"
TICKERS = input("Equity tickers      : ").upper().split()
SHARES = pd.Series({t: float(input(f"  Shares for {t:<6}: ")) for t in TICKERS})

# ================================================================
# 1B) OPTIONS INPUT
# ================================================================
options_list = []
if input("Do you have options in the portfolio? (y/n): ").lower().startswith("y"):
    print("\n→ Enter one option per line. Press ENTER to stop.\n")
    while True:
        underlying = input("  Underlying ticker (ENTER to stop): ").upper()
        if not underlying:
            break

        opt_type = input("    Option type [call/put]          : ").strip().lower()
        contracts = float(input("    Contracts (can be negative)     : "))
        multiplier = int(input("    Multiplier [default = 100]      : ") or 100)
        strike = float(input("    Strike price                    : "))
        maturity = float(input("    Time to maturity (in years)     : "))

        options_list.append({
            "under": underlying,
            "type": opt_type,
            "qty": contracts * multiplier,
            "K": strike,
            "T": maturity
        })

        print(f"  ✔ Added {opt_type.upper()} {underlying} x{contracts}x{multiplier} @K={strike}\n")

# ================================================================
# 2) EQUITY DATA PREPROCESSING
# ================================================================
START_DATE = (pd.Timestamp.today() - BDay(100)).strftime("%Y-%m-%d")
raw_prices = get_raw_prices(TICKERS, start=START_DATE)

currency_map = {}
for ticker in TICKERS:
    try:
        currency_map[ticker] = yf.Ticker(ticker).fast_info.get("currency", BASE) or BASE
    except Exception:
        currency_map[ticker] = "UNKNOWN"

converted_prices = convert_to_base(
    raw_prices,
    currency_mapping=currency_map,
    base_currency=BASE
)

positions_df = create_portfolio(converted_prices, SHARES)
returns, mean_returns, cov_matrix = summary_statistics(converted_prices)

last_prices = converted_prices.iloc[-1]
portfolio_value = float((last_prices * SHARES).sum())
weights = (last_prices * SHARES) / portfolio_value

print("\n===== EQUITY PORTFOLIO =====")
print(f"Total value: {portfolio_value:,.2f} {BASE}")
print("Weights:")
print(weights.to_string(float_format=lambda x: f"{x:.4f}"))

# ================================================================
# 3) FULL DATASET (EQUITY + OPTIONS)
# ================================================================
option_value = 0.0

if options_list:
    print("\n===== PREPARING DATA FOR OPTIONS... =====")
    all_tickers = sorted(set(TICKERS) | {opt["under"] for opt in options_list})
    raw_all = get_raw_prices(all_tickers, start=START_DATE)

    all_currencies = {}
    for t in all_tickers:
        try:
            all_currencies[t] = yf.Ticker(t).fast_info.get("currency", BASE) or BASE
        except Exception:
            all_currencies[t] = "UNKNOWN"

    prices_all = convert_to_base(raw_all, currency_mapping=all_currencies, base_currency=BASE)
    returns_all, mu_all, cov_all = summary_statistics(prices_all)
    hist_vol = returns_all.std() * np.sqrt(252)

    try:
        rf_rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except Exception:
        print("[warning] ^IRX fetch failed. Using fallback rate = 2%")
        rf_rate = 0.02

    shares_vector = pd.Series([SHARES.get(t, 0.0) for t in all_tickers]).values
    spot_prices = prices_all.iloc[-1].values

    for opt in options_list:
        opt.update({
            "asset_index": all_tickers.index(opt["under"]),
            "sigma": float(hist_vol.get(opt["under"], hist_vol.mean())),
            "r": rf_rate
        })

    print("\n===== ENRICHED OPTION DATA =====")
    for opt in options_list:
        print(
            f"{opt['type'].upper():4} {opt['under']:<6} @ K={opt['K']:.2f}  "
            f"T={opt['T']}y  Qty={opt['qty']:.0f}  σ={opt['sigma']:.4f}"
        )

    print("\n===== BLACK-SCHOLES PRICING =====")
    for opt in options_list:
        S = last_prices.get(opt["under"])
        if S is None:
            S = yf.Ticker(opt["under"]).history(period="1d")["Close"].iloc[-1]

        price_opt = black_scholes(
            S=S,
            K=opt["K"],
            tau=opt["T"],
            r=opt["r"],
            sigma=opt["sigma"],
            opt_type=opt["type"]
        )

        val_opt = price_opt * opt["qty"]
        option_value += val_opt

        print(f"{opt['type'].upper():4} {opt['under']:<6}  {opt['qty']:>6.0f}× @ {price_opt:>8.2f} = {val_opt:>10.2f}")

# ================================================================
# 4) TOTAL PORTFOLIO VALUE
# ================================================================
total_value = portfolio_value + option_value

print("\n===== TOTAL PORTFOLIO VALUE =====")
print(f"Equity-only:         {portfolio_value:,.2f} {BASE}")
print(f"Options (B-S):       {option_value:,.2f} {BASE}")
print(f"TOTAL PORTFOLIO:     {total_value:,.2f} {BASE}")
