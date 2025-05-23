
# ================================================================
# VaR and ES Risk Report for Equity + Options Portfolio
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
from functions.simulations import (
    black_scholes,
    monte_carlo_var,
    historical_simulation_var,
    simulation_es
)
from functions.portfolio import asset_normal_var
from functions.factor_models import single_factor_var, factor_models_es
from functions.correlation import ma_correlation_var, ewma_correlation_var, correlation_es
from functions.evt import evt_var, evt_es
from functions.volatility import garch_var, volatility_es

# ========== 1) USER INPUT ==========
BASE = input("Base currency [EUR] : ").strip().upper() or "EUR"
TICKERS = input("Equity tickers      : ").upper().split()
SHARES = pd.Series({t: float(input(f"  Shares for {t:<6}: ")) for t in TICKERS})

# ========== 2) OPTIONS INPUT ==========
options_list = []
if input("Do you have options in the portfolio? (y/n): ").lower().startswith("y"):
    print("\\n→ Enter one option per line. Press ENTER to stop.\\n")
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

# ========== 3) EQUITY DATA PREPROCESSING ==========
START_DATE = (pd.Timestamp.today() - BDay(100)).strftime("%Y-%m-%d")
raw_prices = get_raw_prices(TICKERS, start=START_DATE)

currency_map = {}
for t in TICKERS:
    try:
        currency_map[t] = yf.Ticker(t).fast_info.get("currency", BASE) or BASE
    except:
        currency_map[t] = BASE

converted_prices = convert_to_base(raw_prices, currency_mapping=currency_map, base_currency=BASE)
positions_df = create_portfolio(converted_prices, SHARES)
returns, _, _ = summary_statistics(converted_prices)

last_prices = converted_prices.iloc[-1]
portfolio_value = float((last_prices * SHARES).sum())
weights = (last_prices * SHARES) / portfolio_value

print(f"\\nTotal Portfolio Value (Equity Only): {portfolio_value:,.2f} {BASE}")
print("Weights:")
print(weights.to_string(float_format=lambda x: f"{x:.4f}"))

# ========== 4) OPTIONS DATA PREP ==========
option_value = 0.0
if options_list:
    all_tickers = sorted(set(TICKERS) | {opt["under"] for opt in options_list})
    raw_all = get_raw_prices(all_tickers, start=START_DATE)
    all_currencies = {t: BASE for t in all_tickers}
    prices_all = convert_to_base(raw_all, all_currencies, base_currency=BASE)
    returns_all, mu_all, cov_all = summary_statistics(prices_all)
    hist_vol = returns_all.std() * np.sqrt(252)

    try:
        rf_rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except:
        rf_rate = 0.02

    for opt in options_list:
        opt.update({
            "asset_index": all_tickers.index(opt["under"]),
            "sigma": float(hist_vol.get(opt["under"], hist_vol.mean())),
            "r": rf_rate
        })

        S = last_prices.get(opt["under"])
        if S is None:
            S = yf.Ticker(opt["under"]).history(period="1d")["Close"].iloc[-1]

        price_opt = black_scholes(S, opt["K"], opt["T"], opt["r"], opt["sigma"], opt["type"])
        val_opt = price_opt * opt["qty"]
        option_value += val_opt

total_value = portfolio_value + option_value
print(f"\\nTOTAL PORTFOLIO VALUE (Equity + Options): {total_value:,.2f} {BASE}")

# ========== 5) VAR + ES CALCULATIONS ==========

# -- Asset-Normal (equity only) --
asset_var_df = asset_normal_var(positions_df, confidence_level=0.99)
var_asset_normal = asset_var_df["Diversified_VaR"].iloc[-1]

# -- Monte Carlo (equity + options) --
var_mc, pnl_mc = monte_carlo_var(converted_prices, SHARES.values, options_list)
es_mc = simulation_es(var_mc, pnl_mc)

# -- Historical Simulation (equity + options) --
var_hist, pnl_hist = historical_simulation_var(converted_prices, SHARES.values, options_list)
es_hist = simulation_es(var_hist, pnl_hist)

# -- Single-Factor (equity only) --
benchmark = converted_prices[TICKERS[0]].pct_change().dropna()
sf_result, sf_vol = single_factor_var(returns, benchmark, weights, portfolio_value)
sf_result = factor_models_es(sf_result, sf_vol)
var_sf = sf_result["VaR_monetary"].iloc[-1]
es_sf = sf_result["ES_monetary"].iloc[-1]

# -- MA Correlation (equity only) --
ma_result = ma_correlation_var(positions_df, distribution="normal")
ma_result = correlation_es(ma_result)
var_ma = ma_result["VaR Monetary"].iloc[-1]
es_ma = ma_result["ES Monetary"].iloc[-1]

# -- EWMA Correlation (equity only) --
ewma_result = ewma_correlation_var(positions_df, distribution="normal")
ewma_result = correlation_es(ewma_result)
var_ewma = ewma_result["VaR Monetary"].iloc[-1]
es_ewma = ewma_result["ES Monetary"].iloc[-1]

# -- EVT (equity only) --
ret_port = returns @ weights
evt_result = evt_var(ret_port, wealth=portfolio_value)
evt_result = evt_es(evt_result, wealth=portfolio_value)
var_evt = evt_result["VaR_monetary"].iloc[-1]
es_evt = evt_result["ES_monetary"].iloc[-1]

# -- GARCH (equity only) --
garch_result, _ = garch_var(ret_port, wealth=portfolio_value)
garch_result = volatility_es(garch_result, confidence_level=0.99, wealth=portfolio_value)
var_garch = garch_result["VaR_monetary"].iloc[-1]
es_garch = garch_result["ES_monetary"].iloc[-1]

# ========== 6) SUMMARY ==========
summary_df = pd.DataFrame({
    "Method": [
        "Asset-Normal (EQ)",
        "Monte Carlo (EQ+OPT)",
        "Historical Sim. (EQ+OPT)",
        "Single-Factor (EQ)",
        "MA (EQ)",
        "EWMA (EQ)",
        "EVT (EQ)",
        "GARCH (EQ)"
    ],
    "VaR_99%": [
        var_asset_normal,
        var_mc,
        var_hist,
        var_sf,
        var_ma,
        var_ewma,
        var_evt,
        var_garch
    ],
    "ES_99%": [
        None,
        es_mc,
        es_hist,
        es_sf,
        es_ma,
        es_ewma,
        es_evt,
        es_garch
    ]
})

print("\\n===== SUMMARY: VaR and ES Comparison =====")
print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

