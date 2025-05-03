import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model

# ——— 1. Download raw prices ———

def get_raw_prices(tickers, start="2024-01-01", dbg=False):
    raw = (
        yf.download(" ".join(tickers), start=start,
                    auto_adjust=False, progress=False)["Close"]
        .ffill()
    )
    if dbg:
        print("\n[DEBUG] raw prices tail:\n", raw.tail())
    return raw

# ——— 2. Convert prices to base currency ———

def convert_to_base(raw, cur_map, base="EUR", dbg=False):
    fx_needed = [f"{cur}{base}=X" for cur in set(cur_map.values()) - {base}]
    fx = (
        yf.download(" ".join(fx_needed), start=raw.index[0],
                    auto_adjust=True, progress=False)["Close"]
        .ffill()
        if fx_needed else pd.DataFrame()
    )

    out = pd.DataFrame(index=raw.index)
    for t in raw.columns:
        price = raw[t] * (0.01 if cur_map[t] in {"GBp", "GBX", "ZAc"} else 1)
        if cur_map[t] != base:
            pair = f"{cur_map[t]}{base}=X"
            price = price * fx[pair]
        out[t] = price

    if dbg:
        print("\n[DEBUG] prices in base tail:\n", out.tail())
    return out

# ——— MAIN ———

if __name__ == "__main__":
    BASE    = input("Base currency (e.g. EUR): ").upper()
    TKS     = input("Tickers separated by space: ").upper().split()
    SHARES  = pd.Series({t: float(input(f"Shares of {t}: ")) for t in TKS})
    START   = "2024-01-01"

    raw = get_raw_prices(TKS, start=START, dbg=True)
    cur_map = {t: (yf.Ticker(t).fast_info or {}).get("currency", BASE) for t in TKS}
    for t in TKS:
        if cur_map[t] in {"GBp", "GBX"}: cur_map[t] = "GBP"
        if cur_map[t] == "ZAc": cur_map[t] = "ZAR"

    prices = convert_to_base(raw, cur_map, base=BASE, dbg=True)

    returns = prices.pct_change().dropna()
    meanReturns = returns.mean()
    covMatrix = returns.cov()

    last_prices = prices.iloc[-1]
    port_val = (last_prices * SHARES).sum()
    weights = (last_prices * SHARES) / port_val

    # ——— Monte Carlo VaR with GARCH(1,1) ———
    mc_sims = 1000
    garch_vols = []

    for asset in returns.columns:
        model = arch_model(returns[asset] * 100, vol='Garch', p=1, q=1, dist='normal')
        res = model.fit(disp='off')
        forecast = res.forecast(horizon=1)
        sigma = np.sqrt(forecast.variance.values[-1, 0]) / 100  # back to original scale
        garch_vols.append(sigma)

    garch_vols = np.array(garch_vols)

    Z = np.random.normal(size=(len(weights), mc_sims))
    L = np.linalg.cholesky(covMatrix.corr())
    correlated_Z = L @ Z
    daily_returns = meanReturns.values.reshape(-1, 1) + (garch_vols.reshape(-1, 1) * correlated_Z)
    simulated_returns = weights @ daily_returns
    simulated_end_values = port_val * (1 + simulated_returns)

    VaR_mc = port_val - np.percentile(simulated_end_values, 5)
    CVaR_mc = port_val - simulated_end_values[simulated_end_values <= np.percentile(simulated_end_values, 5)].mean()

    print("\n══════════ RISK METRICS (1-day horizon) ══════════")
    print(f"Portfolio Value           : {port_val:,.2f} {BASE}")
    print(f"Monte Carlo VaR 95% (GARCH): {VaR_mc:,.2f} {BASE}")
    print(f"Monte Carlo CVaR 95% (GARCH): {CVaR_mc:,.2f} {BASE}")

    plt.hist(simulated_end_values, bins=50, edgecolor='k')
    plt.title('Monte Carlo Simulated Portfolio Values (1-day, GARCH-adjusted)')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.axvline(np.percentile(simulated_end_values, 5), color='r', linestyle='--', label='VaR 95%')
    plt.legend()
    plt.show()
