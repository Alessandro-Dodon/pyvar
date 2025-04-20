import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, t as student_t
import matplotlib.pyplot as plt
import statsmodels.api as sm
from zipfile import ZipFile
from io import BytesIO
import requests

# ───────────────────────────────────────────────────────────────────
# Utility Functions
# ───────────────────────────────────────────────────────────────────
def get_raw_prices(tickers, start="2024-01-01"):
    """
    Scarica prezzi "Close" raw per i ticker.
    Restituisce DataFrame con prezzi ffill.
    """
    return (
        yf.download(" ".join(tickers), start=start,
                    auto_adjust=False, progress=False)["Close"]
        .ffill()
    )


def convert_to_base(raw, cur_map, base="EUR"):
    """
    Converte raw prices in valuta base usando FX daily.
    Gestisce pence (GBp, GBX) e centesimi rand (ZAc).
    """
    needed = {cur for cur in cur_map.values() if cur not in {base, "UNKNOWN"}}
    fx_pairs = [f"{cur}{base}=X" for cur in needed]
    fx = (
        yf.download(" ".join(fx_pairs), start=raw.index[0],
                    auto_adjust=True, progress=False)["Close"]
        .ffill() if fx_pairs else pd.DataFrame()
    )
    prices = pd.DataFrame(index=raw.index)
    for t in raw.columns:
        price = raw[t] * (0.01 if cur_map[t] in {"GBp","GBX","ZAc"} else 1)
        if cur_map[t] not in {base, "UNKNOWN"}:
            price *= fx[f"{cur_map[t]}{base}=X"]
        prices[t] = price
    return prices


def compute_returns_stats(prices):
    """
    Calcola returns, media e matrice di covarianza.
    """
    rets = prices.pct_change().dropna()
    return rets, rets.mean(), rets.cov()




# ───────────────────────────────────────────────────────────────────
# VaR/CVaR Methods
# ───────────────────────────────────────────────────────────────────

def historical_var(port_ret, alpha=5):
    """
    Historical VaR e CVaR (worst alpha%).
    """
    varp = np.percentile(port_ret, alpha)
    cvar = port_ret[port_ret <= varp].mean()
    return -varp, -cvar


def parametric_var_cvar(mu_p, sigma_p, dist="normal", alpha=5, dof=6):
    """
    Parametric VaR e CVaR Normal o Student-t.
    """
    a = alpha/100
    if dist == "normal":
        z = norm.ppf(1 - a)
        var = z * sigma_p - mu_p
        cvar = sigma_p * norm.pdf(norm.ppf(a)) / a - mu_p
    else:
        q = student_t.ppf(a, dof)
        f = student_t.pdf(q, dof)
        var = np.sqrt((dof-2)/dof) * student_t.ppf(1 - a, dof) * sigma_p - mu_p
        cvar = -mu_p + sigma_p * (f*(dof + q**2)) / (a*(dof-1))
    return var, cvar


def mc_var_cvar(mu, cov, port_val, weights, sims=5000, alpha=5):
    """
    Monte Carlo VaR e CVaR.
    """
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((len(mu), sims))
    sim_rets = mu.values.reshape(-1,1) + L @ Z
    port_sims = port_val * (1 + weights.values @ sim_rets)
    var = port_val - np.percentile(port_sims, alpha)
    cvar = port_val - port_sims[port_sims <= np.percentile(port_sims, alpha)].mean()
    return var, cvar, port_sims


def sharpe_model_cov(returns, market):
    """
    Costruisce Sigma via Sharpe 1-factor model:
      Σ = ββ' σ²_m + D_idio.
    Ritorna Sigma, betas, idio_vars, sigma_m2.
    """
    sigma_m2 = market.var(ddof=0)
    cov_im = returns.apply(lambda x: x.cov(market))
    betas = cov_im / sigma_m2
    idio_vars = returns.var(ddof=0) - betas.pow(2)*sigma_m2
    tickers = returns.columns
    outer = np.outer(betas, betas)*sigma_m2
    Sigma = pd.DataFrame(outer, index=tickers, columns=tickers)
    for t in tickers:
        Sigma.at[t,t] += idio_vars[t]
    return Sigma, betas, idio_vars, sigma_m2


def compute_ff3factor_var(tickers, shares, base="EUR", start="2024-01-01", alpha=0.95):
    """
    VaR e CVaR via Fama-French 3-factor.
    """
    raw = get_raw_prices(tickers, start=start)
    cur_map = {t: yf.Ticker(t).fast_info.get("currency", base) or base for t in tickers}
    for t in tickers:
        if cur_map[t] in {"GBp","GBX"}: cur_map[t] = "GBP"
        if cur_map[t] == "ZAc": cur_map[t] = "ZAR"
    prices = convert_to_base(raw, cur_map, base)
    rets = prices.pct_change().dropna()
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
           "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip")
    resp = requests.get(url); resp.raise_for_status()
    with ZipFile(BytesIO(resp.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".CSV"))
        ffraw = pd.read_csv(z.open(name), skiprows=3, index_col=0)
    mask = ffraw.index.astype(str).str.match(r"^\d{8}$")
    ff = ffraw.loc[mask].copy()
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    ff.columns = ["Mkt_RF","SMB","HML","RF"]
    ff = ff.astype(float)/100
    ff = ff.sort_index().loc[start:]
    ff = ff.reindex(rets.index).ffill()
    excess = rets.sub(ff["RF"], axis=0)
    X = sm.add_constant(ff[["Mkt_RF","SMB","HML"]])
    betas, idio = {}, {}
    for t in tickers:
        tmp = pd.concat([excess[t], X], axis=1).dropna()
        res = sm.OLS(tmp.iloc[:,0], tmp.iloc[:,1:]).fit()
        betas[t] = res.params.drop("const")
        idio[t]  = res.resid.var(ddof=0)
    B = pd.DataFrame(betas).T.values
    Sigma_f = ff[["Mkt_RF","SMB","HML"]].cov().values
    Sigma = B @ Sigma_f @ B.T + np.diag(pd.Series(idio).values)
    last = prices.iloc[-1]
    port_val = (last * pd.Series(shares)).sum()
    w = (last * pd.Series(shares))/port_val
    var_p = w.values @ Sigma @ w.values
    sigma_p = np.sqrt(var_p)
    z = norm.ppf(alpha)
    VaR = z * sigma_p * port_val
    tail = 1 - alpha
    CVaR = sigma_p * norm.pdf(norm.ppf(tail)) / tail * port_val
    return VaR, CVaR




# ───────────────────────────────────────────────────────────────────
# Main workflow
# ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE   = input("Base currency (EUR): ").upper()
    TKS    = input("Tickers (sep space): ").upper().split()
    SHARES = pd.Series({t: float(input(f"Shares of {t}: ")) for t in TKS})
    START  = "2024-01-01"

    raw   = get_raw_prices(TKS, START)
    cur_map = {t: (yf.Ticker(t).fast_info or {}).get("currency", BASE) for t in TKS}
    prices = convert_to_base(raw, cur_map, BASE)
    rets, mu, cov = compute_returns_stats(prices)

    # Stampa matrice var-cov
    print("\nCovariance matrix of asset returns:")
    print(cov.round(6))

    last  = prices.iloc[-1]
    port_val = (last * SHARES).sum()
    w     = (last * SHARES)/port_val
    mu_p  = w.dot(mu)
    sigma_p = np.sqrt(w.values @ cov.values @ w.values)

    hVar,  hCVar  = historical_var(rets.dot(w), alpha=5)
    nVar,  nCVar  = parametric_var_cvar(mu_p, sigma_p, dist="normal", alpha=5)
    tVar,  tCVar  = parametric_var_cvar(mu_p, sigma_p, dist="t-distribution", alpha=5, dof=6)
    mcVar, mcCVar, sims = mc_var_cvar(mu, cov, port_val, w, sims=5000, alpha=5)
    spy_price = convert_to_base(get_raw_prices(["SPY"], START), {"SPY":"USD"}, BASE)["SPY"]
    market = spy_price.pct_change().dropna()
    Sigma_sh, betas_sh, idio_sh, sigma_m2 = sharpe_model_cov(rets, market)
    sigma_sh = np.sqrt(w.values @ Sigma_sh.values @ w.values)
    sVar = norm.ppf(0.95)*sigma_sh*port_val
    tail = 0.05
    sCVar = sigma_sh * norm.pdf(norm.ppf(tail)) / tail * port_val
    ffVar, ffCVar = compute_ff3factor_var(TKS, SHARES, BASE, START, alpha=0.95)

    print(f"\nPortfolio Value: {port_val:,.2f} {BASE}\n")
    print("VaR Methods:")
    print(f"  Historical VaR 95%      : {hVar*port_val:,.2f}")
    print(f"  Parametric Normal VaR  : {nVar*port_val:,.2f}")
    print(f"  Parametric t VaR       : {tVar*port_val:,.2f}")
    print(f"  MC VaR 95%             : {mcVar:,.2f}")
    print(f"  Sharpe-model VaR 95%   : {sVar:,.2f}")
    print(f"  FF-3factor VaR 95%     : {ffVar:,.2f}\n")
    print("CVaR Methods:")
    print(f"  Historical CVaR 95%     : {hCVar*port_val:,.2f}")
    print(f"  Parametric Normal CVaR : {nCVar*port_val:,.2f}")
    print(f"  Parametric t CVaR      : {tCVar*port_val:,.2f}")
    print(f"  MC CVaR 95%            : {mcCVar:,.2f}")
    print(f"  Sharpe-model CVaR 95%  : {sCVar:,.2f}")
    print(f"  FF-3factor CVaR 95%    : {ffCVar:,.2f}\n")

    pos_val = prices.mul(SHARES, axis=1)
    port_ts = pos_val.sum(axis=1)
    wt_ts   = pos_val.div(port_ts, axis=0)*100
    debug_df = pd.concat([
        pos_val.tail().add_suffix(' VAL'),
        wt_ts.tail().add_suffix(' W%')
    ], axis=1).round(2)
    print("Last 5 positions VAL & W%:")
    print(debug_df)

    print("\nLast 5 portfolio values:")
    print(port_ts.tail())

    plt.hist(sims, bins=50, edgecolor='k')
    plt.axvline(np.percentile(sims,5), color='r', linestyle='--', label='VaR95')
    plt.title('MC Simulated Portfolio Values')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()