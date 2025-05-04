
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm





# ────────── Black-Scholes + MC Equity+Options ──────────
def bs_price(S, K, tau, r, sigma, opt_type="call"):
    if tau <= 0:
        val = max(0.0, S-K) if opt_type=="call" else max(0.0, K-S)
        return float(val)
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*tau)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    if opt_type=="call":
        val = S*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)
    else:
        val = K*np.exp(-r*tau)*norm.cdf(-d2) - S*norm.cdf(-d1)
    return float(val)

def mc_var_portfolio(S0, mu, cov, shares_eq, options,
                     horizon=1/252, alpha=0.05,
                     Nsim=50_000, seed=42):
    np.random.seed(seed)
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(Nsim, len(S0))
    rets_sim = mu.values + Z.dot(L.T)
    S_sim    = S0 * (1 + rets_sim)

    init_prices = [
        bs_price(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    pnl = np.empty(Nsim)
    for i in range(Nsim):
        pl_eq  = shares_eq.dot(S_sim[i]-S0)
        pl_opt = 0.0
        for j,op in enumerate(options):
            pj = bs_price(S_sim[i,op['idx']],
                          op['K'],
                          op['T']-horizon,
                          op['r'],
                          op['sigma'],
                          op['type'])
            pl_opt += op['qty']*(pj - init_prices[j])
        pnl[i] = pl_eq + pl_opt

    var  = -np.percentile(pnl, alpha*100)
    cvar = -pnl[pnl<=-var].mean()
    return var, cvar, pnl
