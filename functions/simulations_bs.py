import numpy as np                      # numerical operations
import pandas as pd                     # data handling (for DataFrame inputs)
from scipy.stats import norm            # cumulative distribution for normals

# ────────── Black-Scholes Price ──────────
def bs_price(S, K, tau, r, sigma, opt_type="call"):
    """
    Compute the Black-Scholes price for a European call or put.
    Returns a float in monetary units.
    """
    if tau <= 0:  # option expired
        # intrinsic value: max(S–K,0) for call, max(K–S,0) for put
        return float(max(0.0, S - K) if opt_type == "call" else max(0.0, K - S))
    # d1 and d2 components of BS formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))  # risk-adjusted return
    d2 = d1 - sigma * np.sqrt(tau)  # volatility term
    if opt_type == "call":
        # call: S·N(d1) – K·e^(–r·τ)·N(d2)
        return float(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2))
    else:
        # put: K·e^(–r·τ)·N(–d2) – S·N(–d1)
        return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ────────── Parametric Monte Carlo VaR/CVaR ──────────
def mc_simulation_var_es(S0, mu, cov, shares_eq, options,
                     horizon=1/252, alpha=0.05,
                     Nsim=50_000, seed=42):
    """
    Monte Carlo (parametric) VaR & CVaR for equity + options.
    - S0         : array of current spot prices, length n_assets
    - mu         : pd.Series of expected returns (daily), length n_assets
    - cov        : covariance matrix of returns (nxn)
    - shares_eq  : array of share counts, length n_assets
    - options    : list of dicts {idx,K,T,r,sigma,type,qty}
                   (empty list for equity-only)
    - horizon    : time step in years (1/252 ≈ 1 trading day)
    - alpha      : tail probability (0.05 → 95% VaR)
    - Nsim, seed : number of simulations and RNG seed
    Returns (var, cvar, pnl_array)
    """
    np.random.seed(seed)                             # reproducibility
    L = np.linalg.cholesky(cov)                      # Cholesky factor for correlations
    Z = np.random.randn(Nsim, len(S0))               # independent standard normals
    rets_sim = mu.values + Z.dot(L.T)                # correlated returns
    S_sim = S0 * (1 + rets_sim)                      # simulated spot prices

    # initial option prices (today)
    init_prices = [
        bs_price(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    pnl = np.empty(Nsim)                             # allocate profit-and-loss array
    for i in range(Nsim):
        # 1) Equity P&L: ΔS · shares
        pl_eq = shares_eq.dot(S_sim[i] - S0)

        # 2) Options P&L: repricing each under scenario i
        pl_opt = 0.0
        for j, op in enumerate(options):
            tau = max(op['T'] - horizon, 0)         # time-to-maturity after one step
            new_p = bs_price(
                S_sim[i, op['idx']],               # simulated spot
                op['K'], tau,
                op['r'], op['sigma'], op['type']
            )
            pl_opt += op['qty'] * (new_p - init_prices[j])

        pnl[i] = pl_eq + pl_opt                    # total P&L for scenario

    # 3) VaR: negative α-quantile of P&L
    var = -np.percentile(pnl, alpha * 100)
    # 4) CVaR: average loss beyond VaR threshold
    cvar = -pnl[pnl <= -var].mean()
    return var, cvar, pnl


def simulate_price_paths(S0, mu, cov, T_days=100, Nsim=1000, seed=42):
    """
    Simulate multi-day price trajectories using GBM with Cholesky.

    Parameters:
    - S0: initial prices (np.array, len = n_assets)
    - mu: expected daily returns (pd.Series or np.array, len = n_assets)
    - cov: daily return covariance matrix (n x n)
    - T_days: time horizon in trading days
    - Nsim: number of Monte Carlo paths
    - seed: random seed

    Returns:
    - paths: np.array of shape (T_days+1, Nsim, n_assets)
    """
    np.random.seed(seed)
    n_assets = len(S0)
    dt = 1 / 252

    # Cholesky factorization
    L = np.linalg.cholesky(cov)

    # Pre-allocate path array
    paths = np.zeros((T_days + 1, Nsim, n_assets))
    paths[0] = S0  # initial prices

    for t in range(1, T_days + 1):
        Z = np.random.randn(Nsim, n_assets)
        rets = mu + Z @ L.T
        paths[t] = paths[t - 1] * (1 + rets)

    return paths


def var_from_simulated_paths(paths, shares_eq, alpha=0.01):
    portf_value_paths = (paths * shares_eq).sum(axis=2)
    pnl = portf_value_paths[-1] - portf_value_paths[0]
    var = -np.percentile(pnl, alpha * 100)
    cvar = -pnl[pnl <= -var].mean()
    return var, cvar, pnl


# ────────── Historical-Simulation VaR/CVaR ──────────
def hist_simulations_var_es(returns_hist, S0, shares_eq, options,
                       alpha=0.05, horizon=1/252, seed=42):
    """
    Unified Historical Simulation VaR & CVaR for equity + options.
    - returns_hist: T×n DataFrame/array of historical returns
    - S0          : array[n] of spot prices
    - shares_eq   : array[n] share counts
    - options     : list of dicts (idx,K,T,r,sigma,type,qty); [] for equity-only
    - alpha       : tail prob (0.05 → 95% VaR)
    - horizon     : in years (1/252 default)
    - seed        : RNG seed
    Returns (var, cvar, pnl_array) in monetary units.
    """
    np.random.seed(seed)                             # reproducibility
    R = returns_hist.values if hasattr(returns_hist, "values") else np.asarray(returns_hist)
    T, _ = R.shape                                   # number of historical observations

    # 1) Bootstrap T scenarios with replacement
    idx = np.random.choice(T, size=T, replace=True)
    R_sim = R[idx]                                   # sampled returns
    S_sim = S0 * (1 + R_sim)                         # simulate prices

    # 2) Price options at time 0
    init_prices = [
        bs_price(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    pnl = np.empty(T)                                # allocate P&L array
    for i in range(T):
        # Equity P&L
        pl_eq = shares_eq.dot(S_sim[i] - S0)
        # Options P&L
        pl_opt = 0.0
        for j, op in enumerate(options):
            tau = max(op['T'] - horizon, 0)         # remaining maturity
            new_p = bs_price(
                S_sim[i, op['idx']],               # simulated spot
                op['K'], tau,
                op['r'], op['sigma'], op['type']
            )
            pl_opt += op['qty'] * (new_p - init_prices[j])
        pnl[i] = pl_eq + pl_opt                    # total P&L

    # VaR and CVaR from the empirical P&L distribution
    var = -np.percentile(pnl, alpha * 100)           # loss at α-quantile
    cvar = -pnl[pnl <= -var].mean()                  # average tail loss
    return var, cvar, pnl
