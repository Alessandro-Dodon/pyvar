import numpy as np
import pandas as pd
from scipy.stats import norm


# ───────────── Black-Scholes + Monte Carlo for Equities and Options ─────────────

def bs_price(S, K, tau, r, sigma, opt_type="call"):
    """
    Computes the Black-Scholes price for a European call or put option.

    Parameters:
    - S (float): Current stock price
    - K (float): Strike price
    - tau (float): Time to maturity (in years)
    - r (float): Risk-free interest rate
    - sigma (float): Volatility of the underlying asset
    - opt_type (str): "call" or "put"

    Returns:
    - float: Option price
    """
    if tau <= 0:
        # If the option has expired, return the intrinsic value
        val = max(0.0, S-K) if opt_type == "call" else max(0.0, K-S)
        return float(val)

    # Compute d1 and d2 terms of the Black-Scholes formula
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # Compute the price depending on option type
    if opt_type == "call":
        val = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)
    else:
        val = K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return float(val)


def mc_var_portfolio(S0, mu, cov, shares_eq, options,
                     horizon=1/252, alpha=0.05,
                     Nsim=50_000, seed=42):
    """
    Computes Value-at-Risk (VaR) and Conditional VaR (CVaR) of a portfolio using
    Monte Carlo simulation, considering equities and options.

    Parameters:
    - S0 (np.array): Initial stock prices
    - mu (pd.Series): Expected returns for each asset
    - cov (np.array): Covariance matrix of returns
    - shares_eq (np.array): Number of shares held in each equity
    - options (list of dict): List of options, each with keys:
        - 'idx', 'K', 'T', 'r', 'sigma', 'type', 'qty'
    - horizon (float): Time horizon in years (default is 1 day)
    - alpha (float): Confidence level for VaR (default 5%)
    - Nsim (int): Number of Monte Carlo simulations
    - seed (int): Random seed for reproducibility

    Returns:
    - var (float): Value-at-Risk
    - cvar (float): Conditional Value-at-Risk
    - pnl (np.array): Profit and loss distribution
    """
    np.random.seed(seed)

    # Generate correlated random returns using Cholesky decomposition
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(Nsim, len(S0))
    rets_sim = mu.values + Z.dot(L.T)

    # Simulated asset prices after one time step
    S_sim = S0 * (1 + rets_sim)

    # Compute initial option prices using Black-Scholes
    init_prices = [
        bs_price(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    # Initialize profit & loss vector
    pnl = np.empty(Nsim)

    for i in range(Nsim):
        # Equity P&L: dot product of shares and price changes
        pl_eq = shares_eq.dot(S_sim[i] - S0)

        # Options P&L: reprice each option at simulated prices
        pl_opt = 0.0
        for j, op in enumerate(options):
            # Price option with new spot and shortened maturity
            pj = bs_price(
                S_sim[i, op['idx']],
                op['K'],
                op['T'] - horizon,
                op['r'],
                op['sigma'],
                op['type']
            )
            pl_opt += op['qty'] * (pj - init_prices[j])

        # Total P&L for this scenario
        pnl[i] = pl_eq + pl_opt

    # Compute Value-at-Risk and Conditional Value-at-Risk
    var = -np.percentile(pnl, alpha * 100)
    cvar = -pnl[pnl <= -var].mean()

    return var, cvar, pnl



# ────────── Historical-Simulation VaR/CVaR for Equity+Options ──────────
def hist_var_portfolio(returns_hist, S0, shares_eq, options,
                       alpha=0.05, horizon=1/252, seed=42):
    """
    Historical‐Simulation VaR & CVaR for an equity + option portfolio.
    """
    # 1. Ensure reproducibility
    np.random.seed(seed)

    # 2. Convert input returns to a NumPy array (T × n_assets)
    R = returns_hist.values if hasattr(returns_hist, "values") else np.asarray(returns_hist)
    T, _ = R.shape  # T = number of historical observations

    # 3. Sample T scenarios with replacement from historical returns
    idx = np.random.choice(T, size=T, replace=True)
    R_sim = R[idx]               # shape (T, n_assets)
    S_sim = S0 * (1 + R_sim)     # simulate prices after one horizon step

    # 4. Compute initial option prices at time 0
    init_prices = [
        bs_price(
            S0[op['idx']],      # current spot for this option
            op['K'],            # strike
            op['T'],            # time to maturity
            op['r'],            # risk-free rate
            op['sigma'],        # volatility
            op['type']          # "call" or "put"
        )
        for op in options
    ]

    # 5. Allocate array for profit & loss of each scenario
    pnl = np.empty(T)

    # 6. Loop over each sampled scenario
    for i in range(T):
        # 6a. Equity P&L: shares × (simulated price – current price)
        pl_eq = shares_eq.dot(S_sim[i] - S0)

        # 6b. Options P&L: reprice each option under simulated spot and time decay
        pl_opt = 0.0
        for j, op in enumerate(options):
            # remaining time to maturity after one horizon
            tau = max(op['T'] - horizon, 0)

            # price the option in scenario i
            new_price = bs_price(
                S_sim[i, op['idx']],  # simulated spot for this option
                op['K'],              # strike
                tau,                  # updated time to maturity
                op['r'],              # risk-free rate
                op['sigma'],          # volatility
                op['type']            # option type
            )

            # accumulate P&L = quantity × (new price – initial price)
            pl_opt += op['qty'] * (new_price - init_prices[j])

        # 6c. Total P&L for scenario i
        pnl[i] = pl_eq + pl_opt

    # 7. Value-at-Risk: negative of the α-quantile of the P&L distribution
    var = -np.percentile(pnl, alpha * 100)

    # 8. Conditional VaR: average loss beyond the VaR threshold
    cvar = -pnl[pnl <= -var].mean()

    # 9. Return VaR, CVaR, and the full P&L series
    return var, cvar, pnl
