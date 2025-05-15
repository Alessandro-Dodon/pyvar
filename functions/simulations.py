"""
Simulation-Based VaR and ES Module
----------------------------------

Provides functions for estimating Value-at-Risk (VaR) and Expected Shortfall (ES)
using simulation-based techniques. These include parametric Monte Carlo methods,
multiday geometric Brownian motion simulations, and both historical and 
bootstrapped historical simulation methods.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- black_scholes: Black-Scholes option pricing function (European options only)
- monte_carlo_var: Parametric Monte Carlo VaR for equity + options portfolios
- multiday_monte_carlo_var: Multiday Monte Carlo VaR (equity-only)
- historical_simulation_var: Historical or bootstrapped VaR (equity + options)
- simulation_es: General-purpose ES from any simulated P&L array

Notes
-----
- All returns are assumed to be daily and in decimal format (e.g., 0.01 = 1%).
- Time-to-maturity (T) in options must be expressed in years.
- Input data must be cleaned prior to use — NaNs are preserved by default.
- Simulation methods assume constant drift and volatility estimated from
  historical data. Options are revalued using static input parameters.
- Expected Shortfall (ES) is computed externally using `simulation_es`, 
  and is valid for all simulated P&L distributions.
- Backtesting is not implemented for this module but can be done externally.
"""

# TODO: check logic behind each simulation

#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np                     
import pandas as pd                     
from scipy.stats import norm
import warnings


# ----------------------------------------------------------
# Black-Scholes Pricing Function
# ----------------------------------------------------------
def black_scholes(S, K, tau, r, sigma, opt_type="call"):
    """
    Compute the Black-Scholes price of a European call or put option.

    Implements the closed-form solution for the fair value of a European-style
    option under the assumption of constant volatility and no arbitrage.
    Volatility must be provided externally.

    Parameters
    ----------
    S : float
        Spot price of the underlying asset.
    K : float
        Strike price of the option.
    tau : float
        Time to maturity in years.
    r : float
        Annual risk-free interest rate (continuous compounding).
    sigma : float
        Annualized volatility of the underlying asset's returns (decimal).
    opt_type : str, optional
        Option type: "call" or "put". Default is "call".

    Returns
    -------
    price : float
        Black-Scholes price of the option in monetary units.
    """
    if tau <= 0:
        # Option has expired → return intrinsic value
        return float(max(0.0, S - K) if opt_type == "call" else max(0.0, K - S))

    # Black-Scholes d1 and d2 terms
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # Option price formula
    if opt_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1))


# ----------------------------------------------------------
# Parametric Monte Carlo VaR (1-day Horizon)
# ----------------------------------------------------------
def monte_carlo_var(price_data, shares, options,
                    confidence_level=0.99,
                    simulations=50_000, seed=1) -> tuple[float, np.ndarray]:
    """
    Monte Carlo Value-at-Risk (VaR) for a 1-day horizon.

    Simulates correlated arithmetic returns to estimate 1-day profit-and-loss
    (P&L) for an equity + options portfolio, and computes the Value-at-Risk
    at the specified confidence level.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price series (T × N assets). Used to compute mean and covariance.
    shares : array-like
        Number of shares held per asset (length = N).
    options : list of dict
        List of option positions. Each dict must contain:
        {'idx', 'K', 'T', 'r', 'sigma', 'type', 'qty'}.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    simulations : int, optional
        Number of Monte Carlo scenarios. Default is 50,000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    var : float
        Value-at-Risk estimate (monetary units, positive).
    pnl : np.ndarray
        Simulated P&L distribution (length = simulations).
    """
    np.random.seed(seed)
    alpha = 1 - confidence_level

    returns = price_data.pct_change().dropna()
    mu = returns.mean()
    cov = returns.cov()
    S0 = price_data.iloc[-1].values

    L = np.linalg.cholesky(cov)
    Z = np.random.randn(simulations, len(S0))
    rets_sim = mu.values + Z.dot(L.T)
    S_sim = S0 * (1 + rets_sim)

    init_prices = [
        black_scholes(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    pnl = np.empty(simulations)
    for i in range(simulations):
        pl_eq = shares.dot(S_sim[i] - S0)
        pl_opt = 0.0
        for j, op in enumerate(options):
            tau = max(op['T'] - 1/252, 0)
            new_p = black_scholes(
                S_sim[i, op['idx']],
                op['K'], tau,
                op['r'], op['sigma'], op['type']
            )
            pl_opt += op['qty'] * (new_p - init_prices[j])
        pnl[i] = pl_eq + pl_opt

    var = -np.percentile(pnl, alpha * 100)
    return var, pnl


# ----------------------------------------------------------
# Multiday Monte Carlo VaR (Equity-only)
# ----------------------------------------------------------
def multiday_monte_carlo_var(price_data, shares,
                              confidence_level=0.99,
                              days_ahead=100, simulations=50_000, seed=1) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Multiday Monte Carlo Value-at-Risk (VaR) for an equity-only portfolio.

    Simulates arithmetic portfolio value paths over a fixed horizon with 
    correlated normal shocks, and computes the terminal Value-at-Risk.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price series (T × N assets) to estimate drift and volatility.
    shares : array-like
        Number of shares held in each asset (length = N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    days_ahead : int, optional
        Number of trading days to simulate. Default is 100.
    simulations : int, optional
        Number of Monte Carlo paths. Default is 50000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    var : float
        Value-at-Risk estimate (monetary).
    pnl : np.ndarray
        Simulated P&L outcomes (length = simulations).
    portfolio_paths : np.ndarray
        Simulated portfolio value paths (shape: [days_ahead + 1, simulations]).
        Ready for plotting.
    """
    np.random.seed(seed)
    alpha = 1 - confidence_level

    returns = price_data.pct_change().dropna()
    mu = returns.mean().values
    cov = returns.cov().values
    S0 = price_data.iloc[-1].values
    n_assets = len(S0)

    L = np.linalg.cholesky(cov)
    asset_paths = np.zeros((days_ahead + 1, simulations, n_assets))
    asset_paths[0] = S0

    for t in range(1, days_ahead + 1):
        Z = np.random.randn(simulations, n_assets)
        rets = mu + Z @ L.T
        asset_paths[t] = asset_paths[t - 1] * (1 + rets)

    portfolio_paths = (asset_paths * shares).sum(axis=2)
    pnl = portfolio_paths[-1] - portfolio_paths[0]
    var = -np.percentile(pnl, alpha * 100)

    return var, pnl, portfolio_paths


# ----------------------------------------------------------
# Historical (and Bootstrapped) Simulation VaR 
# ----------------------------------------------------------
def historical_simulation_var(
    price_data,
    shares,
    options,
    confidence_level=0.99,
    bootstrap=False,
    simulations=None,
    seed=None
) -> tuple[float, np.ndarray]:
    """
    Compute 1-day Value-at-Risk (VaR) using Historical Simulation or
    Bootstrapped Historical Simulation for an equity + options portfolio.

    Simulates daily P&L by applying historical (or resampled) return shocks 
    to current prices. Option values are revalued using Black-Scholes.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical prices (T × N assets) for return estimation.
    shares : array-like
        Number of shares per asset (length = N).
    options : list of dict
        Each dict: {'idx', 'K', 'T', 'r', 'sigma', 'type', 'qty'}.
    confidence_level : float, optional
        Confidence level for VaR (default: 0.99).
    bootstrap : bool, optional
        Whether to resample returns with replacement. Default is False.
    simulations : int or None, optional
        Number of bootstrap scenarios (ignored if bootstrap=False).
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    var : float
        Value-at-Risk estimate (monetary units, positive).
    pnl : np.ndarray
        Simulated P&L scenarios (length = T or simulations).

    Raises
    ------
    Warning
        If 'simulations' or 'seed' is set but ignored because bootstrap=False.
    """
    if seed is not None:
        np.random.seed(seed)

    alpha = 1 - confidence_level
    dt = 1 / 252
    returns = price_data.pct_change().dropna().values
    S0 = price_data.iloc[-1].values
    T = len(returns)

    if not bootstrap and simulations is not None:
        warnings.warn("Argument 'simulations' is ignored because bootstrap=False.")

    if not bootstrap and seed is not None:
        warnings.warn("Argument 'seed' is ignored because bootstrap=False.")

    if bootstrap:
        N = T if simulations is None else simulations
        idx = np.random.choice(T, size=N, replace=True)
        rets_sampled = returns[idx]
    else:
        rets_sampled = returns

    S_sim = S0 * (1 + rets_sampled)

    init_prices = [
        black_scholes(S0[op["idx"]], op["K"], op["T"], op["r"], op["sigma"], op["type"])
        for op in options
    ]

    Nsim = len(rets_sampled)
    pnl = np.empty(Nsim)
    for i in range(Nsim):
        pl_eq = shares.dot(S_sim[i] - S0)
        pl_opt = 0.0
        for j, op in enumerate(options):
            tau = max(op["T"] - dt, 0)
            new_p = black_scholes(
                S_sim[i, op["idx"]],
                op["K"], tau,
                op["r"], op["sigma"], op["type"]
            )
            pl_opt += op["qty"] * (new_p - init_prices[j])
        pnl[i] = pl_eq + pl_opt

    var = -np.percentile(pnl, alpha * 100)
    return var, pnl


# ----------------------------------------------------------
# Simulation Based ES (General)
# ----------------------------------------------------------
def simulation_es(var: float, pnl: np.ndarray) -> float:
    """
    Compute Expected Shortfall (ES) from a simulated P&L distribution.

    This function works with any simulation-based method — including parametric
    Monte Carlo, multiday Monte Carlo, historical simulation, and bootstrapped
    historical simulation — and returns the Expected Shortfall corresponding to
    the same confidence level used to compute the given Value-at-Risk (VaR).

    Parameters
    ----------
    var : float
        Value-at-Risk threshold (positive monetary value).
    pnl : np.ndarray
        Simulated profit-and-loss outcomes (length = number of scenarios).

    Returns
    -------
    es : float
        Expected Shortfall (average loss beyond the VaR threshold), in monetary units.
    """
    es = -pnl[pnl <= -var].mean()
    return es
