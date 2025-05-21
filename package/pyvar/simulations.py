"""
Simulation-Based VaR and ES Module
----------------------------------

Provides functions for estimating Value-at-Risk (VaR) and Expected Shortfall (ES)
using simulation-based techniques. These include parametric Monte Carlo methods,
and both historical and bootstrapped historical simulation methods.
Notice that backtesting is not implemented for this module.

Assumes a buy-and-hold portfolio strategy. If shares drastically change, the 
risk measures in this module should be recalculated.

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
"""


#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np                     
import pandas as pd                     
from scipy.stats import norm


# ----------------------------------------------------------
# Black-Scholes Pricing Function
# ----------------------------------------------------------
def black_scholes(S, K, tau, r, sigma, opt_type="call"):
    """
    Main
    ----
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
    Main
    ----
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
    profit_and_loss : np.ndarray
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
    returns_simulated = mu.values + Z.dot(L.T)
    S_simulated = S0 * (1 + returns_simulated)

    initial_option_prices = [
    black_scholes(
        S0[option['asset_index']], 
        option['K'], option['T'], option['r'], option['sigma'], option['type']
    )
    for option in options
    ]

    profit_and_loss = np.empty(simulations)
    for i in range(simulations):
        pnl_equity = shares.dot(S_simulated[i] - S0)
        pnl_options = 0.0
        for j, option in enumerate(options):
            tau = max(option['T'] - 1/252, 0)
            new_option_price = black_scholes(
                S_simulated[i, option['asset_index']],
                option['K'], tau,
                option['r'], option['sigma'], option['type']
            )
            pnl_options += option['qty'] * (new_option_price - initial_option_prices[j])
        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Multiday Monte Carlo VaR (Equity-only)
# ----------------------------------------------------------
def multiday_monte_carlo_var(price_data, shares,
                              confidence_level=0.99,
                              days_ahead=10, simulations=50_000, seed=1) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Main
    ----
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
        Number of trading days to simulate. Default is 10.
    simulations : int, optional
        Number of Monte Carlo paths. Default is 50,000.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    var : float
        Value-at-Risk estimate (monetary).
    profit_and_loss : np.ndarray
        Simulated P&L outcomes (length = simulations).
    portfolio_paths : np.ndarray
        Simulated portfolio value paths (shape: [days_ahead + 1, simulations]).
        Ready for plotting.

    Raises
    ------
    ValueError
        If 'shares' is not a 1D array or if its length does not match the number of assets
        (i.e., the number of columns in 'price_data').

    Notes
    -----
    - The longer the horizon, the more imprecise the arithmetic approximation. 
    """
    # Convert and validate shares input
    shares = np.asarray(shares)
    if shares.ndim != 1 or shares.shape[0] != price_data.shape[1]:
        raise ValueError(f"'shares' must be a 1D array with length equal to number of assets ({price_data.shape[1]}).")

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
        returns_simulated = mu + Z @ L.T
        asset_paths[t] = asset_paths[t - 1] * (1 + returns_simulated)

    portfolio_paths = (asset_paths * shares).sum(axis=2)
    profit_and_loss = portfolio_paths[-1] - portfolio_paths[0]
    var = -np.percentile(profit_and_loss, alpha * 100)

    return var, profit_and_loss, portfolio_paths


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
    Main
    ----
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
        Each dict: {'asset_index', 'K', 'T', 'r', 'sigma', 'type', 'qty'}.
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
    profit_and_loss : np.ndarray
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
        print("[warning] Argument 'simulations' is ignored because bootstrap=False.")

    if not bootstrap and seed is not None:
        print("[warning] Argument 'seed' is ignored because bootstrap=False.")

    if bootstrap:
        N = T if simulations is None else simulations
        indices = np.random.choice(T, size=N, replace=True)
        sampled_returns = returns[indices]
    else:
        sampled_returns = returns

    S_simulated = S0 * (1 + sampled_returns)

    initial_option_prices = [
        black_scholes(S0[option["asset_index"]], option["K"], option["T"], option["r"],
                      option["sigma"], option["type"])
        for option in options
    ]

    num_simulations = len(sampled_returns)
    profit_and_loss = np.empty(num_simulations)
    for i in range(num_simulations):
        pnl_equity = shares.dot(S_simulated[i] - S0)
        pnl_options = 0.0
        for j, option in enumerate(options):
            tau = max(option["T"] - dt, 0)
            new_option_price = black_scholes(
                S_simulated[i, option["asset_index"]],
                option["K"], tau,
                option["r"], option["sigma"], option["type"]
            )
            pnl_options += option["qty"] * (new_option_price - initial_option_prices[j])
        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Simulation Based ES (General)
# ----------------------------------------------------------
def simulation_es(var: float, profit_and_loss: np.ndarray) -> float:
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
    profit_and_loss : np.ndarray
        Simulated profit-and-loss outcomes (length = number of scenarios).

    Returns
    -------
    es : float
        Expected Shortfall (average loss beyond the VaR threshold), in monetary units.
    """
    es = -profit_and_loss[profit_and_loss <= -var].mean()
    return es
