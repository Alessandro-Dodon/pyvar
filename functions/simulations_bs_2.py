#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np                     
import pandas as pd                     
from scipy.stats import norm
import warnings

###################################################################
# Note: add a function specifically for backtesting, repeating the
#       simulations in a loop? or build that into the simulations 
#       directly?
###################################################################

# ----------------------------------------------------------
# Black-Scholes Pricing Function
# ----------------------------------------------------------
def black_scholes(S, K, tau, r, sigma, opt_type="call"):
    """
    Compute the Black-Scholes price of a European call or put option.

    Parameters:
    - S : float
        Current spot price of the underlying asset.
    - K : float
        Strike price of the option.
    - tau : float
        Time to maturity in years.
    - r : float
        Risk-free interest rate (annualized, continuous compounding).
    - sigma : float
        Volatility of the underlying asset (annualized).
    - opt_type : str
        'call' or 'put' to specify the option type.

    Returns:
    - float: Option price in monetary units.
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
# Parametric Monte Carlo VaR and ES (1-day Horizon)
# ----------------------------------------------------------
def monte_carlo(price_data, shares, options,
                  confidence_level=0.99,
                  simulations=50_000, seed=1):
    """
    Monte Carlo (parametric) VaR & Expected Shortfall for an equity + options portfolio.
    This version assumes a fixed 1-day ahead horizon (1/252 year). 
    No scaling for multi-day VaR is performed.

    Parameters:
    - price_data : pd.DataFrame
        Historical price series (T × n_assets). Used to compute returns, drift, and covariance.
    - shares : array-like
        Number of shares held per asset (length = n_assets).
    - options : list of dicts
        List of option positions. Each dict must contain:
        {idx, K, T, r, sigma, type, qty}. Use an empty list for equity-only.
    - confidence_level : float
        Confidence level (e.g., 0.95 → 95% VaR).
    - simulations : int
        Number of simulation scenarios.
    - seed : int
        Random seed for reproducibility.

    Returns:
    - var : float
        Value-at-Risk estimate (monetary).
    - es : float
        Expected Shortfall estimate (monetary).
    - pnl : np.ndarray
        Simulated profit-and-loss array (length = simulations).
    """
    np.random.seed(seed)
    alpha = 1 - confidence_level

    # Estimate parameters from price data
    returns = price_data.pct_change().dropna()
    mu = returns.mean()
    cov = returns.cov()
    S0 = price_data.iloc[-1].values

    # Simulate correlated 1-day returns and future prices
    L = np.linalg.cholesky(cov)
    Z = np.random.randn(simulations, len(S0))
    rets_sim = mu.values + Z.dot(L.T)
    S_sim = S0 * (1 + rets_sim)

    # Initial option prices at t = 0
    init_prices = [
        black_scholes(S0[op['idx']], op['K'], op['T'], op['r'], op['sigma'], op['type'])
        for op in options
    ]

    pnl = np.empty(simulations)
    for i in range(simulations):
        # Equity P&L
        pl_eq = shares.dot(S_sim[i] - S0)

        # Option P&L after 1 day (adjusted maturity)
        pl_opt = 0.0
        for j, op in enumerate(options):
            tau = max(op['T'] - 1/252, 0)  # always assumes 1-day step
            new_p = black_scholes(
                S_sim[i, op['idx']],
                op['K'], tau,
                op['r'], op['sigma'], op['type']
            )
            pl_opt += op['qty'] * (new_p - init_prices[j])

        pnl[i] = pl_eq + pl_opt

    # Risk metrics
    var = -np.percentile(pnl, alpha * 100)
    es = -pnl[pnl <= -var].mean()

    return var, es, pnl
 
 
# ----------------------------------------------------------
# Multiday Monte Carlo VaR and ES (Equity-only)
# ----------------------------------------------------------
def multiday_monte_carlo(price_data, shares,
                       confidence_level=0.99,
                       days_ahead=100, simulations=1000, seed=1):
    """
    Multiday Monte Carlo simulation for Value-at-Risk (VaR) and Expected Shortfall (ES)
    of an equity-only portfolio using geometric Brownian motion. Options are here not 
    included, as it would require a more complex simulation to model their price over time.

    Inputs:
    - price_data : pd.DataFrame
        Historical price series (T × n_assets) to estimate drift and volatility.
    - shares : array-like
        Number of shares held in each asset (length = n_assets).
    - confidence_level : float
        Confidence level for risk metrics (e.g., 0.99 → 1% tail).
    - days_ahead : int
        Number of trading days to simulate.
    - simulations : int
        Number of Monte Carlo paths.
    - seed : int
        Random seed for reproducibility.

    Returns:
    - var : float
        Value-at-Risk estimate (monetary) after days_ahead.
    - es : float
        Expected Shortfall estimate (monetary) after days_ahead.
    - pnl : np.ndarray
        Simulated profit-and-loss vector (length = simulations).
    - paths : np.ndarray
        Simulated price trajectories (shape: [days + 1, simulations, n_assets])
    """
    np.random.seed(seed)
    alpha = 1 - confidence_level

    # Estimate drift and covariance from historical returns
    returns = price_data.pct_change().dropna()
    mu = returns.mean().values
    cov = returns.cov().values
    S0 = price_data.iloc[-1].values

    # Cholesky factor for correlation structure
    L = np.linalg.cholesky(cov)
    n_assets = len(S0)

    # Pre-allocate price path array
    paths = np.zeros((days_ahead + 1, simulations, n_assets))
    paths[0] = S0  # initial prices

    # Simulate GBM price paths
    for t in range(1, days_ahead + 1):
        Z = np.random.randn(simulations, n_assets)
        rets = mu + Z @ L.T
        paths[t] = paths[t - 1] * (1 + rets)

    # Portfolio P&L calculation
    portfolio_paths = (paths * shares).sum(axis=2)
    pnl = portfolio_paths[-1] - portfolio_paths[0]

    # Risk measures
    var = -np.percentile(pnl, alpha * 100)
    es = -pnl[pnl <= -var].mean()

    return var, es, pnl, paths


# ----------------------------------------------------------
# Historical (and Bootstrapped) Simulation VaR and ES
# ----------------------------------------------------------
def historical_simulation( # Separate or toghether?
    price_data,
    shares,
    options,
    confidence_level=0.99,
    bootstrap=False,
    simulations=None,
    seed=None # Also ignore if the HS is selected, as it makes no difference for HS (double check)
):
    """
    Estimate 1-day portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using either Historical Simulation or Bootstrapped Historical Simulation.

    This method simulates P&L scenarios by applying historical (or resampled) 
    1-day return vectors to current asset prices. Option values are recalculated 
    in each scenario using the Black-Scholes formula with reduced time to maturity.

    Parameters:
    - price_data : pd.DataFrame
        Historical price series (T × n_assets). Used to compute daily returns.
    - shares : array-like
        Number of shares held in each asset (length = n_assets).
    - options : list of dicts
        Each dict must include: {idx, K, T, r, sigma, type, qty}.
    - confidence_level : float, default 0.99
        Confidence level for VaR and ES (e.g., 0.99 → 1% left tail).
    - bootstrap : bool, default False
        If True, performs bootstrapped historical simulation (sampling with replacement).
        If False, uses raw historical returns.
    - simulations : int or None, default None
        Number of bootstrap simulations (only used if bootstrap=True).
        If None, defaults to the number of historical observations.
    - seed : int or None, default None
        Random seed for reproducibility (used only if bootstrap=True).

    Returns:
    - var : float
        Estimated 1-day Value-at-Risk (monetary units).
    - es : float
        Estimated 1-day Expected Shortfall (monetary units).
    - pnl : np.ndarray
        Simulated P&L scenarios.
    """
    if seed is not None:
        np.random.seed(seed)

    alpha = 1 - confidence_level
    dt = 1 / 252
    returns = price_data.pct_change().dropna().values
    S0 = price_data.iloc[-1].values
    T = len(returns)

    # Warn if simulations is set but not used
    if not bootstrap and simulations is not None:
        warnings.warn("Argument 'simulations' is ignored because bootstrap=False.")

    # Sampling step
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
    es = -pnl[pnl <= -var].mean()
    return var, es, pnl


######### Idea for separating functions:
'''def compute_expected_shortfall(pnl, var_threshold):
    """
    Compute Expected Shortfall (ES) given a P&L array and a VaR threshold.

    Parameters:
    - pnl : np.ndarray
        Simulated profit-and-loss array.
    - var_threshold : float
        VaR threshold (positive number, i.e., already -percentile value).

    Returns:
    - es : float
        Expected Shortfall (monetary units).
    """
    return -pnl[pnl <= -var_threshold].mean()
'''