"""
Simulation-Based VaR and Expected Shortfall Module
--------------------------------------------------

This module provides simulation-based methods to estimate Value-at-Risk (VaR) and 
Expected Shortfall (ES) for equity and options portfolios. It includes Monte Carlo, 
historical, filtered, and bootstrapped approaches, both in univariate and multivariate 
settings, with support for time-varying volatility and heavy-tailed distributions 
(e.g., via GARCH and Gaussian Mixture Models).

Volatility risk and interest rate risk are not explicitly modeled. No backtesting 
routines are included.

Assumes a static (buy-and-hold) portfolio strategy. If portfolio composition changes 
significantly, all risk measures should be recalculated.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- monte_carlo_simulation_var: One-day parametric Monte Carlo VaR (equity + options)
- multiday_monte_carlo_simulation_var: Multiday parametric MC VaR (equity-only)
- historical_simulation_var: Historical or bootstrapped VaR (equity + options)
- weighted_historical_simulation_var: Exponentially weighted VaR
- filtered_historical_simulation_var: GARCH-filtered VaR with bootstrapping
- multiday_garch_simulation_var_univariate: Univariate GARCH multiday VaR (equity-only)
- multiday_gmm_simulation_var_univariate: Univariate GMM multiday VaR (equity-only)
- gmm_monte_carlo_simulation_var: One-day GMM VaR (equity + options)
- multiday_gmm_monte_carlo_simulation_var: Multiday GMM VaR (equity-only)
- simulation_es: General-purpose Expected Shortfall from simulated profit-and-loss
"""


#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np                     
from .utils import fit_garch_model
from sklearn.mixture import GaussianMixture
from arch import arch_model
from .options import black_scholes_pricing


#----------------------------------------------------------
# Multiday Garch Simulation VaR (Univariate, Equity-only)
# ----------------------------------------------------------
def multiday_garch_simulation_var_univariate(price_series, forecast_days=22, 
                   n_samples=1000, confidence_level=0.99, 
                   wealth=100_000, distribution="normal", seed=1):
    """
    Main
    ----
    Multiday GARCH(1,1) simulation-based Value-at-Risk (VaR) for equity portfolios.

    Simulates forward price paths using a univariate GARCH(1,1) process on log-returns
    to estimate the profit-and-loss (P&L) distribution over a multi-day horizon. 
    Supports normal and t-distributed residuals.

    Parameters
    ----------
    price_series : pd.Series
        Historical price series for a single asset.
    forecast_days : int, optional
        Number of days to simulate ahead. Default is 22 (≈ 1 month).
    n_samples : int, optional
        Number of simulated price paths. Default is 1000.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    wealth : float, optional
        Initial monetary value of the asset. Default is 100,000.
    distribution : str, optional
        Distribution for GARCH residuals: "normal" or "t". Default is "normal".
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    var : float
        Estimated Value-at-Risk (monetary units, positive).
    profit_and_loss : np.ndarray
        Simulated P&L distribution (length = n_samples).
    price_paths : np.ndarray
        Simulated price paths (n_samples × (forecast_days + 1)).

    Notes
    -----
    - GARCH model fitted to historical log-returns .
    - The t-distribution is standardized to have unit variance.
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
    if seed is not None:
        np.random.seed(seed)

    # Use log-returns in percent units
    return_series = np.log(price_series / price_series.shift(1)).dropna() * 100

    model = arch_model(return_series, vol='Garch', p=1, q=1, dist=distribution)
    model_fit = model.fit(disp="off")

    omega = model_fit.params["omega"]
    alpha = model_fit.params["alpha[1]"]
    beta = model_fit.params["beta[1]"]
    mu = 0.0
    df = model_fit.params["nu"] if distribution == "t" else None

    last_vol = model_fit.conditional_volatility.iloc[-1].item()
    last_ret = return_series.iloc[-1].item()

    price_paths = np.empty((n_samples, forecast_days + 1))
    profit_and_loss = np.empty(n_samples)

    for i in range(n_samples):
        path = np.empty(forecast_days + 1)
        price = wealth
        volatility = last_vol
        ret = last_ret
        path[0] = price

        for t in range(1, forecast_days + 1):
            volatility = np.sqrt(omega + alpha * ret**2 + beta * volatility**2)
            if distribution == "t":
                shock = np.random.standard_t(df)
                shock /= np.sqrt(df / (df - 2))
            else:
                shock = np.random.normal()
            ret_scaled = mu + volatility * shock
            ret = ret_scaled / 100  # convert percent back to decimal
            price *= np.exp(ret)
            path[t] = price

        price_paths[i] = path
        profit_and_loss[i] = path[-1] - path[0]

    var = -np.percentile(profit_and_loss, 100 * (1 - confidence_level))
    return var, profit_and_loss, price_paths


#----------------------------------------------------------
# Multiday GMM Simulation VaR (Univariate, Equity-only)
# ----------------------------------------------------------
def multiday_gmm_simulation_var_univariate(price_series, forecast_days=22,
                          n_samples=1000, confidence_level=0.99,
                          wealth=100_000, seed=1, n_components=2):
    """
    Main
    ----
    Multiday Gaussian Mixture Model (GMM) simulation-based Value-at-Risk (VaR)
    for a univariate equity portfolio.

    Fits a GMM to historical log-returns and simulates forward price paths
    by sampling from the mixture components. Captures return asymmetry and
    heavy tails beyond the Gaussian assumption.

    Parameters
    ----------
    price_series : pd.Series
        Historical price series for a single asset.
    forecast_days : int, optional
        Number of days to simulate ahead. Default is 22.
    n_samples : int, optional
        Number of simulation paths. Default is 1000.
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.99.
    wealth : float, optional
        Initial portfolio value in monetary units. Default is 100,000.
    seed : int, optional
        Random seed for reproducibility.
    n_components : int, optional
        Number of Gaussian components in the mixture. Default is 2.

    Returns
    -------
    var : float
        Estimated Value-at-Risk (monetary units, positive).
    profit_and_loss : np.ndarray
        Simulated P&L distribution (length = n_samples).
    price_paths : np.ndarray
        Simulated price paths (n_samples × (forecast_days + 1)).
    """
    if wealth is not None and wealth <= 0:
        raise ValueError("wealth must be strictly positive if provided")
    
    if seed is not None:
        np.random.seed(seed)

    # Compute log-returns and remove empirical mean
    log_returns = np.log(price_series / price_series.shift(1)).dropna().values.reshape(-1, 1)

    # Fit GMM with n_components
    gmm = GaussianMixture(n_components=n_components, random_state=seed)
    gmm.fit(log_returns)

    weights = gmm.weights_
    means = gmm.means_.flatten()
    std_devs = np.sqrt(gmm.covariances_.flatten())

    # Simulate paths
    price_paths = np.empty((n_samples, forecast_days + 1))
    profit_and_loss = np.empty(n_samples)

    for i in range(n_samples):
        path = np.empty(forecast_days + 1)
        price = wealth
        path[0] = price

        for t in range(1, forecast_days + 1):
            component = np.random.choice(n_components, p=weights)
            shock = np.random.normal(loc=means[component], scale=std_devs[component])
            price *= np.exp(shock)
            path[t] = price

        price_paths[i] = path
        profit_and_loss[i] = path[-1] - path[0]

    # Compute VaR
    alpha = 1 - confidence_level
    var = -np.percentile(profit_and_loss, 100 * alpha)

    return var, profit_and_loss, price_paths


# ----------------------------------------------------------
# Parametric Monte Carlo VaR (1-day Horizon)
# ----------------------------------------------------------
def monte_carlo_simulation_var(
    price_data,
    shares=None,
    options=None,
    confidence_level=0.99,
    simulations=50_000,
    seed=1
) -> tuple[float, np.ndarray]:
    """
    Main
    ----
    Monte Carlo Value-at-Risk (VaR) for 1-day ahead.

    Simulates correlated returns to estimate 1-day profit-and-loss (P&L)
    for an equity + options portfolio, and computes the Value-at-Risk
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

    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    mu = log_returns.mean()
    cov = log_returns.cov()
    S0 = price_data.iloc[-1].values

    L = np.linalg.cholesky(cov)
    Z = np.random.randn(simulations, len(S0))
    log_returns_simulated = mu.values + Z @ L.T
    S_simulated = S0 * np.exp(log_returns_simulated)

    # Option valuation at time 0
    initial_option_prices = []
    if options:
        for opt in options:
            S_init = S0[opt["asset_index"]]
            initial_price = black_scholes_pricing(
                S_init, opt["K"], opt["T"], opt["r"], opt["sigma"], opt["type"]
            )
            initial_option_prices.append(initial_price)

    profit_and_loss = np.empty(simulations)

    for i in range(simulations):
        pnl_equity = 0.0
        if shares is not None:
            pnl_equity = shares.dot(S_simulated[i] - S0)

        pnl_options = 0.0
        if options:
            for j, opt in enumerate(options):
                tau = max(opt["T"] - 1 / 252, 0)
                new_price = black_scholes_pricing(
                    S_simulated[i, opt["asset_index"]],
                    opt["K"], tau, opt["r"], opt["sigma"], opt["type"]
                )
                quantity = opt["quantity"]
                contract_size = opt["contract_size"]
                pnl_options += quantity * contract_size * (new_price - initial_option_prices[j])

        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Multiday Monte Carlo VaR (Equity-only)
# ----------------------------------------------------------
def multiday_monte_carlo_simulation_var(
    price_data,
    shares,
    confidence_level=0.99,
    days_ahead=10,
    simulations=50_000,
    seed=1
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Main
    ----
    Multiday Monte Carlo Value-at-Risk (VaR) for an equity-only portfolio.

    Simulates portfolio value paths over a fixed horizon with 
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
    """
    shares = np.asarray(shares)
    if shares.ndim != 1 or shares.shape[0] != price_data.shape[1]:
        raise ValueError("Shape mismatch between shares and price_data.")

    np.random.seed(seed)
    alpha = 1 - confidence_level

    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    mu = log_returns.mean().values
    cov = log_returns.cov().values
    S0 = price_data.iloc[-1].values
    n_assets = len(S0)

    L = np.linalg.cholesky(cov)
    asset_paths = np.zeros((days_ahead + 1, simulations, n_assets))
    asset_paths[0] = S0

    for t in range(1, days_ahead + 1):
        Z = np.random.randn(simulations, n_assets)
        log_returns_simulated = mu + Z @ L.T
        asset_paths[t] = asset_paths[t - 1] * np.exp(log_returns_simulated)

    # Apply shares (total notional positions: quantity × contract_size)
    portfolio_paths = (asset_paths * shares).sum(axis=2)
    profit_and_loss = portfolio_paths[-1] - portfolio_paths[0]
    var = -np.percentile(profit_and_loss, alpha * 100)

    return var, profit_and_loss, portfolio_paths


# ----------------------------------------------------------
# Historical (and Bootstrapped) Simulation VaR 
# ----------------------------------------------------------
def historical_simulation_var(
    price_data,
    shares=None,
    options=None,
    confidence_level=0.99,
    bootstrap=False,
    simulations=None,
    seed=None
) -> tuple[float, np.ndarray]:
    """
    Main
    ----
    Computes 1-day ahead Value-at-Risk (VaR) using Historical Simulation or
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

    log_returns = np.log(price_data / price_data.shift(1)).dropna().values
    S0 = price_data.iloc[-1].values
    T = len(log_returns)

    if not bootstrap and simulations is not None:
        print("[warning] Argument 'simulations' is ignored because bootstrap=False.")
    if not bootstrap and seed is not None:
        print("[warning] Argument 'seed' is ignored because bootstrap=False.")

    if bootstrap:
        N = T if simulations is None else simulations
        indices = np.random.choice(T, size=N, replace=True)
        sampled_log_returns = log_returns[indices]
    else:
        sampled_log_returns = log_returns
        N = T

    S_simulated = S0 * np.exp(sampled_log_returns)

    # Prepare option prices if any
    if options:
        initial_option_prices = [
            black_scholes_pricing(S0[opt["asset_index"]], opt["K"], opt["T"], opt["r"],
                          opt["sigma"], opt["type"])
            for opt in options
        ]
    else:
        initial_option_prices = []

    profit_and_loss = np.empty(N)

    for i in range(N):
        # Equity P&L (if any)
        if shares is not None:
            pnl_equity = shares.dot(S_simulated[i] - S0)
        else:
            pnl_equity = 0.0

        # Options P&L (if any)
        pnl_options = 0.0
        if options:
            for j, opt in enumerate(options):
                tau = max(opt["T"] - dt, 0)
                new_price = black_scholes_pricing(
                    S_simulated[i, opt["asset_index"]],
                    opt["K"], tau, opt["r"], opt["sigma"], opt["type"]
                )
                pnl_options += opt["quantity"] * opt["contract_size"] * (new_price - initial_option_prices[j])

        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Weighted Historical Simulation VaR 
# ----------------------------------------------------------
def weighted_historical_simulation_var(
    price_data,
    shares=None,
    options=None,
    confidence_level=0.99,
    lambda_decay=0.97
) -> tuple[float, np.ndarray]:
    """
    Main
    ----
    Weighted Historical Simulation (WHS) Value-at-Risk (VaR) using exponential decay.

    Computes 1-day ahead VaR for portfolios composed of equities and/or options.
    Weights past returns using an exponential scheme that gives more importance
    to recent observations. The VaR is estimated using a weighted empirical quantile.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical asset prices (T × N).
    shares : np.ndarray or None
        Portfolio equity exposures in number of shares per asset. If None, only options are considered.
    options : list[dict] or None
        List of option positions. Each dictionary must contain:
        - 'asset_index', 'K', 'T', 'r', 'sigma', 'type'
        - 'quantity', 'contract_size'
    confidence_level : float, optional
        Confidence level for VaR estimation (e.g., 0.99). Default is 0.99.
    lambda_decay : float, optional
        Exponential decay factor λ ∈ (0,1). Default is 0.97.

    Returns
    -------
    var : float
        Estimated Value-at-Risk (monetary units, positive).
    profit_and_loss : np.ndarray
        Simulated 1-day profit-and-loss distribution.
    
    Notes
    -----
    - Weights sum to 1 and decay exponentially backward in time.
    """
    dt = 1 / 252
    returns = price_data.pct_change().dropna().values
    S0 = price_data.iloc[-1].values
    T = len(returns)

    # Exponential decay weights (most recent = highest)
    raw_weights = np.array([lambda_decay**(T - 1 - t) for t in range(T)])
    weights = raw_weights / raw_weights.sum()

    # Simulated equity prices
    S_simulated = S0 * (1 + returns)

    # Option valuation at t=0
    if options:
        initial_option_prices = [
            black_scholes_pricing(S0[opt["asset_index"]], opt["K"], opt["T"],
                          opt["r"], opt["sigma"], opt["type"])
            for opt in options
        ]
    else:
        initial_option_prices = []

    # Compute P&L
    profit_and_loss = np.empty(T)

    for i in range(T):
        # Equity P&L
        pnl_equity = shares.dot(S_simulated[i] - S0) if shares is not None else 0.0

        # Options P&L
        pnl_options = 0.0
        if options:
            for j, opt in enumerate(options):
                tau = max(opt["T"] - dt, 0)
                new_option_price = black_scholes_pricing(
                    S_simulated[i, opt["asset_index"]],
                    opt["K"], tau, opt["r"], opt["sigma"], opt["type"]
                )
                pnl_options += opt["quantity"] * opt["contract_size"] * (new_option_price - initial_option_prices[j])

        profit_and_loss[i] = pnl_equity + pnl_options

    # Weighted quantile estimation
    sorted_indices = np.argsort(profit_and_loss)
    sorted_pnl = profit_and_loss[sorted_indices]
    sorted_weights = weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    var_index = np.searchsorted(cumulative_weights, 1 - confidence_level)
    var = -sorted_pnl[var_index]

    return var, profit_and_loss


# ----------------------------------------------------------
# Filtered Historical (and Bootstrapped) Simulation VaR 
# ----------------------------------------------------------
def filtered_historical_simulation_var(
    price_data,
    shares=None,
    options=None,
    confidence_level=0.99,
    bootstrap=True,
    simulations=50_000,
    seed=None,
    p=1,
    q=1,
    model="GARCH",
    distribution="normal"
) -> tuple[float, np.ndarray]:
    """
    Main
    ----
    Filtered Historical Simulation (FHS) Value-at-Risk (VaR) using GARCH filtering and empirical residuals.

    Estimates 1-day VaR for portfolios of equities and/or options using a GARCH-filtered return series.
    Simulated returns are constructed by rescaling standardized residuals from fitted GARCH models,
    with optional bootstrapping. Suitable for capturing time-varying volatility.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical asset prices (T × N).
    shares : np.ndarray or None
        Number of shares held per asset. If None, only options are considered.
    options : list[dict] or None
        List of option positions. Each dictionary must contain:
        - 'asset_index', 'K', 'T', 'r', 'sigma', 'type'
        - 'quantity', 'contract_size'
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    bootstrap : bool, optional
        If True, resamples residuals with replacement. If False, uses full residual history.
    simulations : int, optional
        Number of simulated return paths (only used if bootstrap=True). Default is 50,000.
    seed : int or None
        Random seed for reproducibility.
    p, q : int, optional
        GARCH model parameters. Default is (1, 1).
    model : str, optional
        GARCH model type ("GARCH", "EGARCH", "APARCH", etc.). Default is "GARCH".
    distribution : str, optional
        Distribution for residuals ("normal", "t", "skewt", etc.). Default is "normal".

    Returns
    -------
    var : float
        Estimated Value-at-Risk (monetary units, positive).
    profit_and_loss : np.ndarray
        Simulated 1-day profit-and-loss distribution.
    
    Notes
    -----
    - Each asset is modeled independently with its own GARCH process.
    - Latest conditional volatilities are used to rescale innovations.
    """
    if seed is not None:
        np.random.seed(seed)

    alpha = 1 - confidence_level
    dt = 1 / 252

    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    returns_array = log_returns.values
    S0 = price_data.iloc[-1].values
    T = len(log_returns)
    N_assets = returns_array.shape[1]

    if not bootstrap and simulations is not None:
        print("[warning] Argument 'simulations' is ignored because bootstrap=False.")
    if not bootstrap and seed is not None:
        print("[warning] Argument 'seed' is ignored because bootstrap=False.")

    # GARCH: estimate residuals and last volatilities
    standardized_residuals = np.empty_like(returns_array)
    latest_vols = np.empty(N_assets)

    for i in range(N_assets):
        series = returns_array[:, i]
        residuals, cond_vol, latest_vol, params = fit_garch_model(
            returns=series, p=p, q=q, model=model, distribution=distribution
        )
        standardized_residuals[:, i] = residuals
        latest_vols[i] = latest_vol

    # Resample standardized residuals
    if bootstrap:
        N = T if simulations is None else simulations
        indices = np.random.choice(T, size=N, replace=True)
        sampled_residuals = standardized_residuals[indices]
    else:
        sampled_residuals = standardized_residuals
        N = T

    # Simulate prices
    scaled_returns = sampled_residuals * latest_vols
    S_simulated = S0 * np.exp(scaled_returns)

    # Option valuation at t=0
    if options:
        initial_option_prices = [
            black_scholes_pricing(S0[opt["asset_index"]], opt["K"], opt["T"],
                          opt["r"], opt["sigma"], opt["type"])
            for opt in options
        ]
    else:
        initial_option_prices = []

    # Compute P&L
    profit_and_loss = np.empty(N)

    for i in range(N):
        # Equity P&L
        pnl_equity = shares.dot(S_simulated[i] - S0) if shares is not None else 0.0

        # Options P&L
        pnl_options = 0.0
        if options:
            for j, opt in enumerate(options):
                tau = max(opt["T"] - dt, 0)
                new_price = black_scholes_pricing(
                    S_simulated[i, opt["asset_index"]],
                    opt["K"], tau, opt["r"], opt["sigma"], opt["type"]
                )
                pnl_options += opt["quantity"] * opt["contract_size"] * (new_price - initial_option_prices[j])

        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Multiday GMM Simulation VaR (Multivariate, Equity-only)
# ----------------------------------------------------------
def gmm_monte_carlo_simulation_var(
    price_data,
    shares=None,
    options=None,
    confidence_level=0.99,
    simulations=50_000,
    seed=1,
    n_components=3
) -> tuple[float, np.ndarray]:
    """
    Main
    ----
    One-day Monte Carlo Value-at-Risk (VaR) using a multivariate Gaussian Mixture Model (GMM).

    Simulates 1-day ahead asset returns using a fitted multivariate GMM,
    capturing fat tails and regime switching. Supports equity-only, options-only, 
    or combined portfolios. Returns both the VaR and the full distribution of 
    simulated profit-and-loss outcomes.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data of the underlying assets (T × N).
    shares : array-like or pd.Series, optional
        Number of shares held in each asset. If None, only options are considered.
    options : list[dict], optional
        Option positions, each with:
        - 'asset_index', 'K', 'T', 'r', 'sigma', 'type'
        - 'quantity', 'contract_size'
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    simulations : int, optional
        Number of Monte Carlo simulations. Default is 50,000.
    seed : int, optional
        Random seed for reproducibility. Default is 1.
    n_components : int, optional
        Number of components in the GMM. Default is 3.

    Returns
    -------
    var : float
        Estimated Value-at-Risk in monetary units (positive).
    profit_and_loss : np.ndarray
        Simulated profit-and-loss distribution.
    """
    np.random.seed(seed)
    alpha = 1 - confidence_level

    log_returns = np.log(price_data / price_data.shift(1)).dropna().values
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gmm.fit(log_returns)

    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_
    S0 = price_data.iloc[-1].values

    profit_and_loss = np.empty(simulations)

    # Precompute option prices at time 0
    initial_option_prices = []
    if options:
        for opt in options:
            S_init = S0[opt["asset_index"]]
            initial_price = black_scholes_pricing(
                S_init, opt["K"], opt["T"], opt["r"], opt["sigma"], opt["type"]
            )
            initial_option_prices.append(initial_price)

    for i in range(simulations):
        # Draw component and sample return vector
        k = np.random.choice(n_components, p=weights)
        shock = np.random.multivariate_normal(means[k], covariances[k])
        S_simulated = S0 * np.exp(shock)

        pnl_equity = 0.0
        if shares is not None:
            pnl_equity = shares.dot(S_simulated - S0)

        pnl_options = 0.0
        if options:
            for j, opt in enumerate(options):
                tau = max(opt["T"] - 1 / 252, 0)
                new_price = black_scholes_pricing(
                    S_simulated[opt["asset_index"]],
                    opt["K"], tau, opt["r"], opt["sigma"], opt["type"]
                )
                quantity = opt["quantity"]
                contract_size = opt["contract_size"]
                pnl_options += quantity * contract_size * (new_price - initial_option_prices[j])

        profit_and_loss[i] = pnl_equity + pnl_options

    var = -np.percentile(profit_and_loss, alpha * 100)
    return var, profit_and_loss


# ----------------------------------------------------------
# Multiday GMM Simulation VaR (Multivariate, Equity-only)
# ----------------------------------------------------------
def multiday_gmm_monte_carlo_simulation_var(
    price_data,
    shares,
    confidence_level=0.99,
    days_ahead=10,
    simulations=50_000,
    seed=1,
    n_components=3
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Main
    ----
    Multiday Monte Carlo Value-at-Risk (VaR) using a multivariate Gaussian Mixture Model (GMM).

    Simulates asset paths over a specified time horizon using daily log-returns
    sampled from a fitted GMM. Only equity positions are supported.
    Returns monetary VaR, full P&L distribution, and simulated portfolio paths.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data of underlying assets (T × N).
    shares : array-like or pd.Series
        Number of shares held in each asset (must match asset dimension).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    days_ahead : int, optional
        Simulation horizon in trading days. Default is 10.
    simulations : int, optional
        Number of Monte Carlo simulations. Default is 50,000.
    seed : int, optional
        Random seed for reproducibility. Default is 1.
    n_components : int, optional
        Number of components in the GMM. Default is 3.

    Returns
    -------
    var : float
        Estimated Value-at-Risk (monetary units, positive).
    profit_and_loss : np.ndarray
        Simulated distribution of final portfolio profit and loss.
    portfolio_paths : np.ndarray
        Simulated portfolio value paths (shape: days + 1 × simulations).
    
    Notes
    -----
    - GMM captures non-Gaussian features of asset returns (e.g., skew, kurtosis).
    """
    shares = np.asarray(shares)
    if shares.ndim != 1 or shares.shape[0] != price_data.shape[1]:
        raise ValueError("Shape mismatch between shares and price_data.")

    np.random.seed(seed)
    alpha = 1 - confidence_level

    log_returns = np.log(price_data / price_data.shift(1)).dropna().values
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gmm.fit(log_returns)

    weights = gmm.weights_
    means = gmm.means_
    covs = gmm.covariances_

    S0 = price_data.iloc[-1].values
    n_assets = len(S0)
    asset_paths = np.zeros((days_ahead + 1, simulations, n_assets))
    asset_paths[0] = S0

    for t in range(1, days_ahead + 1):
        for i in range(simulations):
            k = np.random.choice(n_components, p=weights)
            shock = np.random.multivariate_normal(means[k], covs[k])
            asset_paths[t, i] = asset_paths[t - 1, i] * np.exp(shock)

    portfolio_paths = (asset_paths * shares).sum(axis=2)
    profit_and_loss = portfolio_paths[-1] - portfolio_paths[0]
    var = -np.percentile(profit_and_loss, alpha * 100)

    return var, profit_and_loss, portfolio_paths


# ----------------------------------------------------------
# Simulation Based ES (General)
# ----------------------------------------------------------
def simulation_es(
    var: float,
    profit_and_loss: np.ndarray,
    lambda_decay: float = None
) -> float:
    """
    Compute Expected Shortfall (ES) from a simulated P&L distribution.

    Parameters
    ----------
    var : float
        VaR threshold (positive, monetary units).
    profit_and_loss : np.ndarray
        Simulated profit-and-loss outcomes.
    lambda_decay : float or None
        If set, applies exponential weighting for WHS ES.

    Returns
    -------
    es : float
        Expected Shortfall (monetary units, positive).
    """
    profit_and_loss = np.asarray(profit_and_loss)
    mask = profit_and_loss <= -var

    if not np.any(mask):
        return 0.0

    tail_losses = profit_and_loss[mask]

    # Unweighted ES (historical average)
    if lambda_decay is None:
        return -tail_losses.mean()

    # Check decay value validity
    if not (0 < lambda_decay < 1):
        raise ValueError("lambda_decay must be between 0 and 1")

    n = len(profit_and_loss)
    raw_weights = lambda_decay ** np.arange(n - 1, -1, -1)
    weights = raw_weights / raw_weights.sum()

    tail_weights = weights[mask]
    tail_weights /= tail_weights.sum()

    es = -np.sum(tail_weights * tail_losses)
    return es
