"""
Options VaR Module
------------------

Delta-Normal Value-at-Risk (VaR) estimation utilities for portfolios of 
European-style listed equity options using the Black-Scholes framework.

This module provides analytical risk approximations based on delta exposure 
and static volatility assumptions. Correlations and volatilities are inferred 
from historical price data of the underlying assets.

The approach assumes constant volatility and fixed positions over a short holding 
period. It is best suited for short-term VaR estimation under normal market conditions.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
July 2025

Contents
--------
- black_scholes_pricing: Closed-form option pricing under BSM assumptions
- black_scholes_delta: Option delta from BSM formula
- single_option_var: Delta-Normal VaR for one option position
- options_portfolio_var: Delta-Normal VaR for a portfolio of options
"""


# --------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


# ----------------------------------------------------------
# Black-Scholes Pricing Function
# ----------------------------------------------------------
def black_scholes_pricing(S, K, tau, r, sigma, opt_type="call"):
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
    

# --------------------------------------------------------------------
# Black–Scholes delta 
# --------------------------------------------------------------------
def black_scholes_delta(
    option_type: str,
    spot_price: float,
    strike_price: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float) -> float:
    """
    Compute Black-Scholes delta for a European option.

    Parameters
    ----------
    option_type : str
        "call" or "put".
    spot_price : float
        Current price of the underlying asset.
    strike_price : float
        Option strike price.
    time_to_maturity : float
        Time to expiration in years.
    risk_free_rate : float
        Annualized risk-free rate.
    volatility : float
        Annualized volatility (decimal form).

    Returns
    -------
    delta : float
        Black-Scholes delta (sensitivity to spot changes).
    """
    d1 = (np.log(spot_price / strike_price)
          + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity) \
         / (volatility * np.sqrt(time_to_maturity))

    if option_type.lower() == "call":
        return norm.cdf(d1)
    if option_type.lower() == "put":
        return norm.cdf(d1) - 1
    raise ValueError("option_type must be 'call' or 'put'")


# --------------------------------------------------------------------
#  Single Option VaR (Monetary Only, Static)
# --------------------------------------------------------------------
def single_option_var(
    price_data: pd.Series,
    option_delta: float,
    quantity: int,
    contract_size: int,
    confidence_level: float = 0.99,
    holding_period: int = 1) -> float:
    """
    Compute Delta-Normal VaR (monetary) for a single option position.

    VaR is computed using the delta approximation and estimated 
    volatility from historical prices of the underlying.

    Parameters
    ----------
    price_data : pd.Series
        Historical price series of the underlying asset.
    option_delta : float
        Delta of the option (∂V/∂S).
    quantity : int
        Number of option contracts held.
    contract_size : int
        Number of shares per contract (e.g., 100 for equity options).
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.99.
    holding_period : int, optional
        Holding period in trading days. Default is 1.

    Returns
    -------
    var : float
        Estimated Value-at-Risk in monetary units (positive).
    """
    price_data = price_data.sort_index()
    returns_series = price_data.pct_change().dropna()
    price_data = price_data.loc[returns_series.index]

    sigma = returns_series.std(ddof=1)
    spot_price = price_data.iloc[-1]
    exposure = abs(option_delta * quantity * contract_size * spot_price)
    z = norm.ppf(confidence_level)

    var = z * exposure * sigma * np.sqrt(holding_period)
    return float(var)


# --------------------------------------------------------------------
# Options Portfolio VaR (Monetary Only, Static)
# --------------------------------------------------------------------
def options_portfolio_var(
    price_data: pd.DataFrame,
    deltas: pd.Series,
    contract_sizes: pd.Series,
    quantities: pd.Series,
    confidence_level: float = 0.99,
    holding_period: int = 1) -> float:
    """
    Compute Delta-Normal VaR (monetary) for a portfolio of equity options.

    Uses historical volatilities and correlations of the underlying assets,
    along with portfolio delta exposures, to estimate portfolio-level VaR.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical prices of underlying assets (T × N).
    deltas : pd.Series
        Option deltas (length N), one per asset.
    contract_sizes : pd.Series
        Contract size per option (shares per contract).
    quantities : pd.Series
        Number of contracts held per option.
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.99.
    holding_period : int, optional
        Holding period in trading days. Default is 1.

    Returns
    -------
    var : float
        Estimated portfolio VaR in monetary units (positive).

    Raises
    ------
    ValueError
        If the inputs are misaligned with the columns of the price data.
    """
    price_data = price_data.sort_index()
    returns_frame = price_data.pct_change().dropna()
    price_data = price_data.loc[returns_frame.index]

    if not all(s.index.equals(price_data.columns) for s in [deltas, contract_sizes, quantities]):
        raise ValueError("All parameter series must align with asset columns.")

    sigma_vec = returns_frame.std(ddof=1).values
    corr_matrix = returns_frame.corr().values

    multipliers = (deltas * contract_sizes * quantities).values
    spot_vec = price_data.iloc[-1].values
    x_vector = multipliers * spot_vec

    sigma_matrix = np.diag(sigma_vec) @ corr_matrix @ np.diag(sigma_vec)
    portfolio_var = x_vector @ sigma_matrix @ x_vector
    portfolio_sigma = np.sqrt(portfolio_var)

    z = norm.ppf(confidence_level)
    var = z * portfolio_sigma * np.sqrt(holding_period)
    return float(var)
