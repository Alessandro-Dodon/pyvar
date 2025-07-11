"""
Factor Model VaR and Expected Shortfall Module
----------------------------------------------

Provides modular functions to compute portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
based on linear factor models. Supports both the Sharpe single-index model and the 
Fama-French 3-factor framework. The quantiles are obtained from the normal distribution, as
factor returns are assumed to be normally distributed. ES is estimated using the general 
parametric normal formula based on portfolio volatility.

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
- single_factor_var: Sharpe model — estimates VaR and portfolio volatility
- fama_french_var: Fama-French 3-factor model — estimates VaR and volatility
- factor_models_es: Computes ES from volatility and infers portfolio value from VaR
"""


#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
from .utils import fit_garch_model, load_ff3_factors


# ----------------------------------------------------------
# Single-Factor VaR (Sharpe Model)
# ----------------------------------------------------------
def single_factor_var(
    returns,
    benchmark,
    weights,
    portfolio_value,
    confidence_level=0.99,
    volatility_model="static",
    p=None,
    q=None,
    distribution_volatility=None):
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using a single-factor (Sharpe) model.

    Supports static or dynamic (GARCH-family) volatility modeling of the market factor.
    Computes portfolio VaR assuming all assets share exposure to a single systematic risk factor.
    Estimates asset betas, residual variances, and portfolio volatility under the Sharpe single-index model.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series (columns = tickers, index = dates).
    benchmark : pd.Series
        Market return series (systematic factor).
    weights : pd.Series
        Portfolio weights. Must align with return columns and sum to 1.
    portfolio_value : float
        Current value of the portfolio in monetary units.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    volatility_model : str, optional
        Volatility specification: 'STATIC', 'GARCH', 'EGARCH', etc.
    p, q : int, optional
        GARCH model parameters. Ignored if volatility_model = 'STATIC'.
    distribution_volatility : str, optional
        Distribution for GARCH residuals ('normal', 't', etc.).

    Returns
    -------
    result_data : pd.DataFrame
        With:
        - 'Returns': daily portfolio return
        - 'VaR': Value-at-Risk (decimal)
        - 'VaR Violation': indicator of breaches
        - 'VaR_monetary': VaR scaled by portfolio value

    portfolio_volatility : pd.Series
        Estimated portfolio volatility time series (daily).

    Raises
    ------
    ValueError
        For misaligned data or invalid weights.

    Notes
    -----
    - Residuals are assumed uncorrelated across assets.
    - Factor returns are assumed normally distributed.
    """
    volatility_model = volatility_model.upper()

    if not returns.index.equals(benchmark.index):
        raise ValueError("Benchmark and asset returns must share the same index.")
    if set(returns.columns) != set(weights.index):
        raise ValueError("Weights must match returns columns.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    weights = weights[returns.columns]

    betas = []
    residuals = pd.DataFrame(index=returns.index)
    for ticker in returns.columns:
        slope, _ = np.polyfit(benchmark, returns[ticker], 1)
        betas.append(slope)
        residuals[ticker] = returns[ticker] - slope * benchmark
    betas = np.array(betas)
    residual_cov = np.diag(residuals.var(ddof=1).values)

    if volatility_model == "STATIC":
        if any(param is not None for param in [p, q, distribution_volatility]):
            print("Static volatility selected — p, q, and distribution_volatility are ignored.")
        sigma = benchmark.std(ddof=1)
        factor_volatility_series = pd.Series(sigma, index=benchmark.index)
    else:
        _, cond_vol, _, _ = fit_garch_model(
            returns=benchmark,
            p=p or 1,
            q=q or 1,
            model=volatility_model,
            distribution=distribution_volatility or "normal"
        )
        factor_volatility_series = pd.Series(cond_vol, index=benchmark.index)

    beta_portfolio = weights.values @ betas
    residual_risk = weights.values @ residual_cov @ weights.values
    portfolio_volatility = np.sqrt(
        (beta_portfolio**2) * factor_volatility_series**2 + residual_risk
    )

    z = norm.ppf(confidence_level)
    var_series = z * portfolio_volatility
    portfolio_returns = returns @ weights

    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "VaR": var_series,
        "VaR Violation": portfolio_returns < -var_series,
        "VaR_monetary": var_series * portfolio_value
    })

    return result_data, portfolio_volatility


# -------------------------------------------------------
# Fama-French 3-Factor Model: Value-at-Risk
# -------------------------------------------------------
def fama_french_var(
    returns,
    weights,
    portfolio_value,
    confidence_level=0.99,
    factors=None,
    volatility_model="static",
    p=None,
    q=None,
    distribution_volatility=None):
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using the Fama-French 3-factor model.

    Supports static or dynamic (GARCH-family) volatility modeling for factor returns.
    Fits a linear factor model to estimate asset betas and residual risk. 
    Uses factor covariance and idiosyncratic variance to compute portfolio volatility and VaR.

    If no factor data is provided, Fama-French daily factors are automatically downloaded.
    Custom factor models can also be used by passing a DataFrame with the same structure.

        Parameters
    ----------
    returns : pd.DataFrame
        Asset return series (columns = tickers, index = dates).
    weights : pd.Series
        Portfolio weights. Must align with return columns and sum to 1.
    portfolio_value : float
        Current value of the portfolio in monetary units.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    factors : pd.DataFrame or None, optional
        Daily Fama-French factors. If None, data is downloaded.
    volatility_model : str, optional
        'STATIC', 'GARCH', 'EGARCH', etc.
    p, q : int, optional
        GARCH model order. Ignored if volatility_model = 'STATIC'.
    distribution_volatility : str, optional
        Distribution for GARCH innovations.

    Returns
    -------
    result_data : pd.DataFrame
        With:
        - 'Returns': portfolio return
        - 'Factor_Mkt_RF', 'Factor_SMB', 'Factor_HML': factor exposures
        - 'VaR': Value-at-Risk (decimal)
        - 'VaR Violation': indicator of breaches
        - 'VaR_monetary': VaR scaled by portfolio value

    portfolio_volatility : pd.Series
        Estimated portfolio volatility (daily).

    Raises
    ------
    ValueError
        If data is misaligned or weights invalid.

    Notes
    -----
    - Assumes factor returns are normally distributed.
    - Estimates 1-day VaR from daily factor data.
    """
    volatility_model = volatility_model.upper()

    if returns.isnull().values.any():
        raise ValueError("Missing values in asset returns.")
    if not weights.index.equals(returns.columns):
        raise ValueError("Weights must match return columns and order.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Weights must sum to 1.")

    if factors is None:
        factors = load_ff3_factors(
            start=returns.index.min(),
            end=returns.index.max()
        )
    factors = factors.reindex(returns.index).ffill()

    X = sm.add_constant(factors[["Mkt_RF", "SMB", "HML"]])
    excess = returns.sub(factors["RF"], axis=0)

    betas = {}
    resid_var = {}
    for ticker in returns.columns:
        yx = pd.concat([excess[ticker], X], axis=1).dropna()
        model = sm.OLS(yx.iloc[:, 0], yx.iloc[:, 1:]).fit()
        betas[ticker] = model.params.drop("const")
        resid_var[ticker] = model.resid.var(ddof=0)

    B = pd.DataFrame(betas).T.values
    D = np.diag(pd.Series(resid_var).values)

    if volatility_model == "STATIC":
        if any(param is not None for param in [p, q, distribution_volatility]):
            print("Static volatility selected — p, q, and distribution_volatility are ignored.")
        factor_cov = factors[["Mkt_RF", "SMB", "HML"]].cov().values
        sigma_series = None
    else:
        sigma_series = {}
        for factor in ["Mkt_RF", "SMB", "HML"]:
            _, cond_vol, _, _ = fit_garch_model(
                returns=factors[factor],
                p=p or 1,
                q=q or 1,
                model=volatility_model,
                distribution=distribution_volatility or "normal"
            )
            sigma_series[factor] = pd.Series(cond_vol, index=factors.index)

    portfolio_volatility = []
    for t in returns.index:
        if sigma_series is not None:
            sigma_diag = np.diag([
                sigma_series["Mkt_RF"].get(t, np.nan)**2,
                sigma_series["SMB"].get(t, np.nan)**2,
                sigma_series["HML"].get(t, np.nan)**2
            ])
        else:
            sigma_diag = factor_cov

        cov_matrix = B @ sigma_diag @ B.T + D
        sigma_p = np.sqrt(weights.values @ cov_matrix @ weights.values)
        portfolio_volatility.append(sigma_p)

    portfolio_volatility = pd.Series(portfolio_volatility, index=returns.index)
    z = norm.ppf(confidence_level)
    var_series = z * portfolio_volatility
    portfolio_returns = returns @ weights

    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "Factor_Mkt_RF": factors["Mkt_RF"],
        "Factor_SMB": factors["SMB"],
        "Factor_HML": factors["HML"],
        "VaR": var_series,
        "VaR Violation": portfolio_returns < -var_series,
        "VaR_monetary": var_series * portfolio_value
    })

    return result_data, portfolio_volatility


# -------------------------------------------------------
# Factor Model ES (General)
# -------------------------------------------------------
def factor_models_es(
    result_data,
    portfolio_volatility,
    confidence_level = 0.99) :
    """
    Main
    ----
    Append Expected Shortfall (ES) to a factor-model-based VaR result.

    Computes ES from portfolio volatility assuming normality. Works with both single- and multi-factor models. Infers the portfolio value from the 'VaR' and 'VaR_monetary' ratio.

    Parameters
    ----------
    result_data : pd.DataFrame
        Must include 'VaR' and 'VaR_monetary' columns.
    portfolio_volatility : float
        Portfolio volatility (daily, decimal).
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.

    Returns
    -------
    pd.DataFrame
        Updated with:
        - 'ES': Expected Shortfall (decimal)
        - 'ES_monetary': Expected Shortfall in monetary units

    Raises
    ------
    ValueError
        If required columns are missing or invalid.

    Notes
    -----
    - Assumes normal distribution of returns.
    """
    if "VaR" not in result_data.columns or "VaR_monetary" not in result_data.columns:
        raise ValueError("Missing 'VaR' or 'VaR_monetary' columns in result_data.")
    
    valid_rows = result_data["VaR"] > 0
    if not valid_rows.any():
        raise ValueError("Invalid 'VaR' values — cannot infer portfolio value.")

    inferred_portfolio_value = (
        result_data.loc[valid_rows, "VaR_monetary"].iloc[0] /
        result_data.loc[valid_rows, "VaR"].iloc[0]
    )

    z = norm.ppf(confidence_level)
    tail_probability = 1 - confidence_level
    es_pct = portfolio_volatility * norm.pdf(z) / tail_probability

    result_data = result_data.copy()
    result_data["ES"] = pd.Series(es_pct, index=result_data.index)
    result_data["ES_monetary"] = result_data["ES"] * inferred_portfolio_value

    return result_data












