"""
Factor Model VaR and Expected Shortfall Module
----------------------------------------------

Provides modular functions to compute portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
based on linear factor models. Supports both the Sharpe single-index model and the 
Fama–French 3-factor framework. The quantiles are obtained from the normal distribution, as
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
- fama_french_var: Fama–French 3-factor model — estimates VaR and volatility
- factor_models_es: Computes ES from volatility and infers portfolio value from VaR
"""

#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from io import BytesIO
from zipfile import ZipFile
import requests


# ----------------------------------------------------------
# Single-Factor VaR (Sharpe Model)
# ----------------------------------------------------------
def single_factor_var(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99
) -> tuple[pd.DataFrame, float]:
    """
    Estimate 1-day Value-at-Risk (VaR) using a true single-factor (Sharpe) model,
    keeping only the diagonal residual variances.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return series (columns = tickers, index = dates).
    benchmark : pd.Series
        Market return series (e.g., SPY).
    weights : pd.Series
        Portfolio weights, must sum to 1 and align with returns.columns.
    portfolio_value : float
        Total portfolio value in monetary units.
    confidence_level : float
        Confidence level for VaR (default 0.99).

    Returns
    -------
    result_data : pd.DataFrame
        Columns:
        - Returns: portfolio returns series
        - VaR: constant percentile VaR (decimal loss)
        - VaR Violation: boolean flag of breaches
        - VaR_monetary: VaR in monetary units
    portfolio_volatility : float
        Estimated portfolio volatility under the single-factor model.
    """

    # 1) Validation checks
    if not returns.index.equals(benchmark.index):
        raise ValueError("Benchmark and asset returns must share the same index.")
    if set(returns.columns) != set(weights.index):
        raise ValueError("Weights must match returns columns.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    weights = weights[returns.columns]

    # 2) Estimate betas and compute residuals
    betas = []
    residuals = pd.DataFrame(index=returns.index)
    for ticker in returns.columns:
        r_i = returns[ticker]
        beta, _ = np.polyfit(benchmark, r_i, 1)
        betas.append(beta)
        residuals[ticker] = r_i - beta * benchmark
    betas = np.array(betas)

    # 3) Factor variance
    factor_variance = np.var(benchmark, ddof=1)

    # 4) Build diagonal residual covariance (only idiosyncratic variances)
    var_resid = residuals.var(ddof=1).values
    residual_cov = np.diag(var_resid)

    # 5) Portfolio volatility under single-factor approximation
    portfolio_volatility = np.sqrt(
        (weights.values @ betas)**2 * factor_variance
        + weights.values @ residual_cov @ weights.values
    )

    # 6) Compute VaR percentile
    z = norm.ppf(confidence_level)
    var_pct = z * portfolio_volatility

    # 7) Build result DataFrame
    portfolio_returns = returns @ weights
    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "VaR": var_pct,
    }, index=returns.index)
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]
    result_data["VaR_monetary"] = result_data["VaR"] * portfolio_value

    return result_data, portfolio_volatility


# -------------------------------------------------------
# Fama-French 3-Factor Model — Factor Loader
# -------------------------------------------------------
_FF_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

def load_ff3_factors(start=None, end=None) -> pd.DataFrame:
    """
    Downloads Fama-French 3-factor daily data.
    Returns DataFrame with ['Mkt_RF', 'SMB', 'HML', 'RF'] as fractional returns.
    This is automatically called by the `fama_french_var` function if no factors are provided.
    """
    resp = requests.get(_FF_ZIP_URL, timeout=30)
    resp.raise_for_status()
    zf = ZipFile(BytesIO(resp.content))
    csvf = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    ff = pd.read_csv(zf.open(csvf), skiprows=3, index_col=0)

    mask = ff.index.astype(str).str.match(r"^\d{8}$")
    ff = ff.loc[mask].astype(float) / 100.0
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    ff.columns = ["Mkt_RF", "SMB", "HML", "RF"]

    if start: ff = ff.loc[start:]
    if end:   ff = ff.loc[:end]
    return ff.sort_index()


# -------------------------------------------------------
# Fama-French 3-Factor Model: Value-at-Risk
# -------------------------------------------------------
def fama_french_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
    factors: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, float]:
    """
    Main
    ----
    Estimate Value-at-Risk (VaR) using the Fama-French 3-factor model.

    Fits a linear factor model with Mkt-RF, SMB, and HML factors. Computes asset-level 
    factor betas and residual variance, builds the portfolio risk from factor and idiosyncratic 
    covariances, and estimates parametric VaR. The portfolio volatility is needed for the
    expected shortfall (ES) calculation.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series (columns = tickers, index = dates).
    weights : pd.Series
        Portfolio weights. Must align with return columns and sum to 1.
    portfolio_value : float
        Total current value of the portfolio.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    factors : pd.DataFrame or None
        Optional preloaded Fama-French factors. If None, data is auto-downloaded.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with:
        - 'Returns': daily portfolio returns
        - 'Factor_Returns': Mkt_RF + SMB + HML factors
    portfolio_volatility : float
        Portfolio standard deviation estimated using the FF3 factor structure.

    Raises
    ------
    ValueError
        If weights misalign or do not sum to 1.
        If benchmark data is missing or unmatched.
    Warning
        Printed if short positions are very large (e.g., weight < -1).

    Notes
    -----
    - This VaR assumes factor returns are normally distributed.
    - This function estimates 1-day VaR, as we download the daily factors data.
    - Weights are supposed to be perfectly constant during the period.
    """
    if returns.isnull().values.any():
        raise ValueError("Missing values detected in returns. Handle NaNs before passing.")

    if not weights.index.equals(returns.columns):
        raise ValueError("weights and returns must have identical tickers in the same order.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    if factors is None:
        factors = load_ff3_factors(
            start=returns.index.min(),
            end=returns.index.max()
        )

    factors = factors.reindex(returns.index).ffill()

    X = sm.add_constant(factors[["Mkt_RF", "SMB", "HML"]])
    excess = returns.sub(factors["RF"], axis=0)

    betas, resid_var = {}, {}
    for tkr in returns:
        yx = pd.concat([excess[tkr], X], axis=1).dropna()
        res = sm.OLS(yx.iloc[:, 0], yx.iloc[:, 1:]).fit()
        betas[tkr] = res.params.drop("const")
        resid_var[tkr] = res.resid.var(ddof=0)

    B = pd.DataFrame(betas).T
    Σf = factors[["Mkt_RF", "SMB", "HML"]].cov().values
    Σ = B.values @ Σf @ B.values.T + np.diag(pd.Series(resid_var).values)

    if weights.min() < -1:
        print("[warning] Some weights indicate extreme short positions (e.g., weight < -100%).")
        print("          Ensure this reflects intentional portfolio structure.")

    portfolio_volatility = np.sqrt(weights.values @ Σ @ weights.values)
    z = norm.ppf(confidence_level)
    var_pct = z * portfolio_volatility

    portfolio_returns = returns @ weights
    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "Factor_Mkt_RF": factors["Mkt_RF"],
        "Factor_SMB": factors["SMB"],
        "Factor_HML": factors["HML"]
    }, index=returns.index)

    result_data["VaR"] = var_pct
    result_data["VaR Violation"] = result_data["Returns"] < -result_data["VaR"]
    result_data["VaR_monetary"] = result_data["VaR"] * portfolio_value

    return result_data, portfolio_volatility


# -------------------------------------------------------
# Factor Model ES — Inferred Portfolio Value
# -------------------------------------------------------
def factor_models_es(
    result_data: pd.DataFrame,
    portfolio_volatility: float,
    confidence_level: float = 0.99
) -> pd.DataFrame:
    """
    Main
    ----
    Append Expected Shortfall (ES) to a factor-model-based VaR result DataFrame.

    Infers portfolio value from the ratio of 'VaR_monetary' to 'VaR' and appends
    both decimal and monetary ES columns to the input result DataFrame.
    This can be used with both the one or three factor models, as long as
    the correct dataframe and portfolio volatility are passed.

    Parameters
    ----------
    result_data : pd.DataFrame
        Must contain 'VaR' and 'VaR_monetary' columns (non-zero, constant).
    portfolio_volatility : float
        Portfolio standard deviation (daily, decimal) from a factor model.
    confidence_level : float, optional
        Confidence level for ES (e.g., 0.99). Default is 0.99.

    Returns
    -------
    pd.DataFrame
        Updated result_data with:
        - 'ES': Expected Shortfall in decimal loss
        - 'ES_monetary': Expected Shortfall in currency units

    Raises
    ------
    ValueError
        If 'VaR' or 'VaR_monetary' is missing or contains zeros.

    Notes
    -----
    - This ES assumes factor returns are normally distributed.
    - This function estimates 1-day ES.
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












