"""
Factor Model VaR and Expected Shortfall Module
----------------------------------------------

Provides modular functions to compute portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
based on linear factor models. Supports both the Sharpe single-index model and the 
Fama–French 3-factor framework. ES is estimated using a general parametric Gaussian formula 
based on portfolio volatility.

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

Notes
-----
- All returns must be daily and in decimal format (e.g., 0.01 = 1%).
- Portfolio weights must sum to 1 and match return columns.
- Warnings are issued if extreme short exposures are detected (e.g., weights < –1).
- `factor_models_es` assumes normally distributed returns and infers portfolio value 
  from the ratio of 'VaR_monetary' to 'VaR' unless passed separately.
"""

# NOTE: weights are constant? Check logic and compare it
#       with the other models that use x
#       it seems to me that the weights should be fixed

# NOTE 2: ban short positions?

# NOTE 3: add a function on data download for those added
#         inputs? (like portfolio value or weigts)

# NOTE 4: why not loading the 3 factors silently and directly?

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
    Estimate Value-at-Risk (VaR) using a single-factor (Sharpe) model.

    Computes portfolio VaR assuming all assets share exposure to a single systematic risk factor.
    The method estimates factor betas, residual volatility, and computes portfolio volatility
    under the Sharpe single-index model assumption.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series (columns = tickers, index = dates).
    benchmark : pd.Series
        Market return series (e.g., an index).
    weights : pd.Series
        Portfolio weights per asset. Must sum to 1 and align with `returns` columns.
    portfolio_value : float
        Total current value of the portfolio in monetary units.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with columns:
        - 'Returns': portfolio return series (decimal)
        - 'VaR': constant VaR threshold (decimal loss)
        - 'VaR Violation': boolean flag for VaR breaches
        - 'VaR_monetary': VaR in monetary units

    portfolio_volatility : float
        Portfolio standard deviation estimated under the Sharpe model.

    Raises
    ------
    ValueError
        If the benchmark index does not align with the returns index.
        If weights do not match the number or names of tickers in `returns`.
        If weights do not sum to 1.
    Warning
        If weights include extreme short exposures (e.g., weight < -1), results may be unstable
        or require careful interpretation.
    """
    if not returns.index.equals(benchmark.index):
        raise ValueError("Benchmark and asset return series must have the same datetime index.")

    if not set(returns.columns) == set(weights.index):
        raise ValueError("Weights must match the columns in the return DataFrame (tickers).")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    weights = weights[returns.columns]  # align order

    if weights.min() < -1:
        print("[warning] Some weights indicate extreme short positions (e.g., weight < -100%).")
        print("          Ensure this reflects intentional portfolio structure.")

    # Estimate betas and residuals
    betas = []
    residuals = pd.DataFrame(index=returns.index)
    for ticker in returns.columns:
        r_i = returns[ticker]
        reg = np.polyfit(benchmark, r_i, 1)
        beta = reg[0]
        betas.append(beta)
        predicted = beta * benchmark
        residuals[ticker] = r_i - predicted

    betas = np.array(betas)
    factor_var = np.var(benchmark, ddof=1)
    residual_cov = residuals.cov().values

    portfolio_volatility = np.sqrt(
        (weights @ betas) ** 2 * factor_var + weights @ residual_cov @ weights
    )

    z = norm.ppf(confidence_level)
    var_pct = z * portfolio_volatility

    portf_returns = returns @ weights
    result_data = pd.DataFrame({
        "Returns": portf_returns,
        "Benchmark": benchmark,
        "VaR": pd.Series(var_pct, index=portf_returns.index),
    })
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
    Estimate Value-at-Risk (VaR) using the Fama-French 3-factor model.

    Fits a linear factor model with Mkt-RF, SMB, and HML factors. Computes asset-level 
    factor betas and residual variance, builds the portfolio risk from factor and idiosyncratic 
    covariances, and estimates parametric VaR.

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

    portf_returns = returns @ weights
    result_data = pd.DataFrame({
        "Returns": portf_returns,
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
    Append Expected Shortfall (ES) to a factor-model-based VaR result DataFrame.

    Infers portfolio value from the ratio of 'VaR_monetary' to 'VaR' and appends
    both decimal and monetary ES columns to the input result DataFrame.

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
    tail_prob = 1 - confidence_level
    es_pct = portfolio_volatility * norm.pdf(z) / tail_prob

    result_data = result_data.copy()
    result_data["ES"] = pd.Series(es_pct, index=result_data.index)
    result_data["ES_monetary"] = result_data["ES"] * inferred_portfolio_value

    return result_data




### Old functions (double check later)

# -------------------------------------------------------
# Single-Factor (Sharpe) — Portfolio VaR and ES 
# -------------------------------------------------------
def sharpe_modelOLD(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99
) -> pd.DataFrame:
    """
    Computes portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using a single-factor (Sharpe) model. 
    Parameters:
    - returns : pd.DataFrame
        Asset return time series (columns = tickers).
    - benchmark : pd.Series
        Market return series (same index as returns).
    - weights : pd.Series
        Portfolio weights (must sum to 1). Large negative values (e.g., < -1) trigger a warning.
    - portfolio_value : float
        Total current value of the portfolio.
    - confidence_level : float
        VaR/ES confidence level (e.g., 0.99).

    Returns:
    - result_df : pd.DataFrame
        Contains columns:
        - 'Returns': portfolio return series (decimal)
        - 'VaR': constant VaR threshold (decimal loss)
        - 'ES': constant ES threshold (decimal loss)
        - 'VaR Violation': boolean flag per day
        - 'VaR_monetary': VaR in monetary units
        - 'ES_monetary': ES in monetary units

    Raises:
    - ValueError: If index alignment between returns and benchmark fails.
    """
    # Check index alignment
    if not returns.index.equals(benchmark.index):
        raise ValueError("Index mismatch: 'returns' and 'benchmark' must have identical datetime index.")
    if returns.isnull().values.any() or benchmark.isnull().any():
        raise ValueError("Missing values detected. Please drop or fill NaNs before passing data.")

    # Estimate Sharpe model components
    market_var = benchmark.var(ddof=0)
    cov_with_benchmark = returns.apply(lambda x: x.cov(benchmark))
    betas = cov_with_benchmark / market_var
    idiosyncratic_var = returns.var(ddof=0) - betas.pow(2) * market_var

    tickers = returns.columns
    factor_cov = np.outer(betas, betas) * market_var
    Sigma = pd.DataFrame(factor_cov, index=tickers, columns=tickers)
    for t in tickers:
        Sigma.at[t, t] += idiosyncratic_var[t]

    # Check for extreme short exposures
    min_weight = weights.min()
    if min_weight < -1:
        print("[warning] Some weights indicate extreme short positions (e.g., weight < -100%).")
        print("          Ensure these reflect intentional portfolio structure.")

    # Portfolio risk
    port_vol = np.sqrt(weights.values @ Sigma.values @ weights.values)
    z = norm.ppf(confidence_level)
    tail_prob = 1 - confidence_level

    # Final scalar VaR/ES
    var_pct = z * port_vol
    es_pct = (port_vol * norm.pdf(z) / tail_prob)

    # Create backtestable result DataFrame
    portf_returns = returns @ weights
    result_df = pd.DataFrame({
        "Returns": portf_returns,
        "VaR": pd.Series(var_pct, index=portf_returns.index),
        "ES": pd.Series(es_pct, index=portf_returns.index),
    })
    result_df["VaR Violation"] = result_df["Returns"] < -result_df["VaR"]
    result_df["VaR_monetary"] = result_df["VaR"] * portfolio_value
    result_df["ES_monetary"] = result_df["ES"] * portfolio_value

    return result_df


# -------------------------------------------------------
# Fama-French 3-Factor Model — Factor Loader
# -------------------------------------------------------
_FF_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

def load_ff3_factorsOLD(start=None, end=None) -> pd.DataFrame:
    """
    Downloads Fama-French 3-factor daily data.
    Returns DataFrame with ['Mkt_RF', 'SMB', 'HML', 'RF'] as fractional returns.
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
# Fama-French 3-Factor Model — Portfolio VaR and ES 
# -------------------------------------------------------
def fama_french_modelOLD(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
    factors: pd.DataFrame | None = None # Why this? Double check if there are more versions!
) -> pd.DataFrame:
    """
    Computes portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using the Fama–French 3-factor model.

    Parameters:
    - returns : pd.DataFrame
        Asset return time series (columns = tickers).
    - weights : pd.Series
        Portfolio weights (must sum to 1). Must align with returns columns.
    - portfolio_value : float
        Total value of the portfolio.
    - confidence_level : float
        VaR/ES confidence level (e.g., 0.99).
    - factors : pd.DataFrame or None
        Optional preloaded Fama-French factors. If None, data is auto-downloaded.

    Returns:
    - result_df : pd.DataFrame
        Time series with columns:
        - 'Returns', 'VaR', 'ES', 'VaR Violation', 'VaR_monetary', 'ES_monetary'
    
    Raises:
    Large negative weight values (e.g., < -1) trigger a warning.
    Raises ValueError if misaligned or if weights do not sum to 1.
    """
    if returns.isnull().values.any():
        raise ValueError("Missing values detected in returns. Handle NaNs before passing.")

    # Input checks
    if not weights.index.equals(returns.columns):
        raise ValueError("weights and returns must have identical tickers in the same order.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("weights must sum to 1.")

    if factors is None:
        factors = load_ff3_factors(
            start=returns.index.min(),
            end=returns.index.max()
        )

    # Align to return dates with forward fill in case of market holidays
    factors = factors.reindex(returns.index).ffill()

    # Build regression matrix and excess returns
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

    # Check for extreme short exposures
    min_weight = weights.min()
    if min_weight < -1:
        print("[warning] Some weights indicate extreme short positions (e.g., weight < -100%).")
        print("          Ensure these reflect intentional portfolio structure.")

    port_vol = np.sqrt(weights.values @ Σ @ weights.values)
    z = norm.ppf(confidence_level)
    tail_prob = 1 - confidence_level

    var_pct = z * port_vol
    es_pct = port_vol * norm.pdf(z) / tail_prob

    portf_returns = returns @ weights
    result_df = pd.DataFrame({
        "Returns": portf_returns,
        "VaR": pd.Series(var_pct, index=portf_returns.index),
        "ES": pd.Series(es_pct, index=portf_returns.index),
    })
    result_df["VaR Violation"] = result_df["Returns"] < -result_df["VaR"]
    result_df["VaR_monetary"] = result_df["VaR"] * portfolio_value
    result_df["ES_monetary"] = result_df["ES"] * portfolio_value

    return result_df.dropna()




