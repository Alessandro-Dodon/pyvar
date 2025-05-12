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

#########################################################
# Note: weights are constant? Check logic and compare it
#       with the other models that use x
#       it seems to me that the weights should be fixed
#########################################################
# Note2: ban short positions?
#########################################################
# Note 3: add a function on data download for those added
#         inputs? (like portfolio value or weigts)
#########################################################

# -------------------------------------------------------
# Single-Factor (Sharpe) — Portfolio VaR and ES 
# -------------------------------------------------------
def sharpe_model(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99
) -> tuple[pd.DataFrame, float, float]:
    """
    Computes portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using a single-factor (Sharpe) model. 
    Parameters:
    - returns : pd.DataFrame
        Asset return time series (columns = tickers).
    - benchmark : pd.Series
        Market return series (same index as returns).
    - weights : pd.Series
        Portfolio weights (must sum to 1).
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
    - var : float
        Scalar VaR in monetary units.
    - es : float
        Scalar ES in monetary units.

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

    # Portfolio risk
    port_vol = np.sqrt(weights.values @ Sigma.values @ weights.values)
    z = norm.ppf(confidence_level)
    tail_prob = 1 - confidence_level

    # Final scalar VaR/ES
    var_pct = z * port_vol
    es_pct = (port_vol * norm.pdf(z) / tail_prob)

    var = var_pct * portfolio_value
    es = es_pct * portfolio_value

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

    return result_df, var, es


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
# Fama-French 3-Factor Model — Portfolio VaR and ES 
# -------------------------------------------------------
def fama_french_model(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.99,
    factors: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, float, float]:
    """
    Computes portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using the Fama–French 3-factor model.

    Parameters:
    - returns : pd.DataFrame
        Asset return time series (columns = tickers).
    - weights : pd.Series
        Portfolio weights (must sum to 1).
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
    - var : float
        Scalar VaR in monetary units.
    - es : float
        Scalar ES in monetary units.
    """
    if returns.isnull().values.any():
        raise ValueError("Missing values detected in returns. Handle NaNs before passing.")

    if factors is None:
        factors = load_ff3_factors(start=returns.index[0])
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

    port_vol = np.sqrt(weights.values @ Σ @ weights.values)
    z = norm.ppf(confidence_level)
    tail_prob = 1 - confidence_level

    var_pct = z * port_vol
    es_pct = port_vol * norm.pdf(z) / tail_prob

    var = var_pct * portfolio_value
    es = es_pct * portfolio_value

    portf_returns = returns @ weights
    result_df = pd.DataFrame({
        "Returns": portf_returns,
        "VaR": pd.Series(var_pct, index=portf_returns.index),
        "ES": pd.Series(es_pct, index=portf_returns.index),
    })
    result_df["VaR Violation"] = result_df["Returns"] < -result_df["VaR"]
    result_df["VaR_monetary"] = result_df["VaR"] * portfolio_value
    result_df["ES_monetary"] = result_df["ES"] * portfolio_value

    return result_df.dropna(), var, es
