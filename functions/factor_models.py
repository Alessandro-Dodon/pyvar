import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from io import BytesIO
from zipfile import ZipFile
import requests

# -------------------------------------------------------
# Single-Factor (Sharpe) VaR and ES estimation
# -------------------------------------------------------
def single_factor_var_es(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    port_val: float,
    confidence_level: float = 0.99
) -> tuple[float, float, pd.DataFrame, pd.Series, pd.Series]:
    """
    Computes portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using a single-factor (Sharpe) model.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series (columns = tickers).
    benchmark : pd.Series
        Market return series (e.g., index).
    weights : pd.Series
        Portfolio weights (must sum to 1).
    port_val : float
        Total current value of the portfolio.
    confidence_level : float
        Confidence level (default = 0.99).

    Returns
    -------
    var : float
        Value-at-Risk at given confidence level.
    es : float
        Expected Shortfall at given confidence level.
    Sigma : pd.DataFrame
        Covariance matrix estimated via single-factor model.
    betas : pd.Series
        Asset betas relative to the benchmark.
    idiosyncratic_var : pd.Series
        Asset-specific variance (residual).
    """
    market_var = benchmark.var(ddof=0)
    cov_with_benchmark = returns.apply(lambda x: x.cov(benchmark))
    betas = cov_with_benchmark / market_var
    idiosyncratic_var = returns.var(ddof=0) - betas.pow(2) * market_var

    tickers = returns.columns
    factor_cov = np.outer(betas, betas) * market_var
    Sigma = pd.DataFrame(factor_cov, index=tickers, columns=tickers)
    for t in tickers:
        Sigma.at[t, t] += idiosyncratic_var[t]

    port_vol = np.sqrt(weights.values @ Sigma.values @ weights.values)
    z = norm.ppf(confidence_level)
    var = z * port_vol * port_val
    tail_prob = 1 - confidence_level
    es = (port_vol * norm.pdf(norm.ppf(tail_prob)) / tail_prob) * port_val

    return var, es, Sigma, betas, idiosyncratic_var


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
# Fama-French 3-Factor Model — Portfolio VaR and CVaR
# -------------------------------------------------------
def ff3_var_cvar(
    *,
    returns: pd.DataFrame | None = None,
    prices : pd.DataFrame | None = None,
    weights: pd.Series    | None = None,
    shares : pd.Series    | None = None,
    alpha: float = 0.95,
    factors: pd.DataFrame | None = None,
) -> tuple[float, float, pd.DataFrame, pd.Series, np.ndarray]:
    """
    Computes portfolio VaR and CVaR using the Fama–French 3-factor model,
    and returns all internal components for inspection.

    Returns:
    - VaR (float)
    - CVaR (float)
    - betas (pd.DataFrame)
    - idiosyncratic variances (pd.Series)
    - full covariance matrix Σ (np.ndarray)
    """
    if returns is None and prices is None:
        raise ValueError("must pass `returns` or `prices`")

    if prices is not None and returns is None:
        returns = prices.pct_change().dropna()

    if weights is None and shares is None:
        raise ValueError("must pass `weights` or `shares`")
    if shares is not None:
        latest_price = prices.iloc[-1] if prices is not None else None
        if latest_price is None:
            raise ValueError("to use `shares`, `prices` must be provided")
        port_val = (latest_price * shares).sum()
        weights = (latest_price * shares) / port_val
    else:
        port_val = 1.0
        weights = weights / weights.sum()

    if factors is None:
        factors = load_ff3_factors(start=returns.index[0])
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

    σ_p = np.sqrt(weights.values @ Σ @ weights.values)

    z = norm.ppf(alpha)
    VaR = z * σ_p * port_val
    tail = 1 - alpha
    CVaR = σ_p * norm.pdf(norm.ppf(tail)) / tail * port_val

    return VaR, CVaR, B, pd.Series(resid_var), Σ


'''
# Example usage:
import pandas as pd
from ff3_var import ff3_var_cvar

prices = pd.read_csv("prices_eur.csv", index_col=0, parse_dates=True)
shares = pd.Series({"MSFT": 4, "NVDA": 3})

var, cvar = ff3_var_cvar(prices=prices,
                         shares=shares,
                         alpha=0.95)

print(f"VaR95  : {var:,.2f}")
print(f"CVaR95 : {cvar:,.2f}")
'''
