# ff3_var.py  –––  può finire nel tuo package `riskmetrics`

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from io import BytesIO
from zipfile import ZipFile
import requests




def single_factor_var_es(
    returns: pd.DataFrame,
    benchmark: pd.Series,
    weights: pd.Series,
    port_val: float,
    confidence_level: float = 0.99
) -> tuple[float, float, pd.DataFrame, pd.Series, pd.Series]:
    """
    Compute portfolio Value-at-Risk (VaR) and Expected Shortfall (ES) 
    using a single-factor (Sharpe) model.

    Parameters
    ----------
    returns : pd.DataFrame
        Time series of asset returns (columns are tickers).
    benchmark : pd.Series
        Time series of benchmark returns (e.g., market index).
    weights : pd.Series
        Portfolio weights (must sum to 1).
    portfolio_value : float
        Current total value of the portfolio (monetary units).
    confidence_level : float, optional
        Confidence level for VaR (default is 0.99).

    Returns
    -------
    var : float
        Portfolio VaR at the specified confidence level (monetary units).
    es : float
        Portfolio Expected Shortfall at the specified confidence level (monetary units).
    Sigma : pd.DataFrame
        Covariance matrix estimated by the single-factor model.
    betas : pd.Series
        Beta of each asset with respect to the benchmark.
    idiosyncratic_var : pd.Series
        Estimated idiosyncratic variance of each asset.
    """
    # 1) Estimate market variance
    market_var = benchmark.var(ddof=0)

    # 2) Compute covariance between each asset and the benchmark
    cov_with_benchmark = returns.apply(lambda x: x.cov(benchmark))

    # 3) Compute betas
    betas = cov_with_benchmark / market_var

    # 4) Compute idiosyncratic variances
    idiosyncratic_var = returns.var(ddof=0) - betas.pow(2) * market_var

    # 5) Build total covariance matrix: B Σ_m B' + diag(idiosyncratic_var)
    tickers = returns.columns
    # outer product of betas scaled by market variance
    factor_cov = np.outer(betas, betas) * market_var
    Sigma = pd.DataFrame(factor_cov, index=tickers, columns=tickers)
    # add idiosyncratic variances to the diagonal
    for t in tickers:
        Sigma.at[t, t] += idiosyncratic_var[t]

    # 6) Portfolio volatility
    port_vol = np.sqrt(weights.values @ Sigma.values @ weights.values)

    # 7) Compute VaR
    z = norm.ppf(confidence_level)
    var = z * port_vol * port_val

    # 8) Compute ES
    tail_prob = 1 - confidence_level
    es = (port_vol * norm.pdf(norm.ppf(tail_prob)) / tail_prob) * port_val

    return var, es, Sigma, betas, idiosyncratic_var




#FAMA-FRENCH
# ----------------------------------------------------------------------
# helper per scaricare i fattori Fama-French daily
# ----------------------------------------------------------------------
_FF_ZIP_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/"
    "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)

def load_ff3_factors(start=None, end=None) -> pd.DataFrame:
    """
    Restituisce un DataFrame con colonne
       ['Mkt_RF', 'SMB', 'HML', 'RF'] in frazioni (non %).
    """
    resp = requests.get(_FF_ZIP_URL, timeout=30)
    resp.raise_for_status()
    zf   = ZipFile(BytesIO(resp.content))
    csvf = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
    ff   = pd.read_csv(zf.open(csvf), skiprows=3, index_col=0)

    # righe utili = AAAAMMGG
    mask  = ff.index.astype(str).str.match(r"^\d{8}$")
    ff    = ff.loc[mask].astype(float) / 100.0
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    ff.columns = ["Mkt_RF", "SMB", "HML", "RF"]

    if start: ff = ff.loc[start:]
    if end:   ff = ff.loc[:end]
    return ff.sort_index()

# ----------------------------------------------------------------------
# funzione principale
# ----------------------------------------------------------------------
def ff3_var_cvar(
    *,
    returns: pd.DataFrame | None = None,
    prices : pd.DataFrame | None = None,
    weights: pd.Series    | None = None,
    shares : pd.Series    | None = None,
    alpha: float = 0.95,
    factors: pd.DataFrame | None = None,
) -> tuple[float, float]:
    """
    VaR e CVaR di portafoglio con modello Fama–French 3-fattori.

    Parametri
    ----------
    returns | prices
        O rendimenti (% frazione) DAILY oppure prezzi (in valuta-base).
    weights | shares
        Pesi di portafoglio (sommatoria = 1) **oppure** numero titoli posseduti.
    alpha
        Livello di confidenza (es. 0.95 → VaR al 95 %).
    factors
        DataFrame di fattori FF3 già pronto; se None lo scarica da Ken French.

    Ritorna
    -------
    (VaR, CVaR) in unità monetarie della valuta-base.
    """
    if returns is None and prices is None:
        raise ValueError("devi passare `returns` o `prices`")

    if prices is not None and returns is None:
        returns = prices.pct_change().dropna()

    if weights is None and shares is None:
        raise ValueError("devi passare `weights` o `shares`")
    if shares is not None:
        latest_price = prices.iloc[-1] if prices is not None else None
        if latest_price is None:
            raise ValueError("per usare `shares` servono anche i `prices`")
        port_val = (latest_price * shares).sum()
        weights  = (latest_price * shares) / port_val
    else:
        port_val = 1.0  # scala neutra – il VaR verrà poi scalato
        weights  = weights / weights.sum()

    # fattori FF3
    if factors is None:
        factors = load_ff3_factors(start=returns.index[0])
    factors = factors.reindex(returns.index).ffill()

    # regressione multipla per ogni asset
    X      = sm.add_constant(factors[["Mkt_RF", "SMB", "HML"]])
    excess = returns.sub(factors["RF"], axis=0)

    betas, resid_var = {}, {}
    for tkr in returns:
        yx  = pd.concat([excess[tkr], X], axis=1).dropna()
        res = sm.OLS(yx.iloc[:, 0], yx.iloc[:, 1:]).fit()
        betas[tkr]     = res.params.drop("const")
        resid_var[tkr] = res.resid.var(ddof=0)

    B   = pd.DataFrame(betas).T.values            # n_assets × 3
    Σf  = factors[["Mkt_RF", "SMB", "HML"]].cov().values  # 3 × 3
    Σ   = B @ Σf @ B.T + np.diag(pd.Series(resid_var).values)

    # varianza portafoglio
    σ_p = np.sqrt(weights.values @ Σ @ weights.values)

    z     = norm.ppf(alpha)
    VaR   = z * σ_p * port_val
    tail  = 1 - alpha
    CVaR  = σ_p * norm.pdf(norm.ppf(tail)) / tail * port_val
    return VaR, CVaR


'''
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
