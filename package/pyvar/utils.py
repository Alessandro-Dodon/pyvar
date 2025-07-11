#----------------------------------------------------------
# Packages
# ----------------------------------------------------------
import pandas as pd
from arch import arch_model
from zipfile import ZipFile
from io import BytesIO
import requests
import base64
import itertools
from IPython.display import HTML
import plotly.express as px
import numpy as np


#----------------------------------------------------------
# Checks for Financial Matrices 
#----------------------------------------------------------
def validate_matrix(matrix: pd.DataFrame, context: str = ""):
    """
    Main
    ----
    Perform basic structural and statistical checks on any matrix 
    used in financial modeling (e.g., prices, positions, returns).

    If you don't download the data using our functions, like using 
    your own csv file, you can use this function to check the data
    immediately. We don't recommend however for our basic applications
    to use portfolios with overall negative values (complete shorts).
    Also, notice that a near zero value of portfolio (perfect hedge)
    may be a problem in some other functions. 

    Parameters
    ----------
    matrix : pd.DataFrame
        Time-indexed matrix with assets as columns.
    context : str, optional
        Context label for warnings (e.g., 'raw prices', 'portfolio').

    Warns
    -----
    - If NaNs are present.
    - If sample size is less than number of columns.
    - If any column has near-zero variance.
    - If the covariance matrix is not positive semi-definite.
    """
    label = f"[{context}]" if context else ""

    if matrix.isnull().any().any():
        print(f"[warning] {label} NaNs detected — clean the data before analysis.")

    n_observations, n_assets = matrix.shape 
    if n_observations < n_assets:
        print(f"[warning] {label} Fewer rows ({n_observations}) than columns ({n_assets}) — covariance may be unstable.")

    variances = matrix.var()
    near_zero = variances < 1e-10
    if near_zero.any():
        bad_assets = matrix.columns[near_zero].tolist()
        print(f"[warning] {label} Near-zero variance in: {bad_assets} — may cause instability.")

    cov = matrix.cov().values
    eigenvalues = np.linalg.eigvalsh(cov)
    if (eigenvalues < -1e-8).any():
        print(f"[warning] {label} Covariance matrix not PSD — negative eigenvalues detected.")


#----------------------------------------------------------
# Garch Helper
#----------------------------------------------------------
def fit_garch_model(returns, p=1, q=1, model="GARCH", distribution="normal"):
    """
    Fit a GARCH-family model and return standardized residuals, conditional volatility series,
    latest volatility, and fitted parameters. Always returns cond_vol as a pandas.Series.
    """
    returns_scaled = returns * 100
    index = returns.index if isinstance(returns, pd.Series) else pd.RangeIndex(len(returns))

    match model.upper():
        case "GARCH":
            garch = arch_model(returns_scaled, vol="GARCH", p=p, q=q, dist=distribution)
        case "EGARCH":
            garch = arch_model(returns_scaled, vol="EGARCH", p=p, q=q, dist=distribution)
        case "GJR":
            garch = arch_model(returns_scaled, vol="GARCH", p=p, o=1, q=q, dist=distribution)
        case "APARCH":
            garch = arch_model(returns_scaled, vol="APARCH", p=p, o=1, q=q, dist=distribution)
        case _:
            raise ValueError("Unsupported GARCH model")

    fit = garch.fit(disp="off")
    cond_vol = pd.Series(fit.conditional_volatility / 100, index=index)
    mu = fit.params.get("mu", 0.0)

    residuals = (returns - mu / 100) / cond_vol
    latest_vol = cond_vol.iloc[-1]

    return residuals, cond_vol, latest_vol, fit.params


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
# Scientific and Financial Notation (Plotting)
# -------------------------------------------------------
def format_money(x, _):
    """Format values as human-readable financial units (e.g., 1.2K, 3.4M)."""
    abs_x = abs(x)
    if abs_x >= 1_000_000_000:
        return f"{x/1_000_000_000:.1f}B"
    elif abs_x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    elif abs_x >= 1_000:
        return f"{x/1_000:.1f}K"
    else:
        return f"{x:.0f}"

def format_scientific(y, _):
    """Format values in full scientific notation (e.g., 1e-4)."""
    return f"{y:.0e}"


#----------------------------------------------------------
# Asset Color Map (Plotting)
#----------------------------------------------------------
def get_asset_color_map(assets):
    """
    Main
    ----
    Generate consistent colors for asset-level visualizations.
    Assigns a unique color to each asset using Plotly's qualitative palette, 
    cycling through it as needed. Useful for consistent coloring in portfolio plots.
    This is a support function.

    Parameters
    ----------
    assets : list-like
        List of asset names (strings).

    Returns
    -------
    dict
        Dictionary mapping each asset to a color string.
    """
    base_colors = px.colors.qualitative.Plotly
    color_cycle = itertools.cycle(base_colors)
    return {asset: next(color_cycle) for asset in assets}


#----------------------------------------------------------
# Display helper (Plotting)
#----------------------------------------------------------
def display_high_dpi_inline(png_bytes, width):
    """
    Main
    ----
    Display a high-resolution PNG image inline in a notebook.
    Encodes the image in base64 and renders it with a specified width.

    This is a support function.

    Parameters
    ----------
    png_bytes : bytes
        PNG image in byte format.
    width : int
        Width in pixels for display.

    Returns
    -------
    IPython.display.HTML
        HTML image element for inline display.
    """
    encoded = base64.b64encode(png_bytes).decode("utf-8")
    return HTML(f'<img src="data:image/png;base64,{encoded}" style="width:{width}px;"/>')