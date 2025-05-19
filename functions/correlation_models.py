"""
Correlation-Based VaR and Expected Shortfall Module
------------------------------------------------------

Implements portfolio-level Value-at-Risk (VaR) and Expected Shortfall (ES) 
using time-varying correlation models. Supports both parametric (normal quantile) 
and semi-empirical (empirical quantile from standardized residuals) approaches.

Monetary positions evolve under a buy-and-hold strategy. Risk is estimated using 
either a rolling sample covariance (Moving Average) or an exponentially weighted 
covariance matrix (EWMA).

Expected Shortfall is computed via the es_correlation function, which automatically 
selects between parametric and empirical estimation based on model output.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

- ma_correlation_var: Portfolio VaR using a rolling covariance matrix (MA).
- ewma_correlation_var: Portfolio VaR using an EWMA covariance matrix.
- es_correlation: Expected Shortfall estimator that adapts to the presence of empirical innovations.
"""

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


#----------------------------------------------------------
# MA Correlation VaR (Parametric or Empirical)
#----------------------------------------------------------
def ma_correlation_var(
    x_matrix: pd.DataFrame,
    confidence_level: float = 0.99,
    window_size: int = 20,
    distribution: str = "normal",
) -> pd.DataFrame:
    """
    Main
    ----
    Estimate portfolio Value-at-Risk (VaR) using a moving average covariance matrix.

    This function supports both parametric and semi-empirical approaches:
    - In "normal" mode, VaR is computed using the standard normal quantile.
    - In "empirical" mode, the quantile is estimated from standardized innovations
      estimated from standardized residuals using historical returns and volatility.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions per asset over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    window_size : int, optional
        Rolling window size for covariance estimation. Default is 20.
    distribution : str, optional
        Type of quantile used: "normal" or "empirical". Default is "normal".

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with:
        - 'Returns': Portfolio returns
        - 'Volatility': Conditional volatility
        - 'VaR': Relative VaR (decimal)
        - 'VaR Monetary': Absolute VaR in monetary units
        - 'VaR Violation': Breach indicator
        - 'Innovations': Standardized residuals (if empirical mode)
    
    Raises
    ------
    ValueError
        If weights are invalid in empirical mode.
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_value_series = x_matrix.sum(axis=1)

    # Portfolio returns (always needed)
    weights = x_matrix.div(portfolio_value_series, axis=0)
    portfolio_returns = (weights * returns).sum(axis=1)

    # Rolling covariance matrices
    rolling_covs = returns.rolling(window=window_size).cov()

    # Empirical mode: check weights validity
    if distribution == "empirical":
        min_weight = 0.02
        if (weights.abs().min(axis=1) < min_weight).any():
            raise ValueError("Some asset weights are too small — check portfolio composition.")
        if (weights < -1.0).any().any():
            raise ValueError("Some asset weights exceed 100% short — check portfolio composition.")
        if not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("Weights do not sum to 1 on all dates.")
        
    volatilities = []
    z_scores = []
    valid_index = []

    for t in range(window_size - 1, len(returns)):
        date = returns.index[t]
        cov_matrix = rolling_covs.loc[date]
        w_t = weights.loc[date].values.reshape(-1, 1)
        r_t = portfolio_returns.loc[date]

        portfolio_variance = float(w_t.T @ cov_matrix.values @ w_t)
        portfolio_volatility = np.sqrt(portfolio_variance)

        volatilities.append(portfolio_volatility)
        valid_index.append(date)

        if distribution == "empirical":
            z_t = r_t / portfolio_volatility
            z_scores.append(z_t)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities
    }, index=valid_index)

    if distribution == "empirical":
        z_scores_series = pd.Series(z_scores, index=valid_index)
        z_quantile = np.quantile(z_scores_series, 1 - confidence_level)
        result_data["Innovations"] = z_scores_series
    else:
        z_quantile = norm.ppf(1 - confidence_level)

    result_data["VaR"] = -z_quantile * result_data["Volatility"]
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series.loc[valid_index]
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series.loc[valid_index] < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# EWMA Correlation VaR (Parametric or Empirical)
#----------------------------------------------------------
def ewma_correlation_var(
    x_matrix: pd.DataFrame,
    confidence_level: float = 0.99,
    lambda_decay: float = 0.94,
    distribution: str = "normal",
) -> pd.DataFrame:
    """
    Main
    ----
    Estimate portfolio Value-at-Risk (VaR) using an exponentially weighted moving 
    average (EWMA) covariance matrix.

    This function supports both parametric and semi-empirical approaches:
    - In "normal" mode, VaR is computed using the standard normal quantile.
    - In "empirical" mode, the quantile is estimated from standardized innovations
      estimated from standardized residuals using historical returns and volatility.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions per asset over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    lambda_decay : float, optional
        EWMA decay factor. Default is 0.94.
    distribution : str, optional
        Type of quantile used: "normal" or "empirical". Default is "normal".
    min_weight : float, optional
        Minimum allowed absolute weight in empirical mode. Default is 0.02.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with:
        - 'Returns': Portfolio returns
        - 'Volatility': Conditional volatility
        - 'VaR': Relative VaR (decimal)
        - 'VaR Monetary': Absolute VaR in monetary units
        - 'VaR Violation': Breach indicator
        - 'Innovations': Standardized residuals (if empirical mode)

    Raises
    ------
    ValueError
        If weights are invalid in empirical mode.
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_value_series = x_matrix.sum(axis=1)

    # Portfolio weights and returns
    weights = x_matrix.div(portfolio_value_series, axis=0)
    portfolio_returns = (weights * returns).sum(axis=1)

    if distribution == "empirical":
        min_weight = 0.02
        if (weights.abs().min(axis=1) < min_weight).any():
            raise ValueError("Some asset weights are too small — check portfolio composition.")
        if (weights < -1.0).any().any():
            raise ValueError("Some asset weights exceed 100% short — check portfolio composition.")
        if not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("Weights do not sum to 1 on all dates.")

    # EWMA covariance matrices
    ewma_cov = returns.cov().values
    cov_matrices = []
    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    weights = weights.loc[returns.index]
    portfolio_returns = portfolio_returns.loc[returns.index]
    portfolio_value_series = portfolio_value_series.loc[returns.index]

    volatilities = []
    z_scores = []
    valid_index = []

    for t, sigma in enumerate(cov_matrices):
        date = returns.index[t]
        w_t = weights.loc[date].values.reshape(-1, 1)
        r_t = portfolio_returns.loc[date]

        portfolio_variance = float(w_t.T @ sigma @ w_t)
        portfolio_volatility = np.sqrt(portfolio_variance)

        volatilities.append(portfolio_volatility)
        valid_index.append(date)

        if distribution == "empirical":
            z_scores.append(r_t / portfolio_volatility)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities
    }, index=valid_index)

    if distribution == "empirical":
        z_scores_series = pd.Series(z_scores, index=valid_index)
        z_quantile = np.quantile(z_scores_series, 1 - confidence_level)
        result_data["Innovations"] = z_scores_series
    else:
        z_quantile = norm.ppf(1 - confidence_level)

    result_data["VaR"] = -z_quantile * result_data["Volatility"]
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series.loc[valid_index]
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series.loc[valid_index] < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# Expected Shortfall for Correlation Models (General)
#----------------------------------------------------------
def correlation_es(result_data, confidence_level=0.99):
    """
    Main
    ----
    General-purpose Expected Shortfall (ES) estimator for correlation-based VaR models.

    This function automatically switches between:
    - Empirical ES: if 'Innovations' column is present, averages the lower tail of standardized residuals.
    - Parametric ES: otherwise assumes normality and uses the closed-form Normal ES formula.

    In both cases, portfolio value is inferred using the ratio VaR Monetary / VaR, and both
    decimal and monetary ES are returned.

    Parameters
    ----------
    result_data : pd.DataFrame
        Must include:
        - 'Volatility': Conditional standard deviation (decimal units)
        - 'VaR': Relative Value-at-Risk (decimal loss)
        - 'VaR Monetary': Absolute Value-at-Risk in monetary units
        Optionally:
        - 'Innovations': Standardized residuals (for empirical ES)

    confidence_level : float, optional
        Confidence level for ES estimation (e.g., 0.99). Default is 0.99.

    Returns
    -------
    result_data : pd.DataFrame
        Extended DataFrame with:
        - 'ES': Expected Shortfall in decimal units
        - 'ES Monetary': Expected Shortfall in monetary units

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {"Volatility", "VaR", "VaR Monetary"}
    if not required.issubset(result_data.columns):
        raise ValueError("Data must contain 'Volatility', 'VaR', and 'VaR Monetary' columns.")

    alpha = confidence_level
    portfolio_value = result_data["VaR Monetary"] / result_data["VaR"]

    if "Innovations" in result_data.columns:
        # Empirical ES based on standardized residuals
        threshold = np.percentile(result_data["Innovations"], 100 * (1 - alpha))
        tail_mean = result_data["Innovations"][result_data["Innovations"] < threshold].mean()
        result_data["ES"] = -result_data["Volatility"] * tail_mean
    else:
        # Parametric ES using normal assumption
        z = norm.ppf(1 - alpha)
        pdf_z = norm.pdf(z)
        result_data["ES"] = result_data["Volatility"] * pdf_z / (1 - alpha)

    result_data["ES Monetary"] = result_data["ES"] * portfolio_value

    return result_data


