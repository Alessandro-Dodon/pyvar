"""
Correlation-Based VaR and Expected Shortfall Module
------------------------------------------------------

Implements portfolio-level Value-at-Risk (VaR) and Expected Shortfall (ES) 
estimators based on time-varying correlation models. Assumes normally 
distributed returns and monetary positions that evolve over time in a 
buy-and-hold portfolio setting.

This module includes:
- Moving Average covariance estimation
- EWMA (RiskMetrics-style) covariance estimation
- Parametric ES estimation from conditional volatility

These models are designed to complement the broader framework for 
portfolio risk modeling, particularly when working with monetary 
positions matrices derived from multi-asset portfolios.

Authors
-------
Alessandro Dodon, Niccolò Lecce, Marco Gasparetti

Created
-------
May 2025

Contents
--------
- var_corr_moving_average: Portfolio VaR using moving average covariance matrix
- var_corr_ewma: Portfolio VaR using exponentially weighted covariance (EWMA)
- es_correlation: Expected Shortfall based on volatility and normal quantile
"""

# TODO: check all formulas
# TODO: check normalization

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


#----------------------------------------------------------
# MA Correlation VaR (Parametric)
#----------------------------------------------------------
def var_corr_moving_average(x_matrix, confidence_level=0.99, window_size=20):
    """
    Estimate portfolio Value-at-Risk (VaR) using a moving average covariance matrix.

    Assumes normally distributed returns and computes a rolling sample covariance 
    matrix of asset returns over a fixed window. The portfolio is represented by 
    daily monetary positions and reflects a buy-and-hold strategy. VaR is expressed 
    in both relative (percentage) and monetary terms. Volatility is scaled by total 
    portfolio value at each time step.

    The risk quantile is taken from the standard Normal distribution, which simplifies
    the VaR computation and makes it directly proportional to estimated volatility.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions for each asset over time (T × N).
        Rows are daily observations; columns are asset tickers.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    window_size : int, optional
        Window size (in days) for rolling covariance estimation. Default is 20.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame indexed by date, with the following columns:
        - 'Returns': Daily portfolio return (decimal)
        - 'Volatility': Estimated portfolio volatility (decimal)
        - 'VaR': Relative Value-at-Risk (decimal loss)
        - 'VaR Monetary': Absolute Value-at-Risk in monetary units
        - 'VaR Violation': Boolean flag indicating if actual loss exceeded VaR

    Raises
    ------
    ValueError
        If total portfolio value is zero or negative on any date.
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    # Defensive check for invalid portfolio value
    portfolio_value_series = x_matrix.sum(axis=1)
    if (portfolio_value_series <= 0).any():
        raise ValueError("Portfolio has zero or negative total value on some dates. Adjust positions before risk analysis.")

    rolling_covs = returns.rolling(window=window_size).cov()

    z = norm.ppf(1 - confidence_level)

    volatilities = []
    valid_index = []

    for t in range(window_size - 1, len(returns)):
        date = returns.index[t]
        cov_matrix = rolling_covs.loc[date]
        x_t = x_matrix.loc[date].values.reshape(-1, 1)

        portfolio_variance = float(x_t.T @ cov_matrix.values @ x_t)
        portfolio_value = x_matrix.sum(axis=1).loc[date]
        portfolio_volatility = np.sqrt(portfolio_variance) / portfolio_value

        volatilities.append(portfolio_volatility)
        valid_index.append(date)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities
    }, index=valid_index)

    result_data["VaR"] = -z * result_data["Volatility"]
    portfolio_value_series = x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# RiskMetrics Correlation VaR (Parametric)
#----------------------------------------------------------
def var_corr_ewma(x_matrix, confidence_level=0.99, lambda_decay=0.94):
    """
    Estimate portfolio Value-at-Risk (VaR) using EWMA covariance and a normal quantile.

    Assumes normally distributed returns and computes a time-varying portfolio VaR using 
    an exponentially weighted moving average (EWMA) covariance matrix. The portfolio is 
    represented by daily monetary positions, capturing a buy-and-hold strategy.

    The risk quantile is taken from the standard Normal distribution, which simplifies 
    the VaR computation and ensures consistency with RiskMetrics-style modeling.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions per asset over time (T × N). Rows are daily observations.
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    lambda_decay : float, optional
        EWMA decay factor (e.g., 0.94). Default is 0.94.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame indexed by date, with the following columns:
        - 'Returns': Daily portfolio return (decimal)
        - 'Volatility': Estimated portfolio volatility (decimal)
        - 'VaR': Relative Value-at-Risk (decimal loss)
        - 'VaR Monetary': Absolute Value-at-Risk in monetary units
        - 'VaR Violation': Boolean flag indicating if actual loss exceeded VaR

    Raises
    ------
    ValueError
        If total portfolio value is zero or negative on any date.
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    # Defensive check for invalid portfolio value
    portfolio_value_series = x_matrix.sum(axis=1)
    if (portfolio_value_series <= 0).any():
        raise ValueError("Portfolio has zero or negative total value on some dates. Adjust positions before risk analysis.")

    # EWMA covariance matrices
    ewma_cov = returns.cov().values
    cov_matrices = []
    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    x_matrix = x_matrix.loc[returns.index]
    z = norm.ppf(1 - confidence_level)

    volatilities = []
    valid_index = []

    for t, sigma in enumerate(cov_matrices):
        date = returns.index[t]
        x_t = x_matrix.loc[date].values.reshape(-1, 1)
        portfolio_value = x_matrix.sum(axis=1).loc[date]
        portfolio_variance = float(x_t.T @ sigma @ x_t)
        portfolio_volatility = np.sqrt(portfolio_variance) / portfolio_value

        volatilities.append(portfolio_volatility)
        valid_index.append(date)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities
    }, index=valid_index)

    result_data["VaR"] = -z * result_data["Volatility"]
    portfolio_value_series = x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# Expected Shortfall for Correlation Models (Parametric)
#----------------------------------------------------------
def es_correlation(data, confidence_level=0.99):
    """
    Estimate Expected Shortfall (ES) using a parametric normal formula.

    Computes time-varying Expected Shortfall under the assumption of normally 
    distributed returns. ES is derived from the conditional volatility and 
    scaled by the inferred portfolio value using the ratio of VaR monetary 
    to relative VaR. This function is designed to extend the output of 
    correlation-based VaR models.

    The ES is calculated using the standard Normal distribution, which simplifies 
    the formula and maintains consistency with parametric VaR.

    Parameters
    ----------
    data : pd.DataFrame
        Must include the following columns:
        - 'Volatility': Portfolio volatility in decimal units
        - 'VaR': Relative Value-at-Risk (decimal loss)
        - 'VaR Monetary': Absolute Value-at-Risk in monetary units
    confidence_level : float, optional
        Confidence level for ES estimation (e.g., 0.99). Default is 0.99.

    Returns
    -------
    data : pd.DataFrame
        Extended DataFrame with:
        - 'ES': Expected Shortfall in decimal units
        - 'ES Monetary': Expected Shortfall in monetary units

    Raises
    ------
    ValueError
        If any required column ('Volatility', 'VaR', or 'VaR Monetary') is missing from the input.
    """
    if "Volatility" not in data.columns or "VaR" not in data.columns or "VaR Monetary" not in data.columns:
        raise ValueError("Data must contain 'Volatility', 'VaR', and 'VaR Monetary' columns.")

    z = norm.ppf(1 - confidence_level)
    pdf_z = norm.pdf(z)
    alpha = confidence_level

    # Decimal ES formula for normal: ES = σ × φ(z) / (1 - α)
    data["ES"] = data["Volatility"] * pdf_z / (1 - alpha)

    # Portfolio value inferred from VaR Monetary / VaR
    portfolio_value = data["VaR Monetary"] / data["VaR"]
    data["ES Monetary"] = data["ES"] * portfolio_value

    return data