"""
Correlation-Based VaR and Expected Shortfall Module
------------------------------------------------------

Implements portfolio-level Value-at-Risk (VaR) and Expected Shortfall (ES) 
estimators based on time-varying correlation models. The module supports both 
a simplified parametric approach—assuming normally distributed returns—and a 
semi-empirical method that uses dynamically computed standardized innovations 
to estimate empirical quantiles.

Portfolio positions are represented as monetary exposures evolving over time 
in a buy-and-hold setting. Risk is measured using either rolling sample 
covariance (Moving Average) or exponentially weighted covariance 
(RiskMetrics-style EWMA).

The general-purpose es_correlation function automatically detects whether 
empirical innovations are available and applies the appropriate method 
(parametric or semi-empirical) to compute Expected Shortfall.

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
- var_corr_moving_average_param: Portfolio VaR using moving average covariance matrix (parametric)
- var_corr_ewma_param: Portfolio VaR using exponentially weighted covariance (parametric)
- var_corr_moving_average_sp: Portfolio VaR using moving average with standardized innovations
- var_corr_ewma_sp: Portfolio VaR using EWMA with standardized innovations
- es_correlation: Expected Shortfall based on conditional volatility and either normal or empirical quantile
"""

# TODO: check all formulas
# TODO: check normalization
# TODO: check unified ES logic

#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm


#----------------------------------------------------------
# MA Correlation VaR (Parametric)
#----------------------------------------------------------
def var_corr_moving_average_param(x_matrix, confidence_level=0.99, window_size=20):
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
def var_corr_ewma_param(x_matrix, confidence_level=0.99, lambda_decay=0.94):
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
# MA Correlation VaR (Standardized Innovations)
#----------------------------------------------------------
def var_corr_moving_average_sp(x_matrix, confidence_level=0.99, window_size=20):
    """
    Moving average VaR using weights and empirical innovations distribution.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions per asset over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR (e.g., 0.99). Default is 0.99.
    window_size : int, optional
        Rolling window size for covariance estimation. Default is 20.
    min_weight : float, optional
        Minimum allowed absolute weight (e.g., 0.02 = 2%). Default is 0.02.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with returns, volatility, VaR, VaR monetary, and violation flag.
    """
    min_weight = 0.02

    returns = x_matrix.pct_change().dropna()
    portfolio_value_series = x_matrix.sum(axis=1)
    if (portfolio_value_series <= 0).any():
        raise ValueError("Portfolio has zero or negative total value on some dates.")

    weights = x_matrix.div(portfolio_value_series, axis=0)
    if (weights.abs().min(axis=1) < min_weight).any():
        raise ValueError("Some asset weights are too small — check portfolio composition.")
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Weights do not sum to 1 on all dates.")

    portfolio_returns = (weights * returns).sum(axis=1)
    rolling_covs = returns.rolling(window=window_size).cov()

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

        z_t = r_t / portfolio_volatility
        volatilities.append(portfolio_volatility)
        z_scores.append(z_t)
        valid_index.append(date)

    z_scores = pd.Series(z_scores, index=valid_index)
    z_quantile = np.quantile(z_scores, 1 - confidence_level)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities,
        "Innovations": z_scores
    }, index=valid_index)

    result_data["VaR"] = -z_quantile * result_data["Volatility"]
    result_data["VaR Monetary"] = result_data["VaR"] * x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR Violation"] = result_data["Returns"] * x_matrix.sum(axis=1).loc[valid_index] < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# RiskMetrics Correlation VaR (Standardized Innovations)
#----------------------------------------------------------
def var_corr_ewma_sp(x_matrix, confidence_level=0.99, lambda_decay=0.94):
    """
    EWMA VaR using weights and empirical innovations distribution.

    Parameters
    ----------
    x_matrix : pd.DataFrame
        Monetary positions per asset over time (T × N).
    confidence_level : float, optional
        Confidence level for VaR. Default is 0.99.
    lambda_decay : float, optional
        EWMA decay factor. Default is 0.94.
    min_weight : float, optional
        Minimum allowed absolute weight. Default is 0.02.

    Returns
    -------
    result_data : pd.DataFrame
        DataFrame with returns, volatility, VaR, VaR monetary, and violation flag.
    """
    min_weight = 0.02

    returns = x_matrix.pct_change().dropna()
    portfolio_value_series = x_matrix.sum(axis=1)
    if (portfolio_value_series <= 0).any():
        raise ValueError("Portfolio has zero or negative total value on some dates.")

    weights = x_matrix.div(portfolio_value_series, axis=0)
    if (weights.abs().min(axis=1) < min_weight).any():
        raise ValueError("Some asset weights are too small — check portfolio composition.")
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise ValueError("Weights do not sum to 1 on all dates.")

    portfolio_returns = (weights * returns).sum(axis=1)

    ewma_cov = returns.cov().values
    cov_matrices = []
    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    weights = weights.loc[returns.index]
    volatilities = []
    z_scores = []
    valid_index = []

    for t, sigma in enumerate(cov_matrices):
        date = returns.index[t]
        w_t = weights.loc[date].values.reshape(-1, 1)
        r_t = portfolio_returns.loc[date]

        portfolio_variance = float(w_t.T @ sigma @ w_t)
        portfolio_volatility = np.sqrt(portfolio_variance)
        z_t = r_t / portfolio_volatility

        volatilities.append(portfolio_volatility)
        z_scores.append(z_t)
        valid_index.append(date)

    z_scores = pd.Series(z_scores, index=valid_index)
    z_quantile = np.quantile(z_scores, 1 - confidence_level)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities,
        "Innovations": z_scores
    }, index=valid_index)

    result_data["VaR"] = -z_quantile * result_data["Volatility"]
    result_data["VaR Monetary"] = result_data["VaR"] * x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR Violation"] = result_data["Returns"] * x_matrix.sum(axis=1).loc[valid_index] < -result_data["VaR Monetary"]

    return result_data


#----------------------------------------------------------
# Expected Shortfall for Correlation Models (General)
#----------------------------------------------------------
def es_correlation(data, confidence_level=0.99):
    """
    General-purpose Expected Shortfall (ES) estimator for correlation-based VaR models.

    This function automatically switches between:
    - Empirical ES: if 'Innovations' column is present, averages the lower tail of standardized residuals.
    - Parametric ES: otherwise assumes normality and uses the closed-form Normal ES formula.

    In both cases, portfolio value is inferred using the ratio VaR Monetary / VaR, and both
    decimal and monetary ES are returned.

    Parameters
    ----------
    data : pd.DataFrame
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
    data : pd.DataFrame
        Extended DataFrame with:
        - 'ES': Expected Shortfall in decimal units
        - 'ES Monetary': Expected Shortfall in monetary units

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {"Volatility", "VaR", "VaR Monetary"}
    if not required.issubset(data.columns):
        raise ValueError("Data must contain 'Volatility', 'VaR', and 'VaR Monetary' columns.")

    alpha = confidence_level
    portfolio_value = data["VaR Monetary"] / data["VaR"]

    if "Innovations" in data.columns:
        # Empirical ES based on standardized residuals
        threshold = np.percentile(data["Innovations"], 100 * (1 - alpha))
        tail_mean = data["Innovations"][data["Innovations"] < threshold].mean()
        data["ES"] = -data["Volatility"] * tail_mean
    else:
        # Parametric ES using normal assumption
        z = norm.ppf(1 - alpha)
        pdf_z = norm.pdf(z)
        data["ES"] = data["Volatility"] * pdf_z / (1 - alpha)

    data["ES Monetary"] = data["ES"] * portfolio_value

    return data
