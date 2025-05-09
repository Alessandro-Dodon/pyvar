#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
from arch.__future__ import reindexing
from arch.univariate import ConstantMean, GARCH, StudentsT

#################################################
# Note: double check all formulas 
#       check normalization
#################################################

#----------------------------------------------------------
# MA Correlation VaR (Parametric)
#----------------------------------------------------------
def var_corr_moving_average(x_matrix, confidence_level=0.99, window_size=20):
    """
    Estimate portfolio Value-at-Risk (VaR) using a moving average covariance matrix.

    Computes daily portfolio VaR assuming normally distributed returns and 
    a rolling sample covariance matrix for asset return correlations. Monetary 
    positions vary over time, capturing a buy-and-hold portfolio logic.

    Parameters:
    - x_matrix (pd.DataFrame): Monetary positions per asset (T × N), one row per day.
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - window_size (int): Rolling window size for covariance estimation.

    Returns:
    - result_data (pd.DataFrame): DataFrame with:
        - 'Returns': portfolio return in decimal form
        - 'Volatility': estimated portfolio volatility (decimal)
        - 'VaR': portfolio VaR as a decimal loss
        - 'VaR Monetary': portfolio VaR in monetary units
        - 'VaR Violation': True if actual loss exceeds VaR
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)
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

    Computes daily portfolio VaR using an exponentially weighted moving average (EWMA) 
    of the asset return covariance matrix. Assumes normally distributed returns and 
    time-varying monetary positions under a buy-and-hold strategy.

    Parameters:
    - x_matrix (pd.DataFrame): Monetary positions per asset (T × N), one row per day.
    - confidence_level (float): Confidence level for VaR (e.g., 0.99).
    - lambda_decay (float): EWMA decay factor (e.g., 0.94).

    Returns:
    - result_data (pd.DataFrame): DataFrame with:
        - 'Returns': portfolio return in decimal form
        - 'Volatility': estimated portfolio volatility (decimal)
        - 'VaR': VaR as a decimal loss
        - 'VaR Monetary': VaR in monetary units
        - 'VaR Violation': True if actual loss exceeds VaR
    """
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

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
