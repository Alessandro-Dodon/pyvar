import numpy as np
import pandas as pd
from scipy.stats import norm
from arch.__future__ import reindexing
from arch.univariate import ConstantMean, GARCH, StudentsT



# MA VaR
def var_movingaverage(x_matrix, confidence_level=0.99, window_size=20):
    """
    Estimate portfolio VaR using a moving average (rolling window) of sample covariances.

    This method assumes returns are conditionally normally distributed, and
    uses a rolling historical sample covariance matrix to estimate the time-varying
    variance-covariance matrix of returns.

    VaR is computed as:
        VaR_t = z_alpha * sqrt(x_tᵀ Σ_t x_t)

    This approach uses monetary positions `x` as input rather than weights, 
    which simplifies short position handling.

    Parameters:
    - x_matrix: pd.DataFrame (T × N), monetary positions per asset over time
    - confidence_level: float, VaR confidence level (e.g., 0.99)
    - window_size: int, rolling window size for the moving average

    Assumptions:
    - Portfolio returns are approximately normally distributed over the window
    - Time-varying volatility comes from sample covariances in a rolling window
    """

    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    rolling_covs = returns.rolling(window=window_size).cov()
    z_alpha = norm.ppf(confidence_level)

    var_series = []
    vol_series = []
    valid_index = []

    for t in range(window_size - 1, len(returns)):
        cov_matrix = rolling_covs.loc[returns.index[t]]
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        port_var = float(x_t.T @ cov_matrix.values @ x_t)
        vol_series.append(np.sqrt(port_var))
        var_series.append(z_alpha * np.sqrt(port_var))
        valid_index.append(returns.index[t])

    result_data = pd.DataFrame({
        "Portfolio Return": portfolio_returns.loc[valid_index],
        "Portfolio Volatility": vol_series,
        "VaR": var_series
    }, index=valid_index)

    result_data["VaR Violation"] = result_data["Portfolio Return"] < -result_data["VaR"]

    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = rolling_covs.loc[returns.index[-1]].values
    next_day_vol = float(np.sqrt(x_last.T @ sigma_last @ x_last))
    next_day_var = float(z_alpha * next_day_vol)

    return result_data, next_day_var



# RiskMetrics VaR
def var_riskmetrics(x_matrix, confidence_level=0.99, lambda_decay=0.94):
    """
    Estimate portfolio VaR using the RiskMetrics (EWMA) model.

    This method uses exponential smoothing to estimate a time-varying 
    variance-covariance matrix of returns. It assumes conditional normality
    of returns and decays past shocks with a smoothing factor `lambda_decay`.

    VaR is computed as:
        VaR_t = z_alpha * sqrt(x_tᵀ Σ_t x_t)

    Input `x_matrix` contains monetary positions, enabling proper handling
    of short positions.

    Parameters:
    - x_matrix: pd.DataFrame (T × N), monetary positions per asset over time
    - confidence_level: float, VaR confidence level (e.g., 0.99)
    - lambda_decay: float, exponential decay factor (default = 0.94)

    Assumptions:
    - Conditional normality of portfolio returns
    - Volatility clusters over time, captured by EWMA covariance updates
    """

    # Compute returns matrix from x
    returns = x_matrix.pct_change().dropna()

    # Initialize EWMA covariance matrices
    ewma_cov = returns.cov().values
    cov_matrices = []

    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    # Convert x to aligned positions matrix
    x_matrix = x_matrix.loc[returns.index]

    # Compute portfolio-level VaR
    z_alpha = norm.ppf(confidence_level)
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    var_series = []
    vol_series = []

    for t, sigma in enumerate(cov_matrices):
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        var_t = float(z_alpha * np.sqrt(x_t.T @ sigma @ x_t))
        var_series.append(var_t)
        vol_series.append(float(np.sqrt(x_t.T @ sigma @ x_t)))

    result_data = pd.DataFrame({
        "Portfolio Return": portfolio_returns,
        "Portfolio Volatility": vol_series,
        "VaR": var_series
    }, index=returns.index)

    result_data["VaR Violation"] = result_data["Portfolio Return"] < -result_data["VaR"]

    # Next-day forecast (last covariance + last position)
    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = cov_matrices[-1]
    next_day_vol = float(np.sqrt(x_last.T @ sigma_last @ x_last))
    next_day_var = float(z_alpha * next_day_vol)

    return result_data, next_day_var



# VEC(1,1) VaR
def var_vec(x_matrix, confidence_level=0.99):
    """
    Estimate portfolio VaR using a simplified VEC(1,1) model.

    This model simulates multivariate GARCH-style volatility using a VEC(1,1)
    framework with fixed parameters α and β. It assumes conditional normality
    of returns and computes time-varying covariance matrices recursively.

    VaR is computed as:
        VaR_t = z_alpha * sqrt(x_tᵀ Σ_t x_t)

    This function uses monetary positions `x_matrix` directly rather than portfolio weights.

    Parameters:
    - x_matrix: pd.DataFrame (T × N), monetary positions per asset over time
    - confidence_level: float, VaR confidence level (e.g., 0.99)

    Assumptions:
    - Covariance follows VEC(1,1):  
    Σ_t = ω + α * (r_{t-1} r_{t-1}ᵀ) + β * Σ_{t-1}
    - Portfolio returns are conditionally normal
    """

    returns = x_matrix.pct_change().dropna()
    T, N = returns.shape

    # Initialize with sample VCV
    sigma_t = returns.cov().values
    sigma_series = []

    # VEC(1,1) parameters (constant + autoregressive + innovation)
    alpha = 0.03
    beta = 0.95
    omega = (1 - alpha - beta) * returns.cov().values

    for t in range(1, T):
        r_outer = np.outer(returns.iloc[t-1], returns.iloc[t-1])
        sigma_t = omega + alpha * r_outer + beta * sigma_t
        sigma_series.append(sigma_t.copy())

    # Align x with returns
    x_matrix = x_matrix.loc[returns.index[1:]]
    returns = returns.iloc[1:]

    z_alpha = norm.ppf(confidence_level)
    var_series = []
    vol_series = []
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    for t, sigma in enumerate(sigma_series):
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        port_var = float(x_t.T @ sigma @ x_t)
        vol_series.append(np.sqrt(port_var))
        var_series.append(z_alpha * np.sqrt(port_var))

    result_data = pd.DataFrame({
        "Portfolio Return": portfolio_returns,
        "Portfolio Volatility": vol_series,
        "VaR": var_series
    }, index=portfolio_returns.index)

    result_data["VaR Violation"] = result_data["Portfolio Return"] < -result_data["VaR"]

    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = sigma_series[-1]
    next_day_vol = float(np.sqrt(x_last.T @ sigma_last @ x_last))
    next_day_var = float(z_alpha * next_day_vol)

    return result_data, next_day_var




