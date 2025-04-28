import numpy as np
import pandas as pd
from scipy.stats import norm
from arch.__future__ import reindexing
from arch.univariate import ConstantMean, GARCH, StudentsT



# MA VaR
def var_movingaverage_param(x_matrix, confidence_level=0.99, window_size=20):
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
def var_riskmetrics_param(x_matrix, confidence_level=0.99, lambda_decay=0.94):
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



#------------------------------------------------------------------------------------------------------------------
# We try the same models with z computed empirically, to match the volatility models previoous logic and get the ES
#------------------------------------------------------------------------------------------------------------------



# MA VaR (Empirical)
def var_movingaverage_empirical(x_matrix, confidence_level=0.99, window_size=20):
    """
    Estimate portfolio VaR using a moving average (rolling window) of sample covariances
    and empirical quantiles of portfolio standardized innovations.

    This method assumes no fixed distribution for returns (non-parametric),
    and uses a rolling historical sample covariance matrix to estimate
    the time-varying variance-covariance matrix of returns.

    VaR is computed as:
        VaR_t = - z_alpha_empirical * sqrt(x_tᵀ Σ_t x_t) × portfolio_value_t

    Innovations are computed dynamically as:
        innovation_t = portfolio_return_t / sqrt(x_tᵀ Σ_t x_t)

    where:
    - x_t: vector of monetary positions at time t
    - Σ_t: estimated covariance matrix at time t (from rolling sample)
    - portfolio_return_t: portfolio percentage return at time t

    Parameters:
    - x_matrix: pd.DataFrame (T × N), monetary positions per asset over time
    - confidence_level: float, VaR confidence level (e.g., 0.99)
    - window_size: int, rolling window size for the moving average

    Assumptions:
    - Portfolio returns are not assumed to be Normally distributed
    - Time-varying volatility comes from sample covariances in a rolling window

    Returns:
    - result_data: pd.DataFrame with 'Returns', 'Volatility', 'Innovations', 'VaR', 'VaR Violation'
    - next_day_var: float, VaR estimate for next period (monetary value)
    """
    
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    rolling_covs = returns.rolling(window=window_size).cov()

    volatilities = []
    innovations = []
    valid_index = []

    for t in range(window_size - 1, len(returns)):
        cov_matrix = rolling_covs.loc[returns.index[t]]
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        portfolio_variance = float(x_t.T @ cov_matrix.values @ x_t)
        portfolio_volatility = np.sqrt(portfolio_variance)

        volatilities.append(portfolio_volatility)
        innovations.append(portfolio_returns.iloc[t] / portfolio_volatility)
        valid_index.append(returns.index[t])

    # Create the result DataFrame
    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities,
        "Innovations": innovations
    }, index=valid_index)

    # Compute empirical quantile for VaR
    empirical_quantile = np.percentile(result_data["Innovations"].dropna(), 100 * (1 - confidence_level))

    # Compute VaR series in percentage returns first
    result_data["VaR"] = -result_data["Volatility"] * empirical_quantile

    # Scale to monetary values
    portfolio_value = x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR"] = result_data["VaR"] * portfolio_value

    # Identify VaR violations
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value < -result_data["VaR"]

    # Forecast next day VaR in monetary units
    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = rolling_covs.loc[returns.index[-1]].values
    next_day_vol = float(np.sqrt(x_last.T @ sigma_last @ x_last))
    latest_portfolio_value = x_matrix.sum(axis=1).iloc[-1]
    next_day_var = abs(empirical_quantile * next_day_vol * latest_portfolio_value)

    return result_data, next_day_var



# RiskMetrics VaR (Empirical)
def var_riskmetrics_empirical(x_matrix, confidence_level=0.99, lambda_decay=0.94):
    """
    Estimate portfolio VaR using the RiskMetrics (EWMA) model and empirical quantiles
    of portfolio standardized innovations.

    This method uses exponential smoothing to estimate a time-varying 
    variance-covariance matrix of returns. No distributional assumption is made
    on returns (non-parametric VaR).

    VaR is computed as:
        VaR_t = - z_alpha_empirical * sqrt(x_tᵀ Σ_t x_t) × portfolio_value_t

    Innovations are computed dynamically as:
        innovation_t = portfolio_return_t / sqrt(x_tᵀ Σ_t x_t)

    where:
    - x_t: vector of monetary positions at time t
    - Σ_t: estimated EWMA covariance matrix at time t
    - portfolio_return_t: portfolio percentage return at time t

    Parameters:
    - x_matrix: pd.DataFrame (T × N), monetary positions per asset over time
    - confidence_level: float, VaR confidence level (e.g., 0.99)
    - lambda_decay: float, exponential decay factor (default = 0.94)

    Assumptions:
    - Portfolio returns are not assumed to be Normally distributed
    - Volatility clusters over time, captured by EWMA covariance updates

    Returns:
    - result_data: pd.DataFrame with 'Returns', 'Volatility', 'Innovations', 'VaR', 'VaR Violation'
    - next_day_var: float, VaR estimate for next period (monetary value)
    """
    
    # Compute returns matrix from x
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    # Initialize EWMA covariance matrices
    ewma_cov = returns.cov().values
    cov_matrices = []

    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    # Align x_matrix to returns index
    x_matrix = x_matrix.loc[returns.index]

    volatilities = []
    innovations = []

    for t, sigma in enumerate(cov_matrices):
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        portfolio_variance = float(x_t.T @ sigma @ x_t)
        portfolio_volatility = np.sqrt(portfolio_variance)

        volatilities.append(portfolio_volatility)
        innovations.append(portfolio_returns.iloc[t] / portfolio_volatility)

    # Create result DataFrame
    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "Volatility": volatilities,
        "Innovations": innovations
    }, index=returns.index)

    # Compute empirical quantile for VaR
    empirical_quantile = np.percentile(result_data["Innovations"].dropna(), 100 * (1 - confidence_level))

    # Compute VaR in percentage returns first
    result_data["VaR"] = -result_data["Volatility"] * empirical_quantile

    # Scale to monetary values
    portfolio_value = x_matrix.sum(axis=1)
    result_data["VaR"] = result_data["VaR"] * portfolio_value

    # Identify VaR violations
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value < -result_data["VaR"]

    # Next-day forecast (last covariance + last position)
    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = cov_matrices[-1]
    next_day_vol = float(np.sqrt(x_last.T @ sigma_last @ x_last))
    latest_portfolio_value = x_matrix.sum(axis=1).iloc[-1]
    next_day_var = abs(empirical_quantile * next_day_vol * latest_portfolio_value)

    return result_data, next_day_var