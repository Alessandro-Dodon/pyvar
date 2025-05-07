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
# Note2: can use z from the normal (parametric) to make it much easier.
# Then the volatility is the same, just z changes. ES would be the classic ES formula
# for normal, but the volatility from this VaR with dynamic correlations makes it time-
# varying. 
# Note3: when using VCV and x and x', with z from normal, the VaR is already in monetary terms.

#----------------------------------------------------------
# MA Correlation VaR 
#----------------------------------------------------------
def var_corr_ma(x_matrix, confidence_level=0.99, window_size=20):
    """
    Moving Average VaR Estimation (Empirical, Non-Parametric, with Correlations).

    Estimate portfolio Value-at-Risk (VaR) using a rolling sample covariance matrix
    and empirical quantiles of standardized portfolio innovations.

    Description:
    - This method estimates VaR based on empirical innovations, allowing full asset correlations.
    - Volatility is computed via a rolling sample covariance of asset returns.
    - Innovations (standardized shocks) are computed from portfolio returns and volatility.
    - VaR is first calculated in decimal format (as percentage of wealth), and then scaled
      to monetary units using the total portfolio value.

    Formulas:
    - Portfolio return (decimal units):
        Rₜ = (xₜ · rₜ) / (∑ xₜ)
        where xₜ = monetary positions at t, rₜ = return vector at t

    - Rolling portfolio variance:
        σ²ₜ = xₜᵀ Σₜ xₜ
        where Σₜ = rolling covariance matrix of returns

    - Standardized innovation (unitless):
        εₜ = Rₜ / sqrt(σ²ₜ)

    - VaR in decimal units:
        VaRₜ (decimal) = - Quantile_α(εₜ) × sqrt(σ²ₜ)

    - VaR in monetary units:
        VaRₜ (money) = VaRₜ (decimal) × ∑ xₜ

    - 1-step-ahead forecast:
        VaRₜ₊₁ = - Quantile_α(ε) × sqrt(xₜ₊₁ᵀ Σₜ xₜ₊₁) × ∑ xₜ₊₁

    Parameters:
    - x_matrix (pd.DataFrame):
        Monetary positions per asset (shape: T × N), used to derive returns and portfolio weights.

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - window_size (int):
        Rolling window length used to compute sample covariance Σₜ.

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': portfolio returns (in decimal, e.g., 0.01 = 1%)
        - 'Volatility': conditional portfolio volatility σₜ (decimal units)
        - 'Innovations': standardized residuals εₜ (unitless)
        - 'VaR': VaR as percentage of wealth (decimal)
        - 'VaR Monetary': VaR in monetary units (same currency as xₜ)
        - 'VaR Violation': True if loss > VaR on that day

    - next_day_var_monetary (float):
        Forecasted 1-step-ahead VaR in monetary units.

    Notes:
    - This method does not assume normality.
    - The use of monetary positions allows exact accounting for exposure and leverage.
    - Innovations are used to empirically estimate the left-tail quantile.
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

        # Normalize to percentage of portfolio value
        portfolio_value = x_matrix.sum(axis=1).iloc[t]
        portfolio_volatility /= portfolio_value

        volatilities.append(portfolio_volatility)
        innovations.append(portfolio_returns.iloc[t] / portfolio_volatility)
        valid_index.append(returns.index[t])

    # Assemble result DataFrame
    result_data = pd.DataFrame({
        "Returns": portfolio_returns.loc[valid_index],
        "Volatility": volatilities,
        "Innovations": innovations
    }, index=valid_index)

    # Empirical quantile for VaR
    empirical_quantile = np.percentile(result_data["Innovations"].dropna(), 100 * (1 - confidence_level))

    # VaR in decimal units
    result_data["VaR"] = -result_data["Volatility"] * empirical_quantile

    # VaR in monetary units
    portfolio_value_series = x_matrix.sum(axis=1).loc[valid_index]
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series

    # VaR violations
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series < -result_data["VaR Monetary"]

    # Forecast next-day VaR (monetary)
    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = rolling_covs.loc[returns.index[-1]].values
    next_day_variance = float(x_last.T @ sigma_last @ x_last)
    next_day_vol = np.sqrt(next_day_variance)
    latest_portfolio_value = x_matrix.sum(axis=1).iloc[-1]
    next_day_vol /= latest_portfolio_value  # normalize
    next_day_var_monetary = abs(empirical_quantile * next_day_vol * latest_portfolio_value)

    return result_data, next_day_var_monetary


# ----------------------------------------------------------
# RiskMetrics Correlation VaR 
#----------------------------------------------------------
def var_corr_riskmetrics(x_matrix, confidence_level=0.99, lambda_decay=0.94):
    """
    RiskMetrics VaR Estimation (Empirical, Non-Parametric).

    Estimate portfolio Value-at-Risk (VaR) using an exponentially weighted moving average (EWMA)
    of the asset return covariance matrix and empirical quantiles of standardized portfolio innovations.

    Description:
    - This method does not assume a specific distribution for returns.
    - Volatility is modeled via EWMA of the sample return covariance matrix.
    - Innovations are computed by dividing portfolio returns by the time-varying volatility.
    - VaR is computed in both decimal form (% of portfolio value) and monetary units.

    Formulas:
    - Portfolio return:
        Rₜ = (xₜ · rₜ) / (∑ xₜ)
        where rₜ is the vector of asset returns (decimal) and xₜ the monetary positions.

    - EWMA covariance matrix update:
        Σₜ = λ Σₜ₋₁ + (1 - λ) rₜ rₜᵀ

    - Portfolio variance:
        σ²ₜ = xₜᵀ Σₜ xₜ

    - Standardized innovation:
        εₜ = Rₜ / sqrt(σ²ₜ)

    - VaR (decimal, % of portfolio value):
        VaRₜ = - Quantile(ε) × sqrt(σ²ₜ)

    - VaR (monetary):
        VaRₜ = VaR (decimal) × ∑ xₜ

    Parameters:
    - x_matrix (pd.DataFrame):
        Monetary positions per asset (shape: T × N).

    - confidence_level (float):
        VaR confidence level (e.g., 0.99 for 99% VaR).

    - lambda_decay (float):
        EWMA decay factor (default = 0.94).

    Returns:
    - result_data (pd.DataFrame):
        Time-indexed DataFrame with:
            - 'Returns': portfolio returns (decimal)
            - 'Volatility': time-varying volatility (decimal)
            - 'Innovations': standardized shocks
            - 'VaR': VaR as decimal loss threshold
            - 'VaR Monetary': VaR in monetary units
            - 'VaR Violation': boolean flag for exceedances

    - next_day_var (float):
        1-step ahead forecasted VaR in monetary units using latest x and EWMA Σ.

    Notes:
    - All returns and volatility are in decimal units (e.g., 0.01 = 1%).
    - This method empirically estimates tail quantiles using historical standardized residuals.
    - Output includes both percentage-based and monetary VaR.
    """
    # Compute returns
    returns = x_matrix.pct_change().dropna()
    portfolio_returns = (x_matrix * returns).sum(axis=1) / x_matrix.sum(axis=1)

    ewma_cov = returns.cov().values
    cov_matrices = []

    for t in range(returns.shape[0]):
        r_t = returns.iloc[t].values.reshape(-1, 1)
        ewma_cov = lambda_decay * ewma_cov + (1 - lambda_decay) * (r_t @ r_t.T)
        cov_matrices.append(ewma_cov.copy())

    x_matrix = x_matrix.loc[returns.index]

    volatilities = []
    innovations = []

    for t, sigma in enumerate(cov_matrices):
        x_t = x_matrix.iloc[t].values.reshape(-1, 1)
        portfolio_variance = float(x_t.T @ sigma @ x_t)
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Normalize volatility to percentage of portfolio value
        portfolio_value = x_matrix.sum(axis=1).iloc[t]
        portfolio_volatility /= portfolio_value

        volatilities.append(portfolio_volatility)
        innovations.append(portfolio_returns.iloc[t] / portfolio_volatility)

    result_data = pd.DataFrame({
        "Returns": portfolio_returns,
        "Volatility": volatilities,
        "Innovations": innovations
    }, index=returns.index)

    empirical_quantile = np.percentile(result_data["Innovations"].dropna(), 100 * (1 - confidence_level))

    # Decimal VaR
    result_data["VaR"] = -result_data["Volatility"] * empirical_quantile

    # Monetary VaR
    portfolio_value_series = x_matrix.sum(axis=1)
    result_data["VaR Monetary"] = result_data["VaR"] * portfolio_value_series

    # Violation detection
    result_data["VaR Violation"] = result_data["Returns"] * portfolio_value_series < -result_data["VaR Monetary"]

    # One-step-ahead forecast
    x_last = x_matrix.iloc[-1].values.reshape(-1, 1)
    sigma_last = cov_matrices[-1]
    next_day_vol = np.sqrt(float(x_last.T @ sigma_last @ x_last))
    latest_portfolio_value = x_matrix.sum(axis=1).iloc[-1]
    next_day_vol /= latest_portfolio_value  # Normalize
    next_day_var_monetary = abs(empirical_quantile * next_day_vol * latest_portfolio_value)

    return result_data, next_day_var_monetary