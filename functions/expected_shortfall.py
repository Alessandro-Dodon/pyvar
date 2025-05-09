#----------------------------------------------------------
# Packages
#----------------------------------------------------------
from scipy.stats import norm, t, gennorm
import numpy as np
import pandas as pd
import warnings # for new ES functions
from IPython.display import display # for new ES functions

#################################################
# Note: double check all formulas 
#       and ES for vol should also
#       return the ES for the next day? 
#################################################

#----------------------------------------------------------
# Historical Expected Shortfall (Tail Mean)
#----------------------------------------------------------
def compute_es_historical(result_data, wealth=None): 
    """
    Estimate Expected Shortfall (ES) using historical returns below the VaR threshold.

    This function computes historical ES by averaging the portfolio returns 
    that fall below the historical VaR cutoff already present in the input data. 
    Assumes the 'VaR' column is constant and derived from a specified confidence level.

    Parameters:
    - result_data (pd.DataFrame):
        Must contain 'Returns' and 'VaR' columns.
        Assumes 'VaR' is fixed across time and derived from historical quantiles.
        All values should be in decimal format (e.g., 0.01 = 1%).

    - wealth (float, optional):
        If provided, ES is also returned in monetary units using this portfolio value.

    Returns:
    - result_data (pd.DataFrame):
        Original DataFrame extended with:
            - 'ES': constant Expected Shortfall (decimal loss).
            - 'ES_monetary' (optional): if wealth is provided.

    - es_estimate (float):
        Scalar ES estimate (positive value, e.g., 0.015 for 1.5%).
    """
    var_threshold = result_data["VaR"].iloc[0]
    tail_returns = result_data["Returns"][result_data["Returns"] < -var_threshold]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = tail_returns.mean()

    es_series = pd.Series(-es_value, index=result_data.index)
    result_data["ES"] = es_series

    es_estimate = -es_value

    if wealth is not None:
        result_data["ES_monetary"] = es_series * wealth
        es_estimate *= wealth

    return result_data, es_estimate


#----------------------------------------------------------
# Parametric Expected Shortfall (Normal and t only)
#----------------------------------------------------------
def compute_es_parametric(
    result_data,
    returns,
    confidence_level,
    holding_period=1,
    distribution="normal",
    wealth=None
):   
    """
    Estimate Expected Shortfall (ES) using a parametric distribution.

    Supports Normal and Student-t distributions. The ES is computed as the 
    expected return conditional on exceeding the VaR threshold, scaled for 
    the given holding period.

    Parameters:
    - result_data (pd.DataFrame): 
        DataFrame to be extended with ES estimates.
    - returns (pd.Series): 
        Return series in decimal format.
    - confidence_level (float): 
        Confidence level for ES (e.g., 0.99).
    - holding_period (int): 
        Holding period in days (default = 1).
    - distribution (str): 
        Distribution to use, "normal" or "t".
    - wealth (float, optional): 
        If provided, ES is also returned in monetary units.

    Returns:
    - result_data (pd.DataFrame): 
        Updated with 'ES' (decimal) and optionally 'ES_monetary'.
    - es_estimate (float): 
        Scalar ES value (positive loss magnitude).
    """
    returns_clean = returns.dropna()
    alpha = confidence_level

    if distribution == "normal":
        std_dev = returns_clean.std()
        z = norm.ppf(alpha)
        es_raw = std_dev * norm.pdf(z) / (1 - alpha)
        es_value = es_raw * np.sqrt(holding_period)

    elif distribution == "t":
        df, loc, scale = t.fit(returns_clean)
        t_alpha = t.ppf(alpha, df)
        pdf_val = t.pdf(t_alpha, df)
        factor = (df + t_alpha**2) / (df - 1)
        es_raw = scale * pdf_val * factor / (1 - alpha)
        es_value = es_raw * np.sqrt(holding_period)

    else:
        raise ValueError("Supported distributions: 'normal', 't'")

    result_data["ES"] = pd.Series(es_value, index=result_data.index)
    es_estimate = es_value

    if wealth is not None:
        result_data["ES_monetary"] = result_data["ES"] * wealth
        es_estimate *= wealth

    return result_data, es_estimate


#----------------------------------------------------------
# Expected Shortfall Volatility
#----------------------------------------------------------
def compute_expected_shortfall_volatility(data, confidence_level, subset=None, wealth=None):
    """
    Estimate Expected Shortfall (ES) dynamically using empirical innovations and conditional volatility.

    Computes time-varying ES by averaging standardized residuals (innovations) in the left tail 
    and scaling by model-implied volatility. Useful for volatility models that generate both 
    residuals and conditional standard deviations.

    Parameters:
    - data (pd.DataFrame): 
        Must contain 'Innovations' and 'Volatility' columns.
    - confidence_level (float): 
        Confidence level for ES (e.g., 0.99).
    - subset (tuple, optional): 
        (start_date, end_date) range for computing the empirical tail mean.
    - wealth (float, optional): 
        If provided, returns ES in monetary units.

    Returns:
    - data (pd.DataFrame): 
        Original DataFrame extended with:
        - 'ES': expected shortfall in decimal units.
        - 'ES_monetary' (if wealth is provided): expected shortfall in monetary units.
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")
    
    subset_data = data.copy()
    if subset is not None:
        subset_data = data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"].dropna(), 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    data["ES"] = -data["Volatility"] * tail_mean

    if wealth is not None:
        data["ES_monetary"] = data["ES"] * wealth

    return data


#----------------------------------------------------------
# Expected Shortfall for Correlation Models (Parametric)
#----------------------------------------------------------
def compute_expected_shortfall_corr(data, confidence_level=0.99):
    """
    Estimate Expected Shortfall (ES) using a parametric normal formula.

    Computes time-varying ES under the normality assumption using 
    conditional portfolio volatility. ES is scaled using portfolio 
    value inferred from existing VaR columns.

    Parameters:
    - data (pd.DataFrame): 
        Must contain:
        - 'Volatility': portfolio volatility (decimal units)
        - 'VaR': VaR as % of portfolio value (decimal)
        - 'VaR Monetary': VaR in monetary units
    - confidence_level (float): 
        ES confidence level (e.g., 0.99).

    Returns:
    - data (pd.DataFrame): 
        Extended with:
        - 'ES': expected shortfall in decimal units
        - 'ES Monetary': expected shortfall in monetary units
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


#----------------------------------------------------------
# Marginal Expected Shortfall (ES)
#----------------------------------------------------------
def marginal_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Marginal Expected Shortfall (ES) for each asset in a portfolio.

    Computes marginal ES using Euler decomposition under a normal distribution 
    assumption. Reflects each asset's sensitivity to portfolio tail risk.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of marginal ES (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)
    if position_data.shape[0] < 2:
        warnings.warn("At least two rows of data are required.")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    alpha = confidence_level
    z = norm.ppf(alpha)
    es_scaling = norm.pdf(z) / (1 - alpha)

    positions = position_data.loc[returns_data.index].values
    marginal_es_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            marginal = np.zeros_like(x.flatten())
        else:
            std_dev = np.sqrt(variance)
            portfolio_es = es_scaling * std_dev * np.sqrt(holding_period)
            beta_vector = (sigma_matrix @ x).flatten() / variance
            marginal = portfolio_es * beta_vector

        marginal_es_list.append(marginal)

    return pd.DataFrame(
        marginal_es_list,
        index=returns_data.index.strftime("%Y-%m-%d"),
        columns=position_data.columns
    )


#----------------------------------------------------------
# Component Expected Shortfall (ES)
#----------------------------------------------------------
def component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Component Expected Shortfall (ES) for each asset in a portfolio.

    Computes component ES using Euler decomposition: 
    marginal ES × monetary position.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of component ES (T × N), in monetary units.
    """
    position_data = pd.DataFrame(position_data)

    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    if isinstance(position_data.index[0], pd.Timestamp):
        position_data.index = position_data.index.strftime("%Y-%m-%d")

    aligned_positions = position_data.loc[marginal_df.index]
    component_df = aligned_positions * marginal_df

    return component_df


#----------------------------------------------------------
# Relative Component Expected Shortfall (ES)
#----------------------------------------------------------
def relative_component_es(position_data, confidence_level=0.99, holding_period=1):
    """
    Estimate Relative Component Expected Shortfall (ES) for each asset.

    Computes the share of total ES each asset contributes at each time step:
    relative ES = component ES / total ES.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions per asset (T × N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Holding period in days (default: 1).

    Returns:
    - pd.DataFrame: Time series of relative component ES (T × N), values sum to 1.
    """
    position_data = pd.DataFrame(position_data)

    component_df = component_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    total_es = component_df.sum(axis=1)
    relative_df = component_df.div(total_es, axis=0)

    return relative_df


#----------------------------------------------------------
# Incremental Expected Shortfall (ES)
#----------------------------------------------------------
def incremental_es(position_data, change_vector, confidence_level=0.99, holding_period=1):
    """
    Estimate Incremental Expected Shortfall (ES) for a portfolio.

    Computes how a change in positions affects total ES using Marginal ES 
    and a first-order approximation.

    Parameters:
    - position_data (pd.DataFrame): Monetary positions (T × N).
    - change_vector (list or np.ndarray): Change in positions (length N).
    - confidence_level (float): Confidence level for ES (e.g., 0.99).
    - holding_period (int): Horizon in days (default: 1).

    Returns:
    - pd.Series: Time series of Incremental ES values in monetary units.
    """
    position_data = pd.DataFrame(position_data)

    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period
    )

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    ies_series = marginal_df @ a
    return ies_series
