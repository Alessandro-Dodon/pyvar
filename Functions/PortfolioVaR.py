#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings

#################################################
# Note: double check all formulas and inputs (x)
#       add caller (meta) function?
#################################################

#----------------------------------------------------------
# Asset Normal VaR and Undiversified VaR
#----------------------------------------------------------
def var_asset_normal(
    position_data,
    confidence_level,
    holding_period=1,
    undiversified=False
):
    """
    Asset-Normal VaR Estimation.

    Assumes that asset returns are normally distributed, implying that
    portfolio returns—being linear combinations—are also normally distributed.

    Description:
    - Diversified VaR: accounts for covariances among assets.
    - Undiversified VaR (ρ = 1): assumes full positive correlation between assets, summing individual VaRs.

    Formulas:
    - Diversified VaR at time t:
        VaRₜ = | z × sqrt( xₜᵀ Σ xₜ ) × sqrt(holding_period) |
    - Undiversified VaR at time t:
        VaRₜ = | z × sum(σᵢ × xᵢ) × sqrt(holding_period) |
    where:
        - xₜ: position vector at time t
        - Σ: covariance matrix of returns
        - σᵢ: standard deviation of asset i
        - z: critical value from the standard normal distribution

    Parameters:
    - position_data (pd.DataFrame or np.ndarray): 
        Time series of monetary positions (shape: T × N),
        where T = number of days and N = number of assets.

    - confidence_level (float): 
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional): 
        Holding period in days (default = 1).

    - undiversified (bool, optional): 
        If True, compute undiversified VaR (full correlation assumed).

    Returns:
    - pd.Series: 
        Daily time series of VaR estimates, indexed by date (aligned to returns).

    Notes:
    - Returns are calculated from positions using percentage changes.
    - Undiversified VaR corresponds to the sum of asset-level VaRs under full correlation (ρ = 1).
    - Scaling by sqrt(holding_period) assumes independent daily returns.
    """
    # Convert to DataFrame if needed
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()

    sigma_matrix = returns_data.cov().values
    z_score = norm.ppf(1 - confidence_level)

    positions = position_data.loc[returns_data.index].values
    var_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)

        if undiversified:
            std_devs = np.sqrt(np.diag(sigma_matrix)).reshape(-1, 1)
            var_i = z_score * std_devs * x
            var_t = np.abs(var_i.sum()) * np.sqrt(holding_period)
        else:
            variance = float(x.T @ sigma_matrix @ x)
            std_dev = np.sqrt(variance)
            var_t = np.abs(z_score * std_dev * np.sqrt(holding_period))

        var_list.append(var_t)

    return pd.Series(var_list, index=returns_data.index)


#----------------------------------------------------------
# Marginal VaR (Delta VaR per Asset)
#----------------------------------------------------------
def marginal_var(
    position_data,
    confidence_level,
    holding_period=1
):
    """
    Marginal Value-at-Risk (Marginal VaR) Estimation.

    Compute Marginal VaR (ΔVaR) for each asset in a portfolio at each time step.

    Description:
    - Marginal VaR measures the sensitivity of total portfolio VaR to small changes in each asset's position.
    - It is proportional to each asset's beta relative to portfolio risk:
        ΔVaRᵢ = ( ∂VaR / ∂xᵢ ) ≈ βᵢ × Portfolio VaR
    where:
        - βᵢ = (Σx)ᵢ / (x'Σx)
        - x: portfolio holdings vector
        - Σ: covariance matrix of returns

    Formulas:
    - Portfolio diversified VaR:
        VaRₜ = | z × sqrt( xₜᵀ Σ xₜ ) × sqrt(holding_period) |
    - Marginal VaR for asset i at time t:
        ΔVaRᵢₜ = VaRₜ × ( (Σ xₜ)ᵢ / (xₜᵀ Σ xₜ) )

    Parameters:
    - position_data (pd.DataFrame or np.ndarray): 
        Time series of monetary positions (shape: T × N), with T = days, N = assets.

    - confidence_level (float): 
        Confidence level for VaR (e.g., 0.99).

    - holding_period (int, optional): 
        Holding period in days (default = 1).

    Returns:
    - pd.DataFrame: 
        Time series of Marginal VaRs (T × N), each entry giving the marginal risk contribution 
        of asset i on day t, expressed in monetary units.

    Notes:
    - Relies on diversified VaR logic (full covariance matrix Σ used).
    - If total portfolio variance is ~0, marginal VaRs are set to 0.
    - Scaling by sqrt(holding_period) assumes independent daily returns.
    """
    # Ensure proper format
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    # Get AN VaR from the existing function
    portfolio_var_series = var_asset_normal(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        undiversified=False
    )

    # Align data
    positions = position_data.loc[returns_data.index].values
    marginal_var_list = []

    for xt, var_t in zip(positions, portfolio_var_series.values):
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            delta_var = np.zeros_like(x.flatten())
        else:
            # Compute beta weights (Σx / x'Σx)
            beta_vector = (sigma_matrix @ x).flatten() / variance
            delta_var = var_t * beta_vector  # Monetary marginal VaR

        marginal_var_list.append(delta_var)

    return pd.DataFrame(
        marginal_var_list,
        index=returns_data.index,
        columns=position_data.columns
    )


#----------------------------------------------------------
# Component VaR (using marginal VaR)
#----------------------------------------------------------
def component_var(position_data, confidence_level, holding_period=1):
    """
    Component VaR Estimation.

    Compute the contribution of each asset to total portfolio VaR using:
        ComponentVaRᵢ = xᵢ × ∂VaR/∂xᵢ = xᵢ × ΔVaRᵢ

    Description:
    - Relies on Marginal VaR (ΔVaR) estimates, which capture the sensitivity of portfolio VaR to small changes in each position.
    - Each component reflects the individual risk contribution in monetary units.

    Formulas:
    - Component VaR at time t:
        CVaRᵢₜ = xᵢₜ × ΔVaRᵢₜ
    where:
        - xᵢₜ: monetary position in asset i at time t
        - ΔVaRᵢₜ: marginal VaR of asset i at time t

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (shape: T × N),
        where T = number of days and N = number of assets.

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional):
        VaR horizon in days (default = 1).

    Returns:
    - pd.DataFrame:
        Time series of Component VaRs per asset (T × N), in monetary units.

    Notes:
    - Sum of Component VaRs across assets equals total diversified portfolio VaR.
    - Consistent with the Euler decomposition of VaR under normality and linearity.
    """
    position_data = pd.DataFrame(position_data)
    marginal_df = marginal_var(position_data, confidence_level, holding_period)
    component_df = position_data.loc[marginal_df.index] * marginal_df
    return component_df


#----------------------------------------------------------
# Relative Component VaR
#----------------------------------------------------------
def relative_component_var(position_data, confidence_level, holding_period=1):
    """
    Relative Component VaR Estimation.

    Compute the proportionate contribution of each asset to total portfolio VaR:
        RelativeComponentVaRᵢ = ComponentVaRᵢ / TotalVaR

    Description:
    - Measures how much each asset contributes to overall portfolio risk on a relative basis (percentage share).
    - Useful for identifying risk concentration across assets.

    Formulas:
    - Relative Component VaR at time t:
        RCVaRᵢₜ = CVaRᵢₜ / VaRₜ
    where:
        - CVaRᵢₜ: Component VaR of asset i at time t
        - VaRₜ: total diversified portfolio VaR at time t

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (shape: T × N),
        where T = number of days and N = number of assets.

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional):
        VaR horizon in days (default = 1).

    Returns:
    - pd.DataFrame:
        Time series of Relative Component VaRs per asset (T × N), with values between 0 and 1.

    Notes:
    - Sum of relative components across assets equals 1 at each point in time.
    - Computation is based on Component VaR divided by Total Diversified VaR.
    """
    position_data = pd.DataFrame(position_data)
    
    # Get components
    cvar_df = component_var(position_data, confidence_level, holding_period)
    var_series = var_asset_normal(position_data, confidence_level, holding_period)
    
    # Divide each row by the total VaR of that day
    rcvar_df = cvar_df.div(var_series, axis=0)
    
    return rcvar_df


#----------------------------------------------------------
# Incremental VaR
#----------------------------------------------------------
def incremental_var(position_data, change_vector, confidence_level, holding_period=1):
    """
    Incremental VaR Estimation.

    Compute the impact on total portfolio VaR of a change in positions, using Marginal VaR.

    Description:
    - Measures how a specific change in portfolio composition affects overall risk.
    - First-order approximation based on the gradient (Marginal VaR).

    Formulas:
    - Incremental VaR at time t:
        IVaRₜ = ΔVaRₜ' × a
    where:
        - ΔVaRₜ: vector of marginal VaRs at time t
        - a: change vector (same dimension as the number of assets)

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (shape: T × N),
        where T = number of days and N = number of assets.

    - change_vector (list or np.ndarray):
        Vector specifying changes in holdings (shape: N,).

    - confidence_level (float):
        Confidence level for VaR (e.g., 0.99 for 99% VaR).

    - holding_period (int, optional):
        VaR horizon in days (default = 1).

    Returns:
    - pd.Series:
        Time series of Incremental VaR estimates (one value per day).

    Notes:
    - Positive IVaR means the change increases portfolio risk.
    - Negative IVaR means the change reduces portfolio risk.
    """
    position_data = pd.DataFrame(position_data)
    marginal_df = marginal_var(position_data, confidence_level, holding_period)

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    ivar_series = marginal_df @ a
    return ivar_series



