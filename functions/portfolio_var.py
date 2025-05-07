#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
from IPython.display import display

#################################################
# Note: double check all formulas and inputs (x)
#       check index stuff
#       limit displaying table (too long)
#################################################
# Note2: backtesting VaR AN? good idea or not?
#################################################
# Note3: when using VCV and x and x' (with z from normal)
# the VaR is already in monetary terms
#################################################

#----------------------------------------------------------
# Asset Normal VaR with Diversification Table
#----------------------------------------------------------
def var_asset_normal(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    display_table=False
):
    """
    Asset-Normal VaR Estimation with Diversification Metrics.

    Computes both the diversified and undiversified Value-at-Risk (VaR) for a portfolio 
    assuming normally distributed returns, and derives the diversification benefit as the difference.

    Description:
    - Diversified VaR: uses the full covariance matrix Σ to account for correlations among assets.
    - Undiversified VaR: assumes perfect positive correlation (ρ = 1) and sums individual asset VaRs.
    - The difference between the two reflects the gain from diversification.

    Formulas:
    - Diversified VaR at time t:
        VaRₜ = | z × sqrt( xₜᵀ Σ xₜ ) × sqrt(holding_period) |
    - Undiversified VaR at time t:
        VaRₜ = | z × sum(σᵢ × xᵢ) × sqrt(holding_period) |
    where:
        - xₜ: portfolio position vector at time t
        - Σ: covariance matrix of asset returns
        - σᵢ: standard deviation of asset i
        - z: quantile of the standard normal distribution

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary positions (shape: T × N),
        where T = number of time steps and N = number of assets.

    - confidence_level (float, optional):
        Confidence level for VaR (default = 0.99).

    - holding_period (int, optional):
        Holding period in days over which risk is measured (default = 1).

    - display_table (bool, optional):
        If True, displays the result as a formatted table (only in interactive environments).

    Returns:
    - pd.DataFrame:
        Table indexed by date with three columns:
        ['Diversified_VaR', 'Undiversified_VaR', 'Diversification_Benefit']

    Notes:
    - Returns are internally computed from positions via percentage change.
    - Scaling by sqrt(holding_period) assumes independent daily returns.
    - Clean date strings (YYYY-MM-DD) are used as the index for clarity.
    """
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()

    sigma_matrix = returns_data.cov().values
    z_score = norm.ppf(1 - confidence_level)

    positions = position_data.loc[returns_data.index].values
    div_list = []
    undiv_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)

        # Diversified VaR
        variance = float(x.T @ sigma_matrix @ x)
        std_dev = np.sqrt(variance)
        var_div = np.abs(z_score * std_dev * np.sqrt(holding_period))
        div_list.append(var_div)

        # Undiversified VaR
        std_devs = np.sqrt(np.diag(sigma_matrix)).reshape(-1, 1)
        var_i = z_score * std_devs * x
        var_undiv = np.abs(var_i.sum()) * np.sqrt(holding_period)
        undiv_list.append(var_undiv)

    # Format clean index
    clean_index = returns_data.index.strftime("%Y-%m-%d")

    result = pd.DataFrame({
        "Diversified_VaR": div_list,
        "Undiversified_VaR": undiv_list,
        "Diversification_Benefit": np.array(undiv_list) - np.array(div_list)
    }, index=clean_index)

    if display_table:
        try:
            display(
                result.style
                    .format("{:.2f}")
                    .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
            )
        except ImportError:
            print("Table display is available only in interactive environments.")

    return result


#----------------------------------------------------------
# Marginal VaR (Delta VaR per Asset)
#----------------------------------------------------------
def marginal_var(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    display_table=False
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
        Confidence level for VaR (default = 0.99).

    - holding_period (int, optional): 
        Holding period in days (default = 1).

    - display_table (bool, optional): 
        If True, displays a styled table (default = False).

    Returns:
    - pd.DataFrame: 
        Time series of Marginal VaRs (T × N), each entry giving the marginal risk contribution 
        of asset i on day t, expressed in monetary units.

    Notes:
    - Relies on diversified VaR logic (full covariance matrix Σ used).
    - If total portfolio variance is ~0, marginal VaRs are set to 0.
    - Scaling by sqrt(holding_period) assumes independent daily returns.
    """
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    # Use only diversified VaR from the updated function
    portfolio_var_df = var_asset_normal(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        display_table=False
    )
    portfolio_var_series = portfolio_var_df["Diversified_VaR"]

    positions = position_data.loc[returns_data.index].values
    marginal_var_list = []

    for xt, var_t in zip(positions, portfolio_var_series.values):
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            delta_var = np.zeros_like(x.flatten())
        else:
            beta_vector = (sigma_matrix @ x).flatten() / variance
            delta_var = var_t * beta_vector

        marginal_var_list.append(delta_var)

    result = pd.DataFrame(
        marginal_var_list,
        index=returns_data.index.strftime("%Y-%m-%d"),
        columns=position_data.columns
    )

    if display_table:
        display(
            result.style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    return result


#----------------------------------------------------------
# Component VaR (using marginal VaR)
#----------------------------------------------------------
def component_var(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    display_table=False
):
    """
    Component Value-at-Risk (VaR) Estimation.

    Computes the contribution of each asset to the total portfolio VaR
    using the Euler decomposition via marginal VaR.

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (T × N)

    - confidence_level (float):
        Confidence level for VaR (default = 0.99)

    - holding_period (int, optional):
        VaR horizon in days (default = 1)

    - display_table (bool, optional):
        If True, displays a styled table (default = False)

    Returns:
    - pd.DataFrame:
        Time series of Component VaRs (T × N), in monetary units
    """
    position_data = pd.DataFrame(position_data)

    # Compute Marginal VaR
    marginal_df = marginal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        display_table=False
    )

    # Safely reformat index without mutating original input
    position_data_aligned = position_data.copy()
    if isinstance(position_data_aligned.index[0], pd.Timestamp):
        position_data_aligned.index = position_data_aligned.index.strftime("%Y-%m-%d")

    # Align rows
    aligned_positions = position_data_aligned.loc[marginal_df.index]

    # Compute Component VaR
    component_df = aligned_positions * marginal_df

    if display_table:
        display(
            component_df.style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    return component_df


#----------------------------------------------------------
# Relative Component VaR
#----------------------------------------------------------
def relative_component_var(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    display_table=False
):
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
        Confidence level for VaR (default = 0.99).

    - holding_period (int, optional):
        VaR horizon in days (default = 1).

    - display_table (bool, optional):
        If True, displays a styled table (default = False).

    Returns:
    - pd.DataFrame:
        Time series of Relative Component VaRs per asset (T × N), with values between 0 and 1.

    Notes:
    - Sum of relative components across assets equals 1 at each point in time.
    - Computation is based on Component VaR divided by Total Diversified VaR.
    """
    position_data = pd.DataFrame(position_data)

    # Compute Component VaR and total portfolio VaR
    cvar_df = component_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        display_table=False
    )

    var_df = var_asset_normal(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        display_table=False
    )

    # Extract just the diversified VaR column
    var_series = var_df["Diversified_VaR"]

    # Compute relative contributions
    rcvar_df = cvar_df.div(var_series, axis=0)

    if display_table:
        display(
            rcvar_df.style
                .format("{:.2%}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    return rcvar_df


#----------------------------------------------------------
# Incremental VaR
#----------------------------------------------------------
def incremental_var(
    position_data,
    change_vector,
    confidence_level=0.99,
    holding_period=1,
    display_table=False
):
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
        Confidence level for VaR (default = 0.99).

    - holding_period (int, optional):
        VaR horizon in days (default = 1).

    - display_table (bool, optional):
        If True, displays a styled table (default = False).

    Returns:
    - pd.Series:
        Time series of Incremental VaR estimates (one value per day).

    Notes:
    - Positive IVaR means the change increases portfolio risk.
    - Negative IVaR means the change reduces portfolio risk.
    """
    position_data = pd.DataFrame(position_data)

    # Get Marginal VaR
    marginal_df = marginal_var(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        display_table=False
    )

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    # Compute Incremental VaR
    ivar_series = marginal_df @ a  # dot product over columns

    # No need to change index; marginal_df already returns a string-based index
    if display_table:
        display(
            ivar_series.to_frame("Incremental_VaR")
                .style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    return ivar_series
