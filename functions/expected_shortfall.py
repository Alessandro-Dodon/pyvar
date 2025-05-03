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
#       add a failsafe when user tries to scale
#       ES with dynamic corr methods or portfolio
#       methods (avoid double scaling)
#################################################
# Note2: write a new ES function for the corr models
#       that get both ES in decimals (plotting) and
#       monetary units (scaling), then the caller
#       MUST prevent to use wealth with this function
#################################################

#----------------------------------------------------------
# Historical Expected Shortfall (Tail Mean)
#----------------------------------------------------------
def compute_es_historical(result_data, confidence_level):
    """
    Historical Expected Shortfall Estimation (Tail Mean Method).

    Compute Expected Shortfall (ES) non-parametrically as the average of returns 
    that fall below the Value-at-Risk (VaR) threshold.

    Description:
    - ES represents the expected loss given that the return exceeds the VaR threshold.
    - Computed by averaging all returns lower than -VaR (left tail of the distribution).

    Formula:
    - Expected Shortfall (ES):
        ES = - E[ Return | Return < -VaR ]

    Parameters:
    - result_data (pd.DataFrame):
        Must contain 'Returns' and 'VaR' columns.
        Assumes 'VaR' is constant over time (i.e., historical VaR).
        Returns and VaR must be in decimal format (e.g., 0.01 = 1%).

    - confidence_level (float):
        Confidence level for VaR/ES (e.g., 0.99 for 99%).

    Returns:
    - result_data (pd.DataFrame):
        DataFrame with a new column 'ES' (constant series, in decimals).

    - es_estimate (float):
        Scalar ES estimate (positive decimal loss magnitude, e.g., 0.015 for 1.5%).

    Notes:
    - ES is expressed as a positive number, representing the expected loss.
    - Returns and VaR must be expressed in decimal format throughout.
    """
    var_threshold = result_data["VaR"].iloc[0]
    tail_returns = result_data["Returns"][result_data["Returns"] < -var_threshold]

    if len(tail_returns) == 0:
        es_value = np.nan
    else:
        es_value = tail_returns.mean()

    result_data["ES"] = pd.Series(es_value * -1, index=result_data.index)
    es_estimate = result_data["ES"].iloc[0]
    return result_data, es_estimate


#----------------------------------------------------------
# Parametric Expected Shortfall (Normal and t only)
#----------------------------------------------------------
def compute_es_parametric(result_data, returns, confidence_level, holding_period=1, distribution="normal"):
    """
    Parametric Expected Shortfall Estimation (Normal or Student-t).

    Estimate Expected Shortfall (ES) analytically under the assumption 
    that returns follow either a Normal or Student-t distribution.

    Description:
    - ES represents the expected loss conditional on returns exceeding the VaR threshold.
    - Computed from the assumed distribution's PDF and quantile function.

    Formulas:
    - Normal:
        ES = σ × φ(z) / (1 - α)
    - Student-t:
        ES = scale × pdf(tₐ, df) × (df + tₐ²) / [(df - 1)(1 - α)]

    where:
        - σ = standard deviation of returns
        - φ(z) = standard normal PDF evaluated at z
        - tₐ = Student-t quantile at level α
        - pdf(tₐ, df) = Student-t PDF at tₐ
        - α = confidence level (e.g., 0.99)

    Parameters:
    - result_data (pd.DataFrame): DataFrame to update with an 'ES' column (same length as returns).
    - returns (pd.Series): Return series used to estimate parameters. Must be in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR/ES (e.g., 0.99 for 99%).
    - holding_period (int): Holding period in days (default = 1). Assumes i.i.d. scaling via sqrt(T).
    - distribution (str): Distribution to use: "normal" or "t" (Student-t).

    Returns:
    - result_data (pd.DataFrame): Updated with constant 'ES' column (in decimals, e.g., 0.012 = 1.2%).
    - es_estimate (float): Scalar ES value, in decimal format (positive loss magnitude).

    Notes:
    - ES is returned as a positive number (expected loss).
    - Returns and ES are expressed in decimal units.
    - The estimate is scaled for multi-day holding periods assuming i.i.d. returns.
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

    return result_data, es_estimate


#----------------------------------------------------------
# Expected Shortfall Volatility
#----------------------------------------------------------
def compute_expected_shortfall_volatility(data, confidence_level, subset=None):
    """
    Volatility-Based Expected Shortfall Estimation (Empirical Innovations Method).

    Estimate Expected Shortfall (ES) dynamically over time using the empirical 
    average of standardized residuals (innovations) in the left tail, scaled by 
    the model-implied conditional volatility.

    Description:
    - ES is computed by averaging innovations below a quantile threshold, then scaling 
      this average (the tail mean) by the conditional volatility at each time point.
    - This allows capturing time-varying risk levels.

    Formulas:
    - Tail Mean:
        TailMean = mean(Innovationsₜ where Innovationsₜ < Quantile(Innovations, 1 - α))
    - Time-varying ES:
        ESₜ = - Volatilityₜ × TailMean

    Parameters:
    - data (pd.DataFrame): Must contain:
        - 'Innovations': standardized residuals (mean ≈ 0, std ≈ 1)
        - 'Volatility': conditional standard deviation (in decimal units)
    - confidence_level (float): Confidence level α for ES (e.g., 0.99 for 99%).
    - subset (tuple, optional): A date range (start_date, end_date) used to compute TailMean.

    Returns:
    - data (pd.DataFrame): Original DataFrame with an added 'ES' column (in decimals, e.g., 0.012 = 1.2%).

    Notes:
    - Returns and volatility are assumed in decimal format (e.g., 0.01 = 1%).
    - The ES column represents a positive loss magnitude at each time point.
    - The same tail mean is used for all rows unless a subset is specified.
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")
    
    subset_data = data.copy()
    if subset is not None:
        subset_data = data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"].dropna(), 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    data["ES"] = -data["Volatility"] * tail_mean
    return data


#----------------------------------------------------------
# Expected Shortfall Correlation Models
#----------------------------------------------------------
def compute_expected_shortfall_correlation(data, confidence_level=0.99, subset=None):
    """
    Correlation-Model-Based Expected Shortfall Estimation (Empirical Innovations Method).

    Estimate Expected Shortfall (ES) dynamically over time using standardized portfolio 
    innovations and conditional volatility from correlation-based models. 
    Also computes ES in monetary units by scaling the decimal ES using the current portfolio value.

    Description:
    - ES is computed by averaging standardized innovations below a quantile threshold 
      (left tail), then scaling this average (the tail mean) by the portfolio's conditional 
      volatility to obtain a decimal ES.
    - The monetary ES is then computed by scaling the decimal ES by the portfolio value, 
      inferred from existing VaR and VaR Monetary columns.

    Formulas:
    - Tail Mean (unitless):
        TailMean = mean(εₜ where εₜ < Quantile(ε, 1 - α))

    - Time-varying ES (decimal):
        ESₜ = - Volatilityₜ × TailMean

    - Time-varying ES (monetary):
        ESₜ (money) = ESₜ (decimal) × ∑ xₜ
        where ∑ xₜ is inferred from: ∑ xₜ = VaR Monetary / VaR

    Parameters:
    - data (pd.DataFrame):
        DataFrame must contain the following columns:
            - 'Innovations': standardized portfolio shocks (unitless)
            - 'Volatility': conditional portfolio volatility (in decimals, e.g., 0.01 = 1%)
            - 'VaR': decimal VaR (optional but required for monetary ES)
            - 'VaR Monetary': monetary VaR (optional but required for monetary ES)

    - confidence_level (float):
        Confidence level α for ES (e.g., 0.99 for 99%).

    - subset (tuple, optional):
        A date range (start_date, end_date) used to compute the empirical tail mean.
        If not specified, uses the entire sample.

    Returns:
    - data (pd.DataFrame):
        Original DataFrame extended with:
            - 'ES': expected shortfall in decimal (% of portfolio value)
            - 'ES Monetary': expected shortfall in monetary units

    Notes:
    - Assumes that 'VaR' and 'VaR Monetary' are available to back out ∑ xₜ.
    - The ES column expresses expected loss as a percentage of wealth.
    - The ES Monetary column expresses expected loss in money terms.
    - If VaR-related columns are missing, the function raises an error.
    """
    if "Innovations" not in data.columns or "Volatility" not in data.columns:
        raise ValueError("Data must contain 'Innovations' and 'Volatility' columns.")

    subset_data = data.copy()
    if subset is not None:
        subset_data = data.loc[subset[0]:subset[1]]

    threshold = np.percentile(subset_data["Innovations"].dropna(), 100 * (1 - confidence_level))
    tail_mean = subset_data["Innovations"][subset_data["Innovations"] < threshold].mean()

    # Decimal ES
    data["ES"] = -data["Volatility"] * tail_mean

    # Monetary ES
    if "VaR Monetary" in data.columns and "VaR" in data.columns:
        portfolio_value = data["VaR Monetary"] / data["VaR"]
        data["ES Monetary"] = data["ES"] * portfolio_value
    else:
        raise ValueError("To compute 'ES Monetary', data must include 'VaR' and 'VaR Monetary' columns.")

    return data


### Portfolio Metrics for ES


#----------------------------------------------------------
# Marginal Expected Shortfall 
#----------------------------------------------------------
def marginal_es(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    distribution="normal",
    display_table=False
):
    """
    Marginal Expected Shortfall (Marginal ES) Estimation.

    Compute Marginal ES (ΔES) for each asset in a portfolio at each time step.

    Description:
    - Marginal ES measures the sensitivity of total portfolio ES to small changes in each asset's position.
    - Based on Euler decomposition:
        ΔESᵢ = ( ∂ES / ∂xᵢ ) ≈ βᵢ × Portfolio ES
    where:
        - βᵢ = (Σx)ᵢ / (x'Σx)
        - x: portfolio holdings vector
        - Σ: covariance matrix of returns

    Formulas:
    - Normal:
        ES = σ × φ(z) / (1 - α)
    - Student-t:
        ES = scale × pdf(tₐ, df) × (df + tₐ²) / [(df - 1)(1 - α)]

    Parameters:
    - position_data (pd.DataFrame or np.ndarray): 
        Time series of monetary positions (shape: T × N)

    - confidence_level (float): 
        ES confidence level (e.g., 0.99)

    - holding_period (int, optional): 
        ES horizon in days (default = 1)

    - distribution (str, optional): 
        "normal" or "t" (default = "normal")

    - display_table (bool, optional): 
        If True, displays styled table (default = False)

    Returns:
    - pd.DataFrame: 
        Time series of Marginal ES values (T × N), in monetary units

    Notes:
    - Uses full covariance matrix Σ (diversified ES)
    - Automatically handles multi-day horizon with i.i.d. scaling
    """
    position_data = pd.DataFrame(position_data)

    if position_data.shape[0] < 2:
        warnings.warn("You must provide a time series of positions (at least 2 rows).")
        return None

    position_data = position_data.dropna()
    returns_data = position_data.pct_change().dropna()
    sigma_matrix = returns_data.cov().values

    alpha = confidence_level
    z = norm.ppf(alpha)

    if distribution == "normal":
        es_scaling = norm.pdf(z) / (1 - alpha)

    elif distribution == "t":
        df, loc, scale = t.fit(returns_data.values.flatten())
        t_alpha = t.ppf(alpha, df)
        pdf_val = t.pdf(t_alpha, df)
        factor = (df + t_alpha**2) / (df - 1)
        es_scaling = pdf_val * factor / (1 - alpha)

    else:
        raise ValueError("Supported distributions: 'normal', 't'")

    positions = position_data.loc[returns_data.index].values
    marginal_es_list = []

    for xt in positions:
        x = xt.reshape(-1, 1)
        variance = float(x.T @ sigma_matrix @ x)

        if variance <= 1e-10:
            delta_es = np.zeros_like(x.flatten())
        else:
            std_dev = np.sqrt(variance)
            portfolio_es = es_scaling * std_dev * np.sqrt(holding_period)
            beta_vector = (sigma_matrix @ x).flatten() / variance
            delta_es = portfolio_es * beta_vector

        marginal_es_list.append(delta_es)

    result = pd.DataFrame(
        marginal_es_list,
        index=returns_data.index.strftime("%Y-%m-%d"),
        columns=position_data.columns
    )

    if display_table:
        display(
            result.style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    result.attrs["es_unit"] = "monetary"
    return result


#----------------------------------------------------------
# Component Expected Shortfall (via Marginal ES)
#----------------------------------------------------------
def component_es(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    distribution="normal",
    display_table=False
):
    """
    Component Expected Shortfall (ES) Estimation.

    Computes the contribution of each asset to the total portfolio ES
    using Euler decomposition via Marginal ES.

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (T × N)

    - confidence_level (float):
        ES confidence level (default = 0.99)

    - holding_period (int, optional):
        ES horizon in days (default = 1)

    - distribution (str, optional):
        "normal" or "t" (default = "normal")

    - display_table (bool, optional):
        If True, displays styled table (default = False)

    Returns:
    - pd.DataFrame:
        Time series of Component ESs (T × N), in monetary units
    """
    position_data = pd.DataFrame(position_data)

    # Compute Marginal ES
    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        distribution=distribution,
        display_table=False
    )

    # Format index
    position_data_aligned = position_data.copy()
    if isinstance(position_data_aligned.index[0], pd.Timestamp):
        position_data_aligned.index = position_data_aligned.index.strftime("%Y-%m-%d")

    # Align positions
    aligned_positions = position_data_aligned.loc[marginal_df.index]

    # Compute Component ES
    component_df = aligned_positions * marginal_df

    if display_table:
        display(
            component_df.style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    component_df.attrs["es_unit"] = "monetary"
    return component_df


#----------------------------------------------------------
# Relative Component Expected Shortfall
#----------------------------------------------------------
def relative_component_es(
    position_data,
    confidence_level=0.99,
    holding_period=1,
    distribution="normal",
    display_table=False
):
    """
    Relative Component Expected Shortfall (ES) Estimation.

    Compute the proportionate contribution of each asset to total portfolio ES:
        RelativeComponentESᵢ = ComponentESᵢ / TotalES

    Description:
    - Measures how much each asset contributes to overall portfolio expected shortfall.
    - Useful for identifying risk concentration (e.g., overexposure to tail risk).

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (shape: T × N)

    - confidence_level (float):
        ES confidence level (default = 0.99)

    - holding_period (int, optional):
        ES horizon in days (default = 1)

    - distribution (str, optional):
        "normal" or "t" (default = "normal")

    - display_table (bool, optional):
        If True, displays styled table (default = False)

    Returns:
    - pd.DataFrame:
        Time series of Relative Component ESs (T × N), values between 0 and 1

    Notes:
    - Row sums equal 1 (unless ES = 0, in which case row is NaN)
    - Computation uses Component ES divided by total ES at each t
    """
    position_data = pd.DataFrame(position_data)

    # Compute component ES
    ces_df = component_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        distribution=distribution,
        display_table=False
    )

    # Total portfolio ES per time step
    total_es = ces_df.sum(axis=1)

    # Compute relative contributions
    relative_df = ces_df.div(total_es, axis=0)

    if display_table:
        display(
            relative_df.style
                .format("{:.2%}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    relative_df.attrs["es_unit"] = "relative"
    return relative_df


#----------------------------------------------------------
# Incremental Expected Shortfall
#----------------------------------------------------------
def incremental_es(
    position_data,
    change_vector,
    confidence_level=0.99,
    holding_period=1,
    distribution="normal",
    display_table=False
):
    """
    Incremental Expected Shortfall (ES) Estimation.

    Compute the impact on total portfolio ES of a change in positions, using Marginal ES.

    Description:
    - Approximates the change in portfolio ES caused by a small portfolio reallocation.
    - Based on first-order sensitivity:
        IESₜ = ΔESₜ' × a

    Parameters:
    - position_data (pd.DataFrame or np.ndarray):
        Time series of monetary holdings (shape: T × N)

    - change_vector (list or np.ndarray):
        Vector of position changes (shape: N,)

    - confidence_level (float):
        ES confidence level (default = 0.99)

    - holding_period (int, optional):
        Horizon for ES (default = 1)

    - distribution (str, optional):
        "normal" or "t" (default = "normal")

    - display_table (bool, optional):
        If True, displays a styled table (default = False)

    Returns:
    - pd.Series:
        Time series of Incremental ES values (in monetary units)
    """
    position_data = pd.DataFrame(position_data)

    # Compute Marginal ES
    marginal_df = marginal_es(
        position_data=position_data,
        confidence_level=confidence_level,
        holding_period=holding_period,
        distribution=distribution,
        display_table=False
    )

    a = np.asarray(change_vector).reshape(-1)
    if a.shape[0] != marginal_df.shape[1]:
        raise ValueError("Change vector must match number of assets.")

    # Compute Incremental ES (dot product)
    ies_series = marginal_df @ a

    if display_table:
        display(
            ies_series.to_frame("Incremental_ES")
                .style
                .format("{:.2f}")
                .set_table_styles([{"selector": "caption", "props": [("display", "none")]}])
        )

    ies_series.attrs["es_unit"] = "monetary"
    return ies_series


#----------------------------------------------------------
# Wealth Scaling for ES
#----------------------------------------------------------
def apply_wealth_scaling_es(result_data, es_estimate=None, wealth=None):
    """
    Apply wealth scaling to Expected Shortfall (ES) values.

    Parameters:
    - result_data (pd.DataFrame): Output of an ES function, must include an 'ES' column (in decimals, e.g., 0.01 = 1%).
    - es_estimate (float or None): Scalar ES value to be scaled (in decimals).
    - wealth (float or None): Portfolio value used for converting ES from percentage (decimal) to monetary units.

    Returns:
    - result_data (pd.DataFrame): DataFrame with optional new column 'ES_monetary' (in monetary units).
    - es_estimate (float or None): Scaled scalar ES in monetary units (if provided), else None.

    Notes:
    - Does not overwrite the 'ES' column; only adds 'ES_monetary'.
    - Assumes all ES values are in decimal format (e.g., 0.01 = 1%).
    """
    if wealth is not None and "ES" in result_data.columns:
        result_data["ES_monetary"] = result_data["ES"] * wealth

    if wealth is not None and es_estimate is not None:
        es_estimate = es_estimate * wealth

    return result_data, es_estimate


#----------------------------------------------------------
# Unified Expected Shortfall (ES) Caller with Wealth Scaling
#----------------------------------------------------------
def compute_expected_shortfall(method, result_data, confidence_level=0.99, wealth=None, **kwargs):
    """
    Unified Expected Shortfall (ES) Estimator with Optional Wealth Scaling.

    Dispatches ES computation to one of several supported methods:
    - 'historical': empirical tail average based on past returns.
    - 'parametric': closed-form ES using Normal or Student-t distribution.
    - 'volatility': time-varying ES from scaled innovation tails.
    - 'correlation': ES from correlation-based models with empirical tails.
    - 'marginal': marginal ES per asset using Euler decomposition.
    - 'component': asset-level ES contribution based on marginal ES.
    - 'relative_component': proportional ES contribution per asset.
    - 'incremental': first-order ES impact from position changes.

    Wealth Scaling:
    - Automatically applied to methods that return ES in decimal format.
    - Blocked for methods that return monetary ES or proportions directly.

    Parameters:
    - method (str): One of the supported method names.
    - result_data (pd.DataFrame): Input DataFrame with required columns per method.
    - confidence_level (float): ES confidence level (e.g., 0.99).
    - wealth (float or None): If provided, scales ES results (only for compatible methods).
    - **kwargs: Additional arguments depending on method:
        - 'returns' (pd.Series) for 'parametric'
        - 'holding_period' (int) for 'parametric', 'marginal', etc.
        - 'distribution' (str): 'normal' or 't' (optional)
        - 'subset' (tuple): Date range for volatility/correlation methods
        - 'position_data' (pd.DataFrame): Required for marginal/component/incremental
        - 'change_vector' (array-like): Required for 'incremental'

    Returns:
    - result_data (pd.DataFrame): Output with 'ES' and possibly 'ES_monetary'.
    - es_estimate (float or None): Scalar ES value (only for some methods).

    Notes:
    - All decimal ES values are in percentage-of-wealth units (e.g., 0.012 = 1.2%).
    - Methods returning ES in monetary units block external wealth scaling.
    - 'relative_component' returns proportions and is excluded from scaling.
    """
    method = method.lower()

    if method == "historical":
        result_data, es_estimate = compute_es_historical(result_data, confidence_level)

    elif method == "parametric":
        returns = kwargs.get("returns")
        if returns is None:
            raise ValueError("Parametric ES requires 'returns' (pd.Series).")
        holding_period = kwargs.get("holding_period", 1)
        distribution = kwargs.get("distribution", "normal")
        result_data, es_estimate = compute_es_parametric(
            result_data, returns, confidence_level, holding_period, distribution
        )

    elif method == "volatility":
        subset = kwargs.get("subset", None)
        result_data = compute_expected_shortfall_volatility(result_data, confidence_level, subset)
        es_estimate = None

    elif method == "correlation":
        if wealth is not None:
            raise ValueError("Method 'correlation' already returns monetary ES. 'wealth' must be None.")
        subset = kwargs.get("subset", None)
        result_data = compute_expected_shortfall_correlation(result_data, confidence_level, subset)
        es_estimate = None

    elif method == "marginal":
        if wealth is not None:
            raise ValueError("Method 'marginal' already returns monetary ES. 'wealth' must be None.")
        position_data = kwargs.get("position_data")
        distribution = kwargs.get("distribution", "normal")
        holding_period = kwargs.get("holding_period", 1)
        result_data = marginal_es(position_data, confidence_level, holding_period, distribution)
        es_estimate = None

    elif method == "component":
        if wealth is not None:
            raise ValueError("Method 'component' already returns monetary ES. 'wealth' must be None.")
        position_data = kwargs.get("position_data")
        distribution = kwargs.get("distribution", "normal")
        holding_period = kwargs.get("holding_period", 1)
        result_data = component_es(position_data, confidence_level, holding_period, distribution)
        es_estimate = None

    elif method == "relative_component":
        if wealth is not None:
            raise ValueError("Method 'relative_component' returns proportions. 'wealth' must be None.")
        position_data = kwargs.get("position_data")
        distribution = kwargs.get("distribution", "normal")
        holding_period = kwargs.get("holding_period", 1)
        result_data = relative_component_es(position_data, confidence_level, holding_period, distribution)
        es_estimate = None

    elif method == "incremental":
        if wealth is not None:
            raise ValueError("Method 'incremental' already returns monetary ES. 'wealth' must be None.")
        position_data = kwargs.get("position_data")
        change_vector = kwargs.get("change_vector")
        if change_vector is None:
            raise ValueError("Incremental ES requires 'change_vector'.")
        distribution = kwargs.get("distribution", "normal")
        holding_period = kwargs.get("holding_period", 1)
        result_data = incremental_es(position_data, change_vector, confidence_level, holding_period, distribution)
        es_estimate = None

    else:
        raise ValueError("Method must be one of: 'historical', 'parametric', 'volatility', 'correlation', "
                         "'marginal', 'component', 'relative_component', 'incremental'.")

    if method in {"historical", "parametric", "volatility"}:
        result_data, es_estimate = apply_wealth_scaling_es(result_data, es_estimate, wealth)

    return result_data, es_estimate


