#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import genpareto

#################################################
# Note: double check all formulas 
#################################################

#----------------------------------------------------------
# EVT VaR and ES
#----------------------------------------------------------
def evt(
    returns,
    confidence_level=0.99,
    threshold_percentile=99,
    exceedance_level=None,
    diagnostics=False
):
    """
    EVT-Based VaR and ES Estimation (Peaks Over Threshold Method).

    Estimate Value-at-Risk (VaR), Expected Shortfall (ES), and exceedance probabilities
    using Extreme Value Theory (EVT) applied to the right tail of the loss distribution.

    Description:
    - Uses the Peaks Over Threshold (POT) method with a Generalized Pareto Distribution (GPD)
    fitted to excess losses above a high threshold.
    - Suitable for univariate daily return series in decimal format.

    Formulas:
    Let u be the threshold (e.g., 99th percentile of losses), and y = loss - u.

    - GPD density for exceedances:
        y ∼ GPD(ξ, β), where y = loss − u

    - Value-at-Risk at level q:
        VaR_q = u + (β / ξ) × [ (N / nu × (1 - q))^(−ξ) − 1 ]

    - Expected Shortfall at level q:
        ES_q = [VaR_q + (β - ξ × u)] / (1 - ξ)

    - Probability of loss exceeding a given level x > u:
        P(X > x) = (nu / N) × [1 + ξ × (x - u) / β]^(−1/ξ)

    Parameters:
    - returns (pd.Series): 
        Daily log or simple returns (in decimals, e.g., 0.01 = 1%).

    - confidence_level (float, optional): 
        Confidence level for VaR and ES (default = 0.99).

    - threshold_percentile (float, optional): 
        Percentile (0–100) to define the threshold u (default = 99).

    - exceedance_level (float, optional): 
        Loss level to estimate exceedance probability (must be > u).

    - diagnostics (bool, optional): 
        If True, returns diagnostic info for the tail fit.

    Returns:
    - result_data (pd.DataFrame):
        - 'Returns': original returns
        - 'VaR': constant daily EVT VaR (positive, in decimals)
        - 'ES': constant daily EVT ES (positive, in decimals)
        - 'VaR Violation': True if return < -VaR

    - var_estimate (float): 
        EVT-based VaR estimate (absolute %, e.g., 3.24).

    - es_estimate (float): 
        EVT-based ES estimate (absolute %, e.g., 4.71).

    - prob_exceedance (float or None): 
        Probability of exceeding `exceedance_level` (decimal), or None.

    - diagnostics_dict (dict, if diagnostics=True):
        - 'xi': GPD shape parameter
        - 'beta': GPD scale parameter
        - 'threshold_u': Estimated threshold
        - 'max_support': GPD domain max (∞ if ξ ≥ 0)
        - 'num_exceedances': Number of exceedances above u

    Notes:
    - All inputs and loss levels must be in decimal format (e.g., 0.025 = 2.5%).
    - VaR and ES outputs are returned as positive percentages.
    - This method captures heavy tails and rare events in return distributions.
    """
    returns = returns.dropna()
    returns = pd.Series(returns)
    losses = -returns

    threshold_u = np.percentile(losses, threshold_percentile)
    exceedances = losses[losses > threshold_u] - threshold_u
    num_exceedances = len(exceedances)
    total_observations = len(losses)

    xi_hat, loc_hat, beta_hat = genpareto.fit(exceedances, floc=0)

    prob_exceedance = None
    if exceedance_level is not None:
        if exceedance_level <= threshold_u:
            raise ValueError("Exceedance level must be greater than threshold u.")
        y = exceedance_level - threshold_u
        tail_term = 1 + xi_hat * y / beta_hat
        if tail_term <= 0:
            prob_exceedance = 0.0
        else:
            prob_exceedance = (num_exceedances / total_observations) * tail_term ** (-1 / xi_hat)

    var_evt_value = threshold_u + (beta_hat / xi_hat) * (
        (total_observations / num_exceedances * (1 - confidence_level)) ** (-xi_hat) - 1
    )
    es_evt_value = (var_evt_value + (beta_hat - xi_hat * threshold_u)) / (1 - xi_hat)

    var_estimate = 100 * var_evt_value
    es_estimate = 100 * es_evt_value

    var_series = pd.Series(var_evt_value, index=returns.index)
    es_series = pd.Series(es_evt_value, index=returns.index)

    result_data = pd.DataFrame({
        "Returns": returns,
        "VaR": var_series,
        "ES": es_series
    })
    result_data["VaR Violation"] = returns < -var_series

    if diagnostics:
        x_max = threshold_u - beta_hat / xi_hat if xi_hat < 0 else np.inf
        diagnostics_dict = {
            "xi": xi_hat,
            "beta": beta_hat,
            "threshold_u": threshold_u,
            "max_support": x_max,
            "num_exceedances": num_exceedances
        }
        return result_data, var_estimate, es_estimate, prob_exceedance, diagnostics_dict

    return result_data, var_estimate, es_estimate, prob_exceedance

