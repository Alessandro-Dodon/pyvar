#----------------------------------------------------------
# Packages
#----------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.stats import genpareto

#################################################
# Note: double check all formulas 
#       check probability of exceedance and diagnostics
#################################################

#----------------------------------------------------------
# EVT VaR and ES
#----------------------------------------------------------
def evt(
    returns,
    confidence_level=0.99,
    threshold_percentile=99,
    exceedance_level=None,
    diagnostics=False,
    wealth=None
):
    """
    Estimate VaR, ES, and exceedance probabilities using Extreme Value Theory (EVT).

    Applies the Peaks Over Threshold (POT) approach with a Generalized Pareto Distribution (GPD) 
    to model the extreme right tail of the loss distribution. This method captures rare, high-impact 
    events beyond a high quantile threshold.

    Both VaR and ES estimates are derived from the same fitted GPD parameters, making it efficient 
    to compute them jointly in a single function call.

    Parameters:
    - returns (pd.Series): Daily returns in decimal format (e.g., 0.01 = 1%).
    - confidence_level (float): Confidence level for VaR/ES (default: 0.99).
    - threshold_percentile (float): Quantile for defining tail threshold (default: 99).
    - exceedance_level (float, optional): Loss level to estimate exceedance probability.
    - diagnostics (bool): Return GPD parameters and diagnostics if True.
    - wealth (float, optional): Scale VaR/ES to monetary values if specified.

    Returns:
    - result_data (pd.DataFrame): Columns include:
        - 'Returns', 'VaR', 'ES', 'VaR Violation'
        - Optionally: 'VaR_monetary', 'ES_monetary'
    - var_estimate (float): Estimated VaR (in % or monetary units).
    - es_estimate (float): Estimated ES (in % or monetary units).
    - prob_exceedance (float or None): Probability of exceeding `exceedance_level`, if provided.
    - diagnostics_dict (optional): Dictionary with GPD parameters and fit stats.
    """
    returns = pd.Series(returns).dropna()
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

    # Apply wealth scaling if requested
    if wealth is not None:
        result_data["VaR_monetary"] = result_data["VaR"] * wealth
        result_data["ES_monetary"] = result_data["ES"] * wealth
        var_estimate = var_evt_value * wealth
        es_estimate = es_evt_value * wealth

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

