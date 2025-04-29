import numpy as np
import pandas as pd
from scipy.stats import genpareto



# EVT VaR and ES
def evt(
    returns,
    confidence_level=0.99,
    threshold_percentile=99,
    exceedance_level=None,
    diagnostics=False
):
    """
    Estimate EVT-based Value-at-Risk (VaR), Expected Shortfall (ES), and exceedance probabilities
    using the Peaks Over Threshold (POT) method with Generalized Pareto Distribution (GPD) fitting.

    This function applies EVT to the right tail of the loss distribution, assuming
    losses are the negative of returns. It is suitable for univariate daily log returns.

    Parameters:
    ----------
    returns : pd.Series
        Daily log returns or simple returns in decimal form (e.g., 0.01 = 1%).
    confidence_level : float, optional
        Confidence level for VaR and ES (e.g., 0.99 for 99%). Default is 0.99.
    threshold_percentile : float, optional
        Percentile (0–100) used to select the threshold u for POT (e.g., 99 = top 1% tail). Default is 99.
    exceedance_level : float, optional
        Loss level (in decimals) at which to estimate the probability of exceedance. Must be > threshold u.
    diagnostics : bool, optional
        If True, return tail fit parameters and threshold diagnostics.

    Returns:
    -------
    result_data : pd.DataFrame
        Contains:
        - 'Returns': original return series
        - 'VaR': constant daily EVT VaR estimate (positive loss magnitude, in decimals)
        - 'ES' : constant daily EVT ES estimate (positive loss magnitude, in decimals)
        - 'VaR Violation': boolean indicating whether return < -VaR
    var_estimate : float
        EVT-based Value-at-Risk (in percentage, absolute magnitude).
    es_estimate : float
        EVT-based Expected Shortfall (in percentage, absolute magnitude).
    prob_exceedance : float or None
        Estimated probability of exceeding `exceedance_level` (as a decimal loss), or None if not specified.
    diagnostics_dict : dict (only if diagnostics=True)
        Contains:
        - 'xi'          : GPD shape parameter
        - 'beta'        : GPD scale parameter
        - 'threshold_u' : Threshold value u (in decimals)
        - 'max_support' : Maximum domain of GPD (∞ if xi ≥ 0, otherwise finite)
        - 'num_exceedances': Number of exceedances above threshold

    EVT Formulas:
    ------------
    Let u be the threshold (e.g., 99th percentile of losses), and y = loss - u.

    • GPD(y; ξ, β) fitted to y = losses − u
    • VaR_q = u + (β / ξ) * [ (N / nu * (1 - q))^(−ξ) − 1 ]
    • ES_q  = [VaR_q + (β - ξ * u)] / (1 - ξ)
    • P(X > x) = (nu / N) * [1 + ξ * (x - u) / β]^(−1/ξ)

    Notes:
    -----
    - All returns and loss levels must be in decimal format (e.g., 0.025 = 2.5%).
    - VaR and ES estimates are returned as percentages (e.g., 3.54).
    - Exceedance probabilities are returned as decimals (e.g., 0.012 = 1.2%).
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

