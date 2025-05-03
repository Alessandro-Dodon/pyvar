import numpy as np
import pandas as pd

def mc_var_cvar(mu: pd.Series,
                cov: pd.DataFrame,
                port_val: float,
                weights: pd.Series,
                sims: int = 5000,
                alpha: float = 5.0):
    """
    Monte Carlo VaR e CVaR.

    Args:
        mu       : Serie di rendimenti attesi degli asset.
        cov      : Matrice di covarianza dei rendimenti.
        port_val : Valore corrente del portafoglio.
        weights  : Serie di pesi del portafoglio (sommati a 1).
        sims     : Numero di simulazioni (default=5000).
        alpha    : Percentile per il calcolo del VaR (default=5).

    Returns:
        var      : Value at Risk assoluto a livello alpha.
        cvar     : Conditional VaR assoluto a livello alpha.
        port_sims: Array delle simulazioni dei valori di portafoglio.
    """
    # Matrice di Cholesky per correlazioni
    L = np.linalg.cholesky(cov.values)
    # Simulazioni di shock gaussiani indipendenti
    Z = np.random.standard_normal((len(mu), sims))
    # Simulazioni di rendimenti
    sim_rets = mu.values.reshape(-1, 1) + L @ Z
    # Simulazioni valore portafoglio
    port_sims = port_val * (1 + weights.values @ sim_rets)
    # VaR: perdita massima al percentile alpha
    var = port_val - np.percentile(port_sims, alpha)
    # CVaR: perdita media nella coda
    tail_cut = np.percentile(port_sims, alpha)
    cvar = port_val - port_sims[port_sims <= tail_cut].mean()

    return var, cvar, port_sims
