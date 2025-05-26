"""
pyvar: A Python Toolkit for Modern Financial Risk Management

pyvar is a modular, extensible Python package for financial risk analysis, 
with a primary focus on Value-at-Risk (VaR) and Expected Shortfall (ES). 
It provides a full pipeline for risk estimation—from data ingestion and modeling 
to simulation, backtesting, and visualization—suitable for both academic and professional use.

Main Features
-------------
The package supports historical and parametric VaR/ES methods, GARCH-family models 
(including EGARCH, GJR-GARCH, and APARCH), as well as Extreme Value Theory using the 
Peaks-Over-Threshold (POT) approach. Rolling estimators such as EWMA, ARCH(p), and MA 
are also included. For portfolio-level analysis, pyvar provides time-varying correlation 
models (EWMA, MA), full risk decomposition (marginal, incremental, component, and relative 
contributions), and factor models including CAPM and Fama-French 3-Factor.

Simulation methods include parametric and historical Monte Carlo, with Black-Scholes 
pricing support for options. Backtesting routines cover standard statistical 
tests such as Kupiec’s test, Christoffersen’s test, and their joint likelihood ratio. 
Visualization tools are available for volatility paths, VaR series, portfolio contributions, 
simulated distributions and more.

Authors
-------
- Alessandro Dodon
- Niccolò Lecce
- Marco Gasparetti

Version
-------
0.1
"""
from .data_download import (
    get_raw_prices,
    convert_to_base,
    create_portfolio,
    summary_statistics,
    validate_matrix
)

from .basic_var import (
    historical_var,
    historical_es,
    parametric_var,
    parametric_es
)

from .evt import (
    fit_evt_parameters,
    evt_var,
    evt_es
)

from .volatility import (
    forecast_garch_variance,
    forecast_garch_var,
    ewma_var,
    garch_var,
    arch_var,
    ma_var,
    volatility_es
)

from .portfolio_metrics import (
    asset_normal_var,
    marginal_var,
    component_var,
    relative_component_var,
    incremental_var,
    marginal_es,
    component_es,
    relative_component_es,
    incremental_es
)

from .time_varying_correlations import (
    ma_correlation_var,
    ewma_correlation_var,
    correlation_es
)

from .factor_models import (
    single_factor_var,
    fama_french_var,
    factor_models_es
)

from .simulations import (
    black_scholes,
    monte_carlo_var,
    multiday_monte_carlo_var,
    historical_simulation_var,
    simulation_es
)

from .backtesting import (
    count_violations,
    kupiec_test,
    christoffersen_test,
    joint_lr_test
)

from .plots import (
    display_high_dpi_inline,
    plot_backtest,
    plot_volatility,
    get_asset_color_map,
    plot_var_series,
    plot_risk_contribution_bar,
    plot_risk_contribution_lines,
    plot_correlation_matrix,
    plot_simulated_distribution,
    plot_simulated_paths
)

__version__ = "0.1"
