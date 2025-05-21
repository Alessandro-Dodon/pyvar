"""
pyvar: A Comprehensive Financial Risk Analysis Toolkit

pyvar is a modular Python package designed for financial risk modeling, 
focusing on Value-at-Risk (VaR) and Expected Shortfall (ES). 
It provides analytical tools, interactive visualizations, and rigorous 
backtesting utilities to support risk management and portfolio analysis.

Main Features
-------------
- VaR and ES estimation using historical, parametric, simulation, and EVT methods
- Volatility modeling via GARCH, EWMA, ARCH, and MA models
- Portfolio risk decomposition (marginal, incremental, component VaR/ES)
- Time-varying correlation models (EWMA, MA)
- Factor models including CAPM and Fama-French
- Monte Carlo and historical simulations for equities and derivatives
- Backtesting procedures (Kupiec, Christoffersen, Joint LR tests)
- Interactive and static plots for risk metrics and diagnostics

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

from .portfolio import (
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

from .correlation import (
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
