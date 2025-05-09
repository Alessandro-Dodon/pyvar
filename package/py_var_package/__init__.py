"""
PyVaR: A financial risk analysis toolkit

This package includes functions for estimating and backtesting Value-at-Risk (VaR),
Expected Shortfall (ES), and volatility using GARCH and related models.
It also includes tools for interactive visualizations and portfolio risk decomposition.

Main Features:
- Historical and parametric VaR estimation
- GARCH, EWMA, and other volatility models
- Backtesting of VaR and ES
- Interactive risk visualizations (Plotly)
- Component and incremental portfolio VaR

Authors: Alessandro Dodon, Niccolò Lecce, Marco Gasparetti
Version: 0.1
"""

from .basic_var import (
    var_historical,
    var_parametric
)

from .volatility_var import (
    garch_forecast,
    var_ewma,
    var_garch,
    var_arch,
    var_moving_average,
)

from .expected_shortfall import (
    es_historical,
    es_parametric,
    es_volatility,
    es_correlation,
    marginal_es,
    component_es,
    relative_component_es,
    incremental_es
)

from .backtesting import (
    count_violations,
    kupiec_test,
    christoffersen_test,
    joint_lr_test
)

from .evt import (
    evt
)

from .dynamic_correlations import (
    var_corr_moving_average,
    var_corr_ewma
)

from .plots import (
    plot_backtest,
    plot_volatility,
    get_asset_color_map,
    plot_var_series,
    plot_risk_contribution_bar,
    plot_risk_contribution_lines,
    plot_correlation_matrix,
    
)

from .portfolio_var import (
    var_asset_normal,
    marginal_var,
    component_var,
    relative_component_var,
    incremental_var
)

__version__ = "0.1"
