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

from .BasicVaR import (
    var_historical,
    var_parametric_iid
)

from .VolatilityVaR import (
    garch_forecast,
    var_ewma,
    var_garch,
    var_arch,
    var_moving_average,
)

from .ExpectedShortfall import (
    compute_es_historical,
    compute_expected_shortfall_volatility
)

from .Backtesting import (
    backtest_var,
    subset_backtest_var
)

from .InteractivePlots import (
    interactive_plot_var,
    interactive_plot_es,
    interactive_plot_volatility,
    get_asset_color_map,
    interactive_plot_var_series,
    interactive_plot_risk_contribution_pie,
    interactive_plot_risk_contribution_lines,
    interactive_plot_correlation_matrix,
    
)

from .PortfolioVaR import (
    var_asset_normal,
    marginal_var,
    component_var,
    relative_component_var,
    incremental_var
)

__version__ = "0.1"
