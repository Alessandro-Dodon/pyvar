import pyvar as pv

import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

# Download the VT data from YF
vt_data = yf.download("VT", start="2000-01-01", end="2025-01-01")

# Check for missing values in the entire DataFrame (including the new column)
print(vt_data.isnull().sum())

# Compute percentage returns and remove the first NaN
vt_data["Percent Returns"] = vt_data["Close"].pct_change()
vt_returns = vt_data["Percent Returns"].dropna()

# Set parameters
confidence_level = 0.99 # <----- Can choose 0.95 etc
wealth=100_000 # <----- Wealth invested in our ETF

# Apply Parametric VaR model 
parametric_normal_results = pv.parametric_var(vt_returns, confidence_level, distribution="normal", wealth=wealth) # <----- Can choose t

# Compute ES for the whole period
parametric_normal_results = pv.parametric_es(parametric_normal_results, confidence_level, distribution="normal", wealth=wealth) # <----- Can choose t

# Save static plot to PDF (no display) 
# pv.plot_backtest(parametric_normal_results, interactive=False, output_path="backtest_plot.pdf")

# PUT YOUR WD HERE
pv.plot_backtest(
    parametric_normal_results,
    interactive=False,
    output_path="/Users/aledo/Documents/GitHub/pyvar/package/example_usage/backtest_plot.pdf"
)

# or plot interactive like

# Plot backtest
fig = pv.plot_backtest(parametric_normal_results, interactive=True)
fig.show()


