# pyvar Examples

This folder contains Jupyter notebooks that demonstrate how to use the `pyvar` package in practice.

Each notebook showcases a different method or class of risk models, progressing from basic concepts to advanced applications.

---

## 📘 Example Notebooks

### `basic_var.ipynb`  
A gentle introduction covering the most common Value-at-Risk methods:  
- Parametric (Normal, Cornish-Fisher)  
- Non-parametric (Historical)  
- Comparative visualization and commentary  

---

### `volatility_models.ipynb`  
Demonstrates portfolio volatility estimation using various volatility models:  
- Simple moving average  
- Exponentially weighted moving average (EWMA)  
- GARCH-family models (GARCH, GJR-GARCH, with different distributions)  

---

### `evt.ipynb`  
Applies Extreme Value Theory (EVT) in two distinct use cases:  
- Modeling left-tail behavior in simulated profit and loss  
- Assessing extreme risks from empirical returns  

---

### `analytic_var.ipynb`  
Focuses on analytic VaR estimation for simple portfolios:  
- Portfolio-level parametric VaR  
- Variance-covariance methods  
- Portfolio weights, correlations, and marginal risk contributions  
- An end-to-end view of multi-asset portfolio risk modeling  

---

### `correlation_models.ipynb`  
Explores time-varying correlation structures between assets:  
- Moving average and EWMA covariance estimators  
- Rolling PCA for dimension reduction  
- Ledoit-Wolf shrinkage for improved covariance estimation  

---

### `factor_models.ipynb`  
Implements factor-based risk modeling:  
- Single-factor and multi-factor models  
- Optional combination with GARCH-based volatility modeling  
- Use of economic and statistical factors for portfolio VaR  

---

### `options.ipynb`  
Introduces basic Value-at-Risk models for options portfolios:  
- Delta-normal VaR  
- Limitations of parametric assumptions  
- Examples combining options with equity holdings  

---

### `simulations.ipynb`  
Uses simulation-based methods to model risk under different assumptions:  
- Parametric and non-parametric simulations  
- Applications to both equity-only and equity + options portfolios  
- Visualization of P&L distributions and tail events  

---

## 💡 How to Use

Open any notebook in this folder to explore a specific method. Each one is designed to be self-contained and includes:

- Code explanations and comments  
- Theory refreshers where relevant  
- Visual output and backtesting  
- Ready-to-use examples on real or simulated data

---

Feel free to adapt these notebooks to suit your own analysis or build on them to develop more advanced workflows.
