# pyvar Examples

This folder contains Jupyter notebooks that demonstrate how to use the `pyvar` package in practice.

Each notebook showcases a different method or class of risk models, progressing from basic concepts to advanced applications.

---

## 📘 Example Notebooks

### `basic_var.ipynb`  
A gentle introduction covering the most common Value-at-Risk methods:  
- Parametric (Normal, Cornish-Fisher)  
- Non-parametric (Historical, Hybrid)  
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
- Assessing extreme risks from empirical returns  
- Modeling left-tail behavior in simulated profit and loss  

---

### `analytic_var.ipynb`  
Focuses on analytic VaR estimation for simple portfolios:  
- Asset Normal VaR  
- Undiversified VaR  
- Component VaR  
- Incremental VaR  

---

### `correlation_models.ipynb`  
Explores time-varying correlation structures between assets:  
- Moving Average and EWMA covariance estimators  
- Rolling PCA for denoising the variance-covariance matrix  
- Ledoit-Wolf shrinkage for robust covariance estimation  

---

### `factor_models.ipynb`  
Implements factor-based risk modeling:  
- Single-factor and multi-factor models  
- Combination with GARCH-based volatility modeling  

---

### `options.ipynb`  
Introduces basic Value-at-Risk models for options portfolios:  
- Black-Scholes pricing fundamentals
- Delta-normal VaR for individual options and simple portfolios
- Limitations of those simplified methods   

---

### `simulations.ipynb`  
Uses simulation-based methods to model risk under different assumptions:  
- Parametric and non-parametric simulations  
- Applications to both equity-only and equity + options portfolios  
- Multiday simulations for VaR forecasting

---

## 💡 How to Use

Open any notebook in this folder to explore a specific method. Each one is designed to be self-contained and includes:

- Theory refreshers where relevant  
- Code explanations and comments  
- Visual outputs and backtesting where applicable  
- Ready-to-use examples on real or simulated data

The parameters are easy to adjust, making it simple to experiment or build on the notebooks for your own work.

---

Feel free to adapt these notebooks to suit your own analysis or build on them to develop more advanced workflows.
