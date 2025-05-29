# pyvar Examples

This folder contains example scripts that demonstrate how to use the pyvar package in practice.

---

## 📘 Jupyter Notebook

### `pyvar_example_usage.ipynb`

This is the main tutorial notebook. It:

- Walks through most of the **core functionality** in the package
- Explains the logic behind each method with theory refreshers
- Provides **visualizations**
- Starts with single-asset risk models and ends with full portfolio analysis
- Demonstrates **two simple portfolio strategies** with end-to-end risk evaluation

It’s designed as a real-life walkthrough for new users.

---

## 🤖 Script for LLM Integration

### pyvar_llm_report.py

A practical example showing how to combine the **pyvar** package with a local LLM (via **llm.llm_rag**) to:

1. Calculate **VaR** & **ES** on an equity + options portfolio  
2. **Backtest** multiple VaR models (Kupiec, Christoffersen, joint tests)  
3. _(Optional)_ Display interactive charts  
4. _(Optional)_ Ask an LLM for automated interpretation and produce a PDF report

### Installation

1. **Clone the repository** and go into the `examples` folder:  
   ```bash
   git clone https://github.com/Alessandro-Dodon/pyvar.git
   cd pyvar
   cd examples

2. **Install core dependencies**:  
   ```bash
   pip install -r requirements.txt

3. **LM Studio setup** (_OPTIONAL_):
   
   To download and configure your local LM Studio correctly, follow the step-by-step notebook:                 `llm/tutorial_llm.ipynb`


### Configuration
Open `pyvar_llm_report.py` and adjust at the top:

```python
# VaR & ES confidence level
CONFIDENCE_LEVEL       = 0.99

# History window (business days)
LOOKBACK_BUSINESS_DAYS = 300

# Toggle interactive charts
SHOW_PLOTS             = True   # or False

# Toggle LLM interpretation & PDF report
RUN_LLM_INTERPRETATION = True   # or False

# Local LLM endpoint & model
rag.LMSTUDIO_ENDPOINT  = "http://<your-host>:<port>"
rag.API_PATH           = "/v1/completions"
rag.MODEL_NAME         = "qwen-3-4b-instruct"
```


### Quick Start
Run `pyvar_llm_report.py`

Enter when prompted:

- **Base currency**
  Input one base currecy (e.g. _EUR_, _CHF_, _USD_...), then press Enter.

- **Equity tickers**
  Enter all equity tickers at once, space-separated and exactly as on Yahoo Finance (e.g. _MSFT_ _ISP.MI_ _NESN.SW_), then press Enter.

- **Number of shares per ticker**  
  When prompted for each ticker, type the number of shares you hold and press **Enter** to confirm. Repeat until you’ve entered a value for every ticker.


- **Option positions** (**y** if you have options position in the portfolio, **n** if not)
 - if **y**, for each option input:
   - **Underlying stock** (e.g. AAPL), click Enter
   - **Type of options**: call or put, click Enter
   - **Number of contracts**, click Enter
   - **Multiplier** (number of stock x contract, default is 100), click Enter
   - **Strike price**, click Enter
   - **Time to maturity** (in years) (e.g. 1 day = 0,00396), click Enter
- To add another option repeat the steps above or click enter to launch the analysis  

### Output

When the script finishes, you’ll get:

- **Console**:  
  - Portfolio positions table  
  - VaR & ES metrics  
  - Backtest summary  


<summary> Charts (optional)</summary> 

If `SHOW_PLOTS = True`, interactive charts will open in your browser.


<summary> PDF Report (optional)</summary>

If `RUN_LLM_INTERPRETATION = True`, the LLM interpretation runs automatically and a PDF report is generated (e.g., in `./reports/`).
</details>

### Output

When the script finishes, you’ll get:

- **Console**:  
  - Portfolio positions table  
  - VaR & ES metrics  
  - Backtest summary  

- **Charts (optional)**  
  <details>
  <summary>Show details</summary>

  If `SHOW_PLOTS = True`, interactive charts will open in your browser.

  </details>

- **PDF Report (optional)**  
  <details>
  <summary>Show details</summary>

  If `RUN_LLM_INTERPRETATION = True`, the LLM interpretation runs automatically and a PDF report is generated (e.g., in `./reports/`).

  </details>
  
### Troubleshooting

- **Missing data / NaN**  
  Tickers without valid price history are dropped automatically (check console warnings).

- **Error “At least two time steps are required”**  
  Increase `LOOKBACK_BUSINESS_DAYS` or verify data availability.

- **LLM connection issues**  
  Ensure `rag.LMSTUDIO_ENDPOINT` is reachable and the service is running.




---

Feel free to run or adapt these examples to suit your own analysis!






