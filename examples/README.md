# pyvar Examples

This folder contains example scripts that demonstrate how to use the pyvar package in practice.

---

## 📘 Jupyter Notebook

### `pyvar_example_usage.ipynb`

This is the main tutorial notebook. It serves as a real-life walkthrough for new users and includes:

- A tour of the core functionality in the package
- Explanations of the logic behind each method, with theory refreshers
- Visualizations to support interpretation
- A progression from single-asset risk models to full portfolio analysis
- Two simple portfolio strategies, with end-to-end risk evaluation

---

## 🤖 Script for LLM Integration

### `pyvar_llm_report.py`

A practical example showing how to combine the pyvar package with a local LLM to:

- Calculate VaR & ES on an equity + options portfolio  
- Backtest multiple VaR models (Kupiec, Christoffersen, joint tests)  
- _(Optional)_ Use an LLM to automatically interpret results and generate a clear, simplified PDF report

### Setup

1. Clone the repository and go into the `examples` folder:  
   ```bash
   git clone https://github.com/Alessandro-Dodon/pyvar.git
   cd pyvar
   cd examples

2. Install core dependencies:  
   ```bash
   pip install -r requirements.txt

3. _(Optional)_ Setup LM Studio:
   <details>
   <summary>Show details</summary>
  
   To download and configure your local LM Studio correctly, follow the step-by-step notebook:                 `llm/tutorial_llm.ipynb`
   
   </details>

### Configuration
Open `pyvar_llm_report.py` and adjust at the top:

```python
# VaR & ES confidence level
CONFIDENCE_LEVEL = 0.99 

# Lookback period (business days)
LOOKBACK_BUSINESS_DAYS = 300

# Toggle LLM interpretation & PDF report
RUN_LLM_INTERPRETATION = True # or False
ANSWER_LLM_LENGHT = 500 # Length of the LLM answer in tokens

# Local LLM endpoint & model
LMSTUDIO_ENDPOINT  = "http://<your-host>:<port>" # Local LM Studio server URL
API_PATH           = "/v1/completions"
MODEL_NAME = "your-model-name-here" # Model's API identifier

```
You can also change the prompt by going in `llm/llm_rag.py` and changing the `prompt_sections` variable in the `build_rag_prompt` function. Find more info on that in the `llm/` folder.

If you plan to use the LLM interpretation, make sure your LM Studio server is running, the `rag.LMSTUDIO_ENDPOINT` is reachable and the specified `MODEL_NAME` is loaded

### Quick Start
Run `pyvar_llm_report.py`. Enter when prompted:

- **Base currency**  
  Input one base currecy (e.g. _EUR_, _CHF_, _USD_...), then press Enter.

- **Equity tickers**  
  Enter all equity tickers at once, space-separated and exactly as on Yahoo Finance (e.g. _MSFT_ _ISP.MI_ _NESN.SW_), then press Enter.

- **Number of shares per ticker**  
  When prompted for each ticker, type the number of shares you hold and press Enter to confirm. Repeat until you’ve entered a value for every ticker.


- **Option positions** (y if you have options position in the portfolio, n if not)
 - if **y**, for each option input:
   - **Underlying stock** (e.g. AAPL), click Enter
   - **Type of options**: call or put, click Enter
   - **Number of contracts**, click Enter
   - **Multiplier** (number of stock x contract, default is 100), click Enter
   - **Strike price**, click Enter
   - **Time to maturity** (in years), click Enter
- To add another option repeat the steps above or click enter to launch the analysis
  

### Output

When the script finishes, you’ll get:

- **Console**:  
  - Portfolio positions table  
  - Portfolio VaR & ES   
  - Backtest results
  - Summary text

- **_(Optional)_ PDF Report**  
  <details>
  <summary>Show details</summary>

  If `RUN_LLM_INTERPRETATION = True`, the LLM interpretation runs automatically and a simple PDF report is generated.

  </details>
  
### Troubleshooting

- **Missing data / NaN**  
  Tickers without valid price history are dropped automatically (check console warnings).

- **Error “At least two time steps are required”**  
  Increase `LOOKBACK_BUSINESS_DAYS` or verify data availability.

- **LLM connection issues**  
  Ensure `rag.LMSTUDIO_ENDPOINT` is reachable and the service is running.


---


## ⚠️ Disclaimer

The LLM integration is experimental. Do not rely solely on its output, always verify results with traditional methods or a qualified expert.  



---

Feel free to run or adapt these examples to suit your own analysis!






