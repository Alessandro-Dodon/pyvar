# pyvar Examples

This folder contains example scripts that demonstrate how to use the pyvar package in practice.

---

## 📘 Jupyter Notebook

### `pyvar_example_usage.ipynb`

This is the main tutorial notebook. It:

- Walks through most of the core functionality in the package
- Explains the logic behind each method with theory refreshers
- Provides visualizations
- Starts with single-asset risk models and ends with full portfolio analysis
- Demonstrates two simple portfolio strategies with end-to-end risk evaluation

It’s designed as a real-life walkthrough for new users.

---

## 🤖 Script for LLM Integration

## pyvar_llm_report.py

A minimal, practical example showing how to combine the **pyvar** package with a local LLM (via **llm.llm_rag**) to:

- Calculate VaR & ES on an equity + options portfolio  
- Backtest your VaR models (Kupiec, Christoffersen, joint tests)  
- (Optionally) plot risk charts interactively  
- (Optionally) query an LLM for automated interpretation and generate a PDF report  

---

### 📋 Prerequisites

- **Python 3.8+**  
- A working **virtual environment** (recommended)  
- The following packages installed (see `requirements.txt` if provided):


---

Feel free to run or adapt these examples to suit your own analysis!


VaR & ES Risk Report for Equity + Options Portfolio
An example of combining pyvar with a local LLM (llm.llm_rag) to:

- Compute Value-at-Risk (VaR) and Expected Shortfall (ES) for equity and options portfolios

- Backtest each VaR model (Kupiec, Christoffersen, joint tests)

- Optionally display interactive risk charts

- Optionally invoke an LLM for automated interpretation and output a PDF report

📋 Prerequisites
Python 3.8+

Virtual environment (recommended)

Install required packages:

bash
Copy
Edit
pip install pandas numpy yfinance pandas_datareader pyvar llm_rag
If you enable LLM interpretation, have your LLM endpoint running (e.g. LM Studio).

## 🔧 Configuration
Open pyvar_llm_report.py and adjust at the top:

### VaR/ES confidence level
CONFIDENCE_LEVEL       = 0.99

### Number of days to include in the backtest
LOOKBACK_BUSINESS_DAYS = 300

### Toggle features
SHOW_PLOTS             = True   # set False to skip interactive charts
RUN_LLM_INTERPRETATION = True   # set False to skip LLM + PDF

### Local LLM endpoint (we use the model qwen-3-4b-instruct)
- rag.LMSTUDIO_ENDPOINT  = "http://xxx.x.x.x:xxxx"
- rag.API_PATH           = "/v1/completions"
- rag.MODEL_NAME         = "qwen-3-4b-instruct"


## ▶️ Quick Start
Run python pyvar_llm_report.py

Enter when prompted:

- Base currency (e.g. EUR, CHF, USD...)

- Equity tickers (space separated and as they are on yahoo finance, e.g. NVDA ISP.MI NESN.SW)

- Number of shares per ticker

- Option positions (y if you have options position in the portfolio, n if not)
   - if y, for each option input:
   - Underlying stock (e.g. AAPL)
   - Type of options: call or put
   - Number of contracts
   - Multiplier (number of stock x contract, default is 100)
   - Strike price
   - Time to maturity (in years) (e.g. 1 day = 0,00396)
- Then add another option by repeating the steps above or click enter to launch the analysis  

Watch the console for tables of positions, VaR/ES metrics, backtest summaries.

(Optional) Charts will pop up if SHOW_PLOTS=True.

(Optional) LLM interpretation will run and PDF report generate if RUN_LLM_INTERPRETATION=True.

📂 Output
Console:

Equity & options positions

VaR & ES values by model

Backtest results (violations, rates, p-values)

LLM commentary (if enabled)

Interactive windows (if enabled):

VaR series, volatility estimates, risk contributions, correlation matrix, backtest charts

PDF report (if enabled):

Comprehensive report embedding metrics, plots, and AI-generated insights

🛠 Troubleshooting
Missing data / NaNs: tickers with no valid price history are dropped automatically (you’ll see a warning).

Error “At least two time steps are required”: increase LOOKBACK_BUSINESS_DAYS or verify data availability.

LLM connection issues: confirm your rag.LMSTUDIO_ENDPOINT is reachable and the service is running.
