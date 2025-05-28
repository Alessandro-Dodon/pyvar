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
