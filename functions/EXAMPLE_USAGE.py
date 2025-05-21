from pandas.tseries.offsets import BDay
import pandas as pd
import yfinance as yf


from data_download import get_raw_prices, convert_to_base, create_portfolio, summary_statistics




# ==================================================================
# 1) INPUT UTENTE – Base currency, Ticker e azioni
# ==================================================================
BASE   = input("Valuta base [EUR] : ").strip().upper() or "EUR"
TKS    = input("Tickers equity    : ").upper().split()
SHARES = pd.Series({t: float(input(f"  Shares {t:<6}: ")) for t in TKS})

# ==================================================================
# 2) DATA DOWNLOAD + PREPROCESSING
# ==================================================================
from pandas.tseries.offsets import BDay
import pandas as pd
import yfinance as yf

START = (pd.Timestamp.today() - BDay(100)).strftime("%Y-%m-%d")

# Step 1: Scarica prezzi raw
raw_prices = get_raw_prices(TKS, start=START)

# Step 2: Detect currencies via yfinance
currencies = {}
for t in TKS:
    try:
        currencies[t] = yf.Ticker(t).fast_info.get("currency", BASE) or BASE
    except Exception:
        currencies[t] = "UNKNOWN"

# Step 3: Conversione a valuta base
prices = convert_to_base(raw_prices, currency_mapping=currencies, base_currency=BASE)

# Step 4: Portafoglio monetario (posizioni = prezzi * azioni)
positions_df = create_portfolio(prices, SHARES)

# Step 5: Statistiche
returns, mu, cov = summary_statistics(prices)

# Step 6: Ultimo prezzo, pesi, valore portafoglio
last_prices   = prices.iloc[-1]
port_val_eq   = float((last_prices * SHARES).sum())
weights       = (last_prices * SHARES) / port_val_eq

# ==================================================================
# DEBUG STAMPA
# ==================================================================
print("\n===== DEBUG: Valore portafoglio e pesi =====")
print(f"Valore portafoglio equity: {port_val_eq:,.2f} {BASE}")
print("\nPesi:")
print(weights.to_string(float_format=lambda x: f"{x:.4f}"))
