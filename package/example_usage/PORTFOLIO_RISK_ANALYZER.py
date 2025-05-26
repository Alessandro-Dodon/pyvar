#================================================================
# VaR and ES Risk Report for Equity + Options Portfolio (with plots)
# ================================================================
import os, sys

# individua la cartella root del progetto (tre livelli sopra questo file)
project_root = os.path.abspath(
    os.path.join(__file__, os.pardir, os.pardir, os.pardir)
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay

import pyvar as pv
from pyvar.backtesting import (
    count_violations,
    kupiec_test,
    christoffersen_test,
    joint_lr_test
)
from pyvar.plots import (
    plot_backtest,
    plot_volatility,
    plot_var_series,
    plot_risk_contribution_bar,
    plot_risk_contribution_lines,
    plot_correlation_matrix
)

import llm.llm_rag as rag
from llm.pdf_reporting import save_report_as_pdf, open_report_as_pdf

# ----------------------------------------------------------
# Patch: override the imported plot functions to auto-show + title
# ----------------------------------------------------------
def _auto_show_wrapper(fn):
    def inner(*args, interactive=True, title=None, **kwargs):
        fig = fn(*args, interactive=interactive, **kwargs)
        if fig is not None:
            if title:
                fig.update_layout(title=title)
            if interactive:
                fig.show()
        return fig
    return inner

plot_backtest                = _auto_show_wrapper(plot_backtest)
plot_volatility              = _auto_show_wrapper(plot_volatility)
plot_var_series              = _auto_show_wrapper(plot_var_series)
plot_risk_contribution_bar   = _auto_show_wrapper(plot_risk_contribution_bar)
plot_risk_contribution_lines = _auto_show_wrapper(plot_risk_contribution_lines)
plot_correlation_matrix      = _auto_show_wrapper(plot_correlation_matrix)


if __name__ == "__main__":
    '''# 1) INPUT UTENTE
    BASE     = "EUR"
    TICKERS  = ["NVDA", "MSFT"]
    SHARES   = pd.Series({"NVDA": 3, "MSFT": 4})
    CONF     = 0.99


    # 2) OPTIONS INPUT (demo)
    options_list = [
        {"under": "AAPL", "type": "call", "contracts": 1, "multiplier": 100,
         "qty": 100, "K": 210.0, "T": 1.0}]'''
    



     # ——————————————————————————————
    # 1) INPUT UTENTE INTERATTIVO
    # ——————————————————————————————

    # --- LLM CONFIGURATION ---
    rag.LMSTUDIO_ENDPOINT = "http://127.0.0.1:1234"
    rag.API_PATH          = "/v1/completions"
    rag.MODEL_NAME        = "qwen-3-4b-instruct"
    #--------------------------------------------------------------
    
    # Base valutaria
    BASE = input("Valuta base [EUR]: ").strip().upper() or "EUR"

    # Ticker equity
    TICKERS = input("Tickers equity (separati da spazio): ").upper().split()
    if not TICKERS:
        print("Nessun ticker inserito, esco.")
        sys.exit(1)

    # Numero di azioni per ticker
    SHARES = {}
    for t in TICKERS:
        while True:
            try:
                q = float(input(f"  Shares {t:<6}: "))
                SHARES[t] = q
                break
            except ValueError:
                print("    Numero non valido, riprova.")

    # Opzioni
    options_list = []
    if input("Hai opzioni? (s/n): ").lower().startswith("s"):
        while True:
            u = input("  Sottostante (ENTER per terminare): ").upper()
            if not u:
                break
            typ = input("    Tipo (call/put): ").lower()
            contr = float(input("    Contratti (numero): "))
            mult = int(input("    Multiplier [100]: ") or 100)
            K = float(input("    Strike: "))
            T = float(input("    Time-to-maturity (anni): "))
            options_list.append({
                "under": u,
                "type": typ,
                "contracts": contr,
                "multiplier": mult,
                "qty": contr * mult,
                "K": K,
                "T": T
            })
            print(f"    → {typ.upper()} {u}  {contr}×{mult}  K={K}")

    # Trasforma SHARES in pd.Series
    SHARES = pd.Series(SHARES)


    CONF = 0.99
    START_DATE = (pd.Timestamp.today() - BDay(300)).strftime("%Y-%m-%d")

    # 3) EQUITY DATA PREP
    START_DATE       = (pd.Timestamp.today() - BDay(300)).strftime("%Y-%m-%d")
    raw_prices       = pv.get_raw_prices(TICKERS, start=START_DATE)
    currency_map     = {t: yf.Ticker(t).fast_info.get("currency", BASE) or BASE for t in TICKERS}
    converted_prices = pv.convert_to_base(raw_prices, currency_mapping=currency_map, base_currency=BASE)
    positions_df,    = [pv.create_portfolio(converted_prices, SHARES)]
    returns, mu, cov = pv.summary_statistics(converted_prices)

    last_prices      = converted_prices.iloc[-1]
    portfolio_value  = float((last_prices * SHARES).sum())
    weights          = (last_prices * SHARES) / portfolio_value

    # Debug: risk-free
    try:
        rf_rate = yf.Ticker("^IRX").history(period="1d")["Close"].iloc[-1] / 100
    except:
        rf_rate = 0.02

    # 4) OPTIONS DATA PREP
    option_value = 0.0
    if options_list:
        all_tks      = sorted(set(TICKERS) | {o["under"] for o in options_list})
        raw_all      = pv.get_raw_prices(all_tks, start=START_DATE)
        prices_all   = pv.convert_to_base(
            raw_all,
            currency_mapping={t: BASE for t in all_tks},
            base_currency=BASE
        )
        returns_all, mu_all, cov_all = pv.summary_statistics(prices_all)
        hist_vol     = returns_all.std() * np.sqrt(252)

        for opt in options_list:
            opt.update({
                "asset_index": all_tks.index(opt["under"]),
                "sigma": float(hist_vol.get(opt["under"], hist_vol.mean())),
                "r": rf_rate
            })
            S = last_prices.get(opt["under"]) \
                or yf.Ticker(opt["under"]).history(period="1d")["Close"].iloc[-1]
            price_opt = pv.black_scholes(
                S, opt["K"], opt["T"], opt["r"], opt["sigma"], opt["type"]
            )
            option_value += price_opt * opt["qty"]

    total_value = portfolio_value + option_value

    # Debug prints
    print("\n\n=== DEBUG PORTFOLIO ===")
    # valori singole posizioni
    for ticker, qty in SHARES.items():
        price = last_prices[ticker]
        print(f"  {ticker}: {qty} × {price:.2f} = {qty * price:.2f} {BASE}")
    # valore equity
    print(f"\n  Portfolio equity value: {portfolio_value:.2f} {BASE}")
    
    # valore singole opzioni
    print("\n=== DEBUG OPTIONS ===")
    for opt in options_list:
        # prezzo unitario dell'opzione
        price_opt = pv.black_scholes(
            last_prices.get(opt["under"]) or yf.Ticker(opt["under"])
                                     .history(period="1d")["Close"].iloc[-1],
            opt["K"], opt["T"], rf_rate, opt["sigma"], opt["type"]
        )
        opt_val = opt["contracts"] * opt["multiplier"] * price_opt
        print(
            f"  {opt['contracts']}×{opt['multiplier']} {opt['type'].upper()} on {opt['under']}  "
            f"@ {price_opt:.2f} → {opt_val:.2f} {BASE}"
        )
    print(f"\n  Options total value:    {option_value:.2f} {BASE}")
    print("======================\n")
    
    # totale equity + opzioni
    print(f"  Total portfolio value:  {total_value:.2f} {BASE}")
    
    # tasso risk-free
    print(f"  Risk-free rate:         {rf_rate:.4%}")
    print("=========================\n")

    # 5) VAR + ES CALCULATION
    df_an   = pv.asset_normal_var(positions_df, confidence_level=CONF)
    var_an  = df_an["Diversified_VaR"].iloc[-1]

    var_mc_eq, pnl_mc_eq = pv.monte_carlo_var(converted_prices, SHARES.values, [], confidence_level=CONF)
    es_mc_eq = pv.simulation_es(var_mc_eq, pnl_mc_eq)

    var_mc_opt, pnl_mc_opt = pv.monte_carlo_var(converted_prices, SHARES.values, options_list, confidence_level=CONF)
    es_mc_opt = pv.simulation_es(var_mc_opt, pnl_mc_opt)

    var_hist_eq, pnl_hist_eq = pv.historical_simulation_var(converted_prices, SHARES.values, [], confidence_level=CONF)
    es_hist_eq = pv.simulation_es(var_hist_eq, pnl_hist_eq)

    var_hist_opt, pnl_hist_opt = pv.historical_simulation_var(converted_prices, SHARES.values, options_list, confidence_level=CONF)
    es_hist_opt = pv.simulation_es(var_hist_opt, pnl_hist_opt)

    
    # SINGGLE FACTOR MODEL con SPY come benchmark

    # 1) Scarica SPY e convertilo in EUR
    spy_raw = pv.get_raw_prices(["SPY"], start=START_DATE)
    # se SPY è in USD, fai la conversione reale
    spy_prices = pv.convert_to_base(
        spy_raw,
        currency_mapping={"SPY": "USD"},
        base_currency=BASE
    )
    
    # 2) Rendimenti SPY
    spy_ret = spy_prices["SPY"].pct_change().dropna()
    
    # 3) Allinea indici con returns
    common_idx         = returns.index.intersection(spy_ret.index)
    returns_aligned    = returns.loc[common_idx]
    spy_ret_aligned    = spy_ret.loc[common_idx]
    # (i pesi verranno riallineati internamente, ma puoi allinearli esplicitamente)
    weights_aligned    = weights.loc[returns_aligned.columns]
    
    # 4) Chiamata con SPY
    df_sf, vol_sf = pv.single_factor_var(
        returns_aligned,
        spy_ret_aligned,
        weights_aligned,
        portfolio_value,
        confidence_level=CONF
    )
    df_sf = pv.factor_models_es(df_sf, vol_sf, confidence_level=CONF)
    var_sf, es_sf = (
        df_sf["VaR_monetary"].iloc[-1],
        df_sf["ES_monetary"].iloc[-1]
    )


    df_ff3, vol_ff3 = pv.fama_french_var(
        returns, weights, portfolio_value, confidence_level=CONF
    )
    df_ff3          = pv.factor_models_es(df_ff3, vol_ff3, confidence_level=CONF)
    var_ff3, es_ff3 = (
        df_ff3["VaR_monetary"].iloc[-1],
        df_ff3["ES_monetary"].iloc[-1]
    )

    df_ma           = pv.ma_correlation_var(positions_df, distribution="normal", confidence_level=CONF)
    df_ma           = pv.correlation_es(df_ma)
    var_ma, es_ma   = (
        df_ma["VaR Monetary"].iloc[-1],
        df_ma["ES Monetary"].iloc[-1]
    )

    df_ewma         = pv.ewma_correlation_var(positions_df, distribution="normal", confidence_level=CONF)
    df_ewma         = pv.correlation_es(df_ewma)
    var_ewma, es_ewma = (
        df_ewma["VaR Monetary"].iloc[-1],
        df_ewma["ES Monetary"].iloc[-1]
    )

    ret_port        = returns.dot(weights)
    df_evt          = pv.evt_var(ret_port, wealth=portfolio_value)
    df_evt          = pv.evt_es(df_evt, wealth=portfolio_value)
    var_evt, es_evt = (
        df_evt["VaR_monetary"].iloc[-1],
        df_evt["ES_monetary"].iloc[-1]
    )

    df_garch_var, _     = pv.garch_var(ret_port, confidence_level=CONF, wealth=portfolio_value)
    df_garch_var        = pv.volatility_es(df_garch_var, confidence_level=CONF, wealth=portfolio_value)
    var_garch, es_garch = (
        df_garch_var["VaR_monetary"].iloc[-1],
        df_garch_var["ES_monetary"].iloc[-1]
    )
    df_garch_bt = df_garch_var[["Returns", "VaR", "VaR Violation"]]

    df_arch_var, _   = pv.arch_var(ret_port, confidence_level=CONF, wealth=portfolio_value)
    df_arch_var      = pv.volatility_es(df_arch_var, confidence_level=CONF, wealth=portfolio_value)
    df_arch_bt       = df_arch_var[["Returns", "VaR", "VaR Violation"]]

    df_ewma_var2, _  = pv.ewma_var(ret_port, confidence_level=CONF, decay_factor=0.94, wealth=portfolio_value)
    df_ewma_bt2      = df_ewma_var2[["Returns", "VaR", "VaR Violation"]]

    df_ma_var2, _    = pv.ma_var(ret_port, confidence_level=CONF, window=20, wealth=portfolio_value)
    df_ma_bt2        = df_ma_var2[["Returns", "VaR", "VaR Violation"]]

    # 6) BACKTESTING (PARAMETRIC + VOLATILITY-BASED)
    df_an_bt = pd.DataFrame({
        "Returns": ret_port,
        "VaR": df_an["Diversified_VaR"] / portfolio_value
    })
    df_an_bt["VaR Violation"] = df_an_bt["Returns"] < -df_an_bt["VaR"]

    backtest_data = {
        "Asset-Normal" : df_an_bt,
        "Sharpe-Factor": df_sf[["Returns", "VaR", "VaR Violation"]],
        "FF3-Factor"   : df_ff3[["Returns", "VaR", "VaR Violation"]],
        "GARCH(1,1)"   : df_garch_bt,
        "ARCH(p)"      : df_arch_bt,
        "EWMA(λ=0.94)" : df_ewma_bt2,
        "MA (20d)"     : df_ma_bt2,
    }

    # --- 6b) Plot backtests (interactive, each in its own browser tab) ---
    for name, df_bt in backtest_data.items():
        plot_backtest(
            df_bt,
            interactive=True,
            title=f"Backtest {name}"
        )

    # 7) ADDITIONAL PLOTS (interactive)
    plot_volatility(df_garch_var["Volatility"], interactive=True, title="Volatility Estimate")
    plot_var_series(df_an, interactive=True, title="Diversified vs Undiversified VaR")
    comp_df = pv.component_var(positions_df, confidence_level=CONF)
    plot_risk_contribution_bar(comp_df, interactive=True, title="Average Component VaR")
    plot_risk_contribution_lines(comp_df, interactive=True, title="Component VaR Over Time")
    plot_correlation_matrix(positions_df, interactive=True, title="Return Correlation Matrix")

    # 8) SINTESI – EQUITY VaR & ES
    metrics_eq = {
        "Asset-Normal VaR": var_an,
        "Sharpe-Factor VaR": var_sf,
        "FF3 VaR": var_ff3,
        "Monte Carlo VaR": var_mc_eq,
        "Hist Sim VaR": var_hist_eq,
        "MA VaR": var_ma,
        "EWMA VaR": var_ewma,
        "EVT VaR": var_evt,
        "GARCH VaR": var_garch,
        "Monte Carlo ES": es_mc_eq,
        "Hist Sim ES": es_hist_eq,
        "Sharpe-Factor ES": es_sf,
        "FF3 ES": es_ff3,
        "MA ES": es_ma,
        "EWMA ES": es_ewma,
        "EVT ES": es_evt,
        "GARCH ES": es_garch
    }

    print("\n===== SYNTHESIS – EQUITY VaR =====\n")
    var_keys = [k for k in metrics_eq if "VaR" in k]
    table_v = pd.Series({k: metrics_eq[k] for k in var_keys}, name="Value").to_frame()
    table_v["Pct_of_Port"] = table_v["Value"] / portfolio_value
    print(table_v.sort_values("Value", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\n===== SYNTHESIS – EQUITY ES =====\n")
    es_keys = [k for k in metrics_eq if "ES" in k]
    table_e = pd.Series({k: metrics_eq[k] for k in es_keys}, name="Value").to_frame()
    table_e["Pct_of_Port"] = table_e["Value"] / portfolio_value
    print(table_e.sort_values("Value", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    if options_list:
        metrics_opt = {
            "Monte Carlo VaR (EQ+OPT)": var_mc_opt,
            "Hist Sim VaR (EQ+OPT)": var_hist_opt,
            "Monte Carlo ES (EQ+OPT)": es_mc_opt,
            "Hist Sim ES (EQ+OPT)": es_hist_opt
        }
        print("\n===== SYNTHESIS – EQUITY + OPTIONS =====\n")
        tbl = pd.Series(metrics_opt, name="Value").to_frame()
        tbl["Pct_of_Port"] = tbl["Value"] / total_value
        print(tbl.sort_values("Value", ascending=False).to_string(float_format=lambda x: f"{x:,.2f}"))

    # 9) BACKTEST RESULTS: violations, rate & p-values
    records = []
    for name, df_bt in backtest_data.items():
        n_viol, rate = count_violations(df_bt)
        kup = kupiec_test(n_viol, len(df_bt), CONF)
        ch  = christoffersen_test(df_bt)
        jn  = joint_lr_test(kup["LR_uc"], ch["LR_c"])
        records.append({
            "Model": name,
            "Violations": n_viol,
            "Violation Rate": rate,
            "Kupiec p-value": kup["p_value"],
            "Christoffersen p-value": ch["p_value"],
            "Joint p-value": jn["p_value"]
        })
    results_df = pd.DataFrame(records).set_index("Model")
    print("\n===== BACKTEST RESULTS =====\n")
    print(results_df.to_string(float_format=lambda x: f"{x:.3f}"))


    # --- BUILD SUMMARY_TEXT FOR PROMPT (PATCHED: ONLY VaR) ---
    summary_lines = []
    for name, value in metrics_eq.items():
        # skip everything that non contenga "VaR"
        if "VaR" not in name:
            continue
        
        # tolgo il suffisso per creare la chiave di ricerca
        base_key = name.replace(" VaR", "")
        # match “Asset-Normal” → “Asset-Normal”, “FF3” → “FF3-Factor”, ecc.
        matches = [idx for idx in results_df.index if base_key in idx]
    
        if matches:
            d = results_df.loc[matches[0]]
            summary_lines.append(
                f"{name} has a value of {value:.2f} {BASE}, "
                f"backtest showed {int(d['Violations'])} violations "
                f"({d['Violation Rate']:.3f}), "
                f"Kupiec p={d['Kupiec p-value']:.3f}, "
                f"Christoffersen p={d['Christoffersen p-value']:.3f}, "
                f"Joint p={d['Joint p-value']:.3f}."
            )
        else:
            summary_lines.append(
                f"{name} has a value of {value:.2f} {BASE}, backtest not performed."
            )
    
    summary_text = "\n".join(summary_lines)

    print("\n===== SUMMARY TEXT =====\n"
          f"{summary_text}\n")

    # get vectorstore (or use cached summary)
    vectordb = rag.get_vectorstore([r"C:\Users\nickl\Documents\GitHub\VaR\llm\knowledge_base.pdf"])
    combined = {
        "VaR & ES Metrics": metrics_eq,
        "Backtest Summary": results_df.to_dict(orient="index")
    }

    # build prompt including summary_text prefix
    prompt = rag.build_rag_prompt(
        combined=combined,
        vectordb=vectordb,
        portfolio_value=portfolio_value,
        base=BASE,
        summary_text=summary_text  # new kwarg
    )

    llm_output = rag.ask_llm(prompt, max_tokens=1500, temperature=0.1)

    print("===== LLM INTERPRETATION =====")
    print(llm_output)

    # generate PDF
    open_report_as_pdf(
        metrics=metrics_eq,
        weights=weights,
        interpretation=llm_output,
        opt_list=options_list,
        backtest_results=results_df
    )
    print("!! PDF report generated !!")