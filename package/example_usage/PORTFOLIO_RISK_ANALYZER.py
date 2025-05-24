#================================================================
# VaR and ES Risk Report for Equity + Options Portfolio (with plots)
# ================================================================
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
    # 1) INPUT UTENTE
    BASE     = "EUR"
    TICKERS  = ["NVDA", "MSFT"]
    SHARES   = pd.Series({"NVDA": 3, "MSFT": 4})
    CONF     = 0.99

    # --- LLM CONFIGURATION ---
    rag.LMSTUDIO_ENDPOINT = "http://127.0.0.1:1234"
    rag.API_PATH          = "/v1/completions"
    rag.MODEL_NAME        = "qwen-3-4b-instruct"
    #--------------------------------------------------------------

    # 2) OPTIONS INPUT (demo)
    options_list = [
        {"under": "AAPL", "type": "call", "contracts": 1, "multiplier": 100,
         "qty": 100, "K": 210.0, "T": 1.0}
    ]

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

    # 5) VAR + ES CALCULATION
    df_an   = pv.asset_normal_var(positions_df, confidence_level=CONF)
    var_an  = df_an["Diversified_VaR"].iloc[-1]

    var_mc_eq, pnl_mc_eq = pv.monte_carlo_var(converted_prices, SHARES.values, [])
    es_mc_eq = pv.simulation_es(var_mc_eq, pnl_mc_eq)

    var_mc_opt, pnl_mc_opt = pv.monte_carlo_var(converted_prices, SHARES.values, options_list)
    es_mc_opt = pv.simulation_es(var_mc_opt, pnl_mc_opt)

    var_hist_eq, pnl_hist_eq = pv.historical_simulation_var(converted_prices, SHARES.values, [])
    es_hist_eq = pv.simulation_es(var_hist_eq, pnl_hist_eq)

    var_hist_opt, pnl_hist_opt = pv.historical_simulation_var(converted_prices, SHARES.values, options_list)
    es_hist_opt = pv.simulation_es(var_hist_opt, pnl_hist_opt)

    benchmark      = converted_prices[TICKERS[0]].pct_change().dropna()
    df_sf, vol_sf  = pv.single_factor_var(
        returns, benchmark, weights, portfolio_value, confidence_level=CONF
    )
    df_sf          = pv.factor_models_es(df_sf, vol_sf, confidence_level=CONF)
    var_sf, es_sf  = (
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

    df_ma           = pv.ma_correlation_var(positions_df, distribution="normal")
    df_ma           = pv.correlation_es(df_ma)
    var_ma, es_ma   = (
        df_ma["VaR Monetary"].iloc[-1],
        df_ma["ES Monetary"].iloc[-1]
    )

    df_ewma         = pv.ewma_correlation_var(positions_df, distribution="normal")
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


    # ----------------------------------------------------------------
    # 10) LLM INTERPRETATION via RAG
    # ----------------------------------------------------------------
    # 10a) crea/carica il vectorstore dai tuoi PDF di contesto
    vectordb = get_vectorstore(["/path/alla/tuadocumentazione.pdf"])
    # 10b) prepara il dizionario completo da passare al prompt
    combined = {
        "VaR & ES Metrics": metrics_eq,          # il tuo dict con VaR, ES ecc.
        "Backtest Summary": results_df.to_dict()
    }
    prompt = build_rag_prompt(combined, vectordb, portfolio_value, BASE)
    llm_output = ask_llm(prompt, max_tokens=100, temperature=0.2)

    print("\n===== LLM INTERPRETATION =====\n")
    print(llm_output)
