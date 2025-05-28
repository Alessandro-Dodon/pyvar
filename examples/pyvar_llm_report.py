#================================================================
# VaR and ES Risk Report for Equity + Options Portfolio 
# (Calculate VaR(CONF;1), backtest it, and generate a PDF report with pdf interpretation)
#================================================================

import os, sys
import pandas as pd
import numpy as np
import yfinance as yf
from pandas.tseries.offsets import BDay
import pandas_datareader.data as web
import pyvar as pv
from pyvar.backtesting import count_violations, kupiec_test, christoffersen_test, joint_lr_test
from pyvar.plots import (
    plot_backtest, plot_volatility, plot_var_series,
    plot_risk_contribution_bar, plot_risk_contribution_lines,
    plot_correlation_matrix
)


# ----------------------------------------------------------
# PROJECT-ROOT for llm_rag and pdf_reporting
# ----------------------------------------------------------
project_root = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ----------------------------------------------------------
# CONFIGURATION CONSTANTS
# ----------------------------------------------------------
CONFIDENCE_LEVEL = 0.99 # Confidence level for VaR and ES calculations
LOOKBACK_BUSINESS_DAYS = 300 # Number of business days to include in the analysis


# ----------------------------------------------------------
# OPTIONAL FEATURES (set to False to skip)
# ----------------------------------------------------------
SHOW_PLOTS = True        # when False, skips all interactive charts
RUN_LLM_INTERPRETATION = True  # when False, skips the LLM call & PDF

# LLM endpoint & model
LMSTUDIO_ENDPOINT = "http://127.0.0.1:1234"
API_PATH          = "/v1/completions"
MODEL_NAME        = "qwen-3-4b-instruct"


# ----------------------------------------------------------
# HELPERS
# ----------------------------------------------------------
def compute_var_and_es(var_func, *args, **kwargs):
    var, pnl = var_func(*args, **kwargs)
    return var, pv.simulation_es(var, pnl)

def summarize_backtest(df):
    violations, rate = count_violations(df)
    kup = kupiec_test(violations, len(df), CONFIDENCE_LEVEL)
    ch  = christoffersen_test(df)
    jn  = joint_lr_test(kup["LR_uc"], ch["LR_c"])
    return violations, rate, kup["p_value"], ch["p_value"], jn["p_value"]

def _auto_show_wrapper(fn):
    def inner(*args, interactive=True, title=None, **kwargs):
        fig = fn(*args, interactive=interactive, **kwargs)
        if fig is not None and interactive:
            if title: fig.update_layout(title=title)
            fig.show()
        return fig
    return inner

# patch plots
plot_backtest                = _auto_show_wrapper(plot_backtest)
plot_volatility              = _auto_show_wrapper(plot_volatility)
plot_var_series              = _auto_show_wrapper(plot_var_series)
plot_risk_contribution_bar   = _auto_show_wrapper(plot_risk_contribution_bar)
plot_risk_contribution_lines = _auto_show_wrapper(plot_risk_contribution_lines)
plot_correlation_matrix      = _auto_show_wrapper(plot_correlation_matrix)


if __name__ == "__main__":

    CONF = CONFIDENCE_LEVEL  # Confidence level for VaR and ES calculations
    days_window = LOOKBACK_BUSINESS_DAYS   #Number of business days to include in the analysis
    START_DATE = (pd.Timestamp.today() - BDay(days_window)).strftime("%Y-%m-%d")

    # ——————————————————————————————
    # 1) INTERACTIVE INPUTS
    # ——————————————————————————————
    
    # Base currency
    BASE = input("Choose a base currency [e.g. EUR]: ").strip().upper() or "EUR"

    # Equity tickers
    TICKERS = input("Enter equity tickers (space separated): ").upper().split()
    if not TICKERS:
        print("No tickers entered, exiting.")
        sys.exit(1)

    # Number of shares per ticker
    SHARES = {}
    for t in TICKERS:
        while True:
            try:
                q = float(input(f"  Number of shares for {t:<6}: "))
                SHARES[t] = q
                break
            except ValueError:
                print("    Invalid number, please try again.")

    # Options
    options_list = []
    if input("Do you have options? (y/n): ").lower().startswith("y"):
        while True:
            u = input("  Underlying (ENTER to finish): ").upper()
            if not u:
                break
            typ = input("    Type (call/put): ").lower()
            contr = float(input("    Number of contracts: "))
            mult = int(input("    Multiplier [100]: ") or 100)
            K = float(input("    Strike price: "))
            T = float(input("    Time to maturity (years): "))
            options_list.append({
                "under": u,
                "type": typ,
                "contracts": contr,
                "multiplier": mult,
                "qty": contr * mult,
                "K": K,
                "T": T
            })
            print(f"    → {typ.upper()} {u}  {contr}×{mult}  Strike={K}")


    # Transform SHARES in pd.Series
    SHARES = pd.Series(SHARES)


    # 3) EQUITY DATA PREP
    # Fetch, fill, convert in one go
    raw = pv.get_raw_prices(TICKERS, start=START_DATE).ffill().bfill()
    currency_map = {t: yf.Ticker(t).fast_info.get("currency", BASE) or BASE for t in TICKERS}
    converted_prices = pv.convert_to_base(raw, currency_mapping=currency_map, base_currency=BASE).ffill().bfill()

    # Drop tickers that have no valid data at all
    #  (e.g. AAPL failed to download)
    valid_equities = raw.columns[raw.notna().any()]
    dropped = set(raw.columns) - set(valid_equities)
    if dropped:
        print(f"[warning] dropping tickers with no data: {dropped}")
        raw = raw[valid_equities]
        converted_prices = converted_prices[valid_equities]
        SHARES = SHARES.loc[valid_equities]

    # 2) Recompute returns and check length
    returns = pv.compute_returns(converted_prices)
    if len(returns) < 2:
        sys.exit("Not enough data points after cleaning to compute VaR.")

    # Portfolio stats
    positions_df = pv.create_portfolio(converted_prices, SHARES)
    returns = pv.compute_returns(converted_prices)
    last_prices = converted_prices.iloc[-1]
    portfolio_value = (last_prices * SHARES).sum()
    weights = (last_prices * SHARES) / portfolio_value

    # Risk‐free
    try:
        df_rf = web.DataReader("DGS3", "fred",
                            pd.Timestamp.today() - pd.Timedelta(days=7),
                            pd.Timestamp.today()).dropna()
        rf_rate = df_rf.iloc[-1, 0] / 100
    except Exception:
        rf_rate = 0.02

    # 4) OPTIONS DATA PREP
    option_value = 0.0
    if options_list:
        # 1) Aggregate all underlyings + equities in one fetch
        all_tickers = sorted(set(TICKERS) | {opt['under'] for opt in options_list})
        raw_all = pv.get_raw_prices(all_tickers, start=START_DATE).ffill().bfill()
        converted_all = pv.convert_to_base(
            raw_all,
            currency_mapping={t: BASE for t in all_tickers},
            base_currency=BASE
        ).ffill().bfill()

        # 2) Historical volatility for each ticker
        returns_all = pv.compute_returns(converted_all).dropna()
        hist_vol = returns_all.std() * np.sqrt(252)

        # 3) Map column order for Monte Carlo and compute each option
        price_cols = list(converted_all.columns)
        for opt in options_list:
            if opt['under'] not in price_cols:
                raise ValueError(f"Underlying {opt['under']} not in price data")
            opt['asset_index'] = price_cols.index(opt['under'])
            opt['sigma'] = float(hist_vol.get(opt['under'], hist_vol.mean()))
            opt['r'] = rf_rate

            # get last spot directly from the converted series
            spot_price = converted_all[opt['under']].iloc[-1]
            unit_price = pv.black_scholes(
                spot_price, opt['K'], opt['T'], opt['r'], opt['sigma'], opt['type']
            )
            option_value += unit_price * opt['qty']

    total_value = portfolio_value + option_value


     # === DEBUG SUMMARY TABLES ===
    # Portfolio positions
    positions_summary = pd.DataFrame({
        'Ticker': SHARES.index,
        'Shares': SHARES.values,
        'Price': last_prices.loc[SHARES.index].values
    })
    positions_summary['Value'] = positions_summary['Shares'] * positions_summary['Price']
    print("\n\n=== EQUITY POSITIONS ===")
    print(positions_summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\n  Portfolio equity value: {portfolio_value:.2f} {BASE}\n")

    # Options positions (if any)
    if options_list:
        options_summary = pd.DataFrame(options_list)
        options_summary['MarketPrice'] = options_summary.apply(
            lambda row: pv.black_scholes(
                last_prices.get(row['under'])
                or yf.Ticker(row['under']).history(period="1d")['Close'].iloc[-1],
                row['K'], row['T'], rf_rate, row['sigma'], row['type']
            ), axis=1
        )
        options_summary['Value'] = (
            options_summary['contracts']
            * options_summary['multiplier']
            * options_summary['MarketPrice']
        )
        display_cols = ['under', 'type', 'contracts', 'multiplier', 'MarketPrice', 'Value']
        print("=== OPTIONS POSITIONS ===")
        print(
            options_summary[display_cols]
                .rename(columns={
                    'under':'Underlying','type':'OptionType',
                    'contracts':'Contracts','multiplier':'Multiplier',
                    'MarketPrice':'Price','Value':'Value'
                })
                .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )
        print(f"\n  Options total value: {option_value:.2f} {BASE}\n")

    # Totals
    print(f"=== TOTAL PORTFOLIO VALUE: {total_value:.2f} {BASE} ===")
    print(f"=== RISK-FREE RATE: {rf_rate:.4%} ===\n")


    # 5) VAR + ES CALCULATION
    returns_portfolio = returns.dot(weights)

    #  — Asset Normal 
    df_asset_normal = pv.asset_normal_var(positions_df, confidence_level=CONF)
    var_asset_normal = df_asset_normal["Diversified_VaR"].iloc[-1]

    #  — Monte Carlo SImulations (equity only)
    var_mc_eq, es_mc_eq = compute_var_and_es(
        pv.monte_carlo_var,
        converted_prices, SHARES.values, [], 
        confidence_level=CONF
    )

    #  — Monte Carlo Simulations (equity + options)
    var_mc_opt, es_mc_opt = compute_var_and_es(
        pv.monte_carlo_var,
        converted_prices, SHARES.values, options_list,
        confidence_level=CONF
    )

    #  — Historical Simulations (equity only)
    var_hist_sim_eq, es_hist_sim_eq = compute_var_and_es(
        pv.historical_simulation_var,
        converted_prices, SHARES.values, [],
        confidence_level=CONF
    )

    #  — Historical Simulations (equity + options)
    var_hist_sim_opt, es_hist_sim_opt = compute_var_and_es(
        pv.historical_simulation_var,
        converted_prices, SHARES.values, options_list,
        confidence_level=CONF
    )

    # Sharpe model using SPY as benchmark
    spy_raw = pv.get_raw_prices(["SPY"], start=START_DATE)
    spy_prices = pv.convert_to_base(spy_raw, {"SPY":"USD"}, base_currency=BASE).ffill().bfill()
    spy_ret = spy_prices["SPY"].pct_change().dropna()
    common_idx = returns.index.intersection(spy_ret.index)
    aligned_returns = returns.loc[common_idx]
    aligned_spy     = spy_ret.loc[common_idx]
    aligned_weights = weights.loc[aligned_returns.columns]
    df_sharpe_model, vol_sharpe_model = pv.single_factor_var(
        aligned_returns, aligned_spy, aligned_weights,
        portfolio_value, confidence_level=CONF
    )
    df_sharpe_model = pv.factor_models_es(df_sharpe_model, vol_sharpe_model, confidence_level=CONF)
    var_sharpe_model, es_sharpe_model = df_sharpe_model["VaR_monetary"].iat[-1], df_sharpe_model["ES_monetary"].iat[-1]


    # Fama-French 3 Factor Model
    df_ff3, vol_ff3 = pv.fama_french_var(
        returns, weights, portfolio_value, confidence_level=CONF
    )
    df_ff3 = pv.factor_models_es(df_ff3, vol_ff3, confidence_level=CONF)
    var_ff3, es_ff3 = df_ff3["VaR_monetary"].iat[-1], df_ff3["ES_monetary"].iat[-1]


    # Volatility-based Moving Average & EWMA VaR
    df_ma_var, _ = pv.ma_var(
        returns_portfolio,
        confidence_level=CONF,
        window=20,
        wealth=portfolio_value
    )
    df_ewma_var, _ = pv.ewma_var(
        returns_portfolio,
        confidence_level=CONF,
        decay_factor=0.94,
        wealth=portfolio_value
    )

    # Moving Average (ma) & Exponential Weighted Moving Average (ewma) correlations
    df_ma = pv.ma_correlation_var(
        positions_df,
        distribution="normal",
        confidence_level=CONF
    )
    df_ma = pv.correlation_es(df_ma)
    var_ma, es_ma = df_ma["VaR Monetary"].iat[-1], df_ma["ES Monetary"].iat[-1]

    df_ewma = pv.ewma_correlation_var(
        positions_df,
        distribution="normal",
        confidence_level=CONF
    )
    df_ewma = pv.correlation_es(df_ewma)
    var_ewma, es_ewma = df_ewma["VaR Monetary"].iat[-1], df_ewma["ES Monetary"].iat[-1]


    # Extreme Value Theory (EVT)
    df_evt   = pv.evt_var(returns_portfolio, wealth=portfolio_value)
    df_evt   = pv.evt_es(df_evt, wealth=portfolio_value)
    var_evt, es_evt = df_evt["VaR_monetary"].iat[-1], df_evt["ES_monetary"].iat[-1]

    # GARCH(1,1)
    df_garch, intercept = pv.garch_var(returns_portfolio, confidence_level=CONF, wealth=portfolio_value)
    df_garch    = pv.volatility_es(df_garch, confidence_level=CONF, wealth=portfolio_value)
    var_garch, es_garch = df_garch["VaR_monetary"].iat[-1], df_garch["ES_monetary"].iat[-1]

    # Component VaR
    component_var_df = pv.component_var(positions_df, confidence_level=CONF)

    # 6) BACKTESTING
    # Map each model name to its DataFrame of VaR results
    model_data = {
        "Asset-Normal":     df_asset_normal,
        "Sharpe-Factor":    df_sharpe_model,
        "FF3-Factor":       df_ff3,
        "GARCH(1,1)":       df_garch,
        "EWMA(λ=0.94)":     df_ewma_var,
        "MA (20d)":         df_ma_var,
        "EVT":               df_evt
    }

    backtest_data = {}
    for name, df_model in model_data.items():
        # pick the last column containing "VaR"
        var_col = [c for c in df_model.columns if "VaR" in c][-1]
        # build the backtest DataFrame
        df_bt = pd.DataFrame({
            "Returns": returns_portfolio,
            "VaR":      df_model[var_col] / portfolio_value
        }, index=returns_portfolio.index)
        df_bt["VaR Violation"] = df_bt["Returns"] < -df_bt["VaR"]
        backtest_data[name] = df_bt

    
    
    # 7) PLOT DATA  (only if SHOW_PLOTS)
    if SHOW_PLOTS:
        for model_name, df_bt in backtest_data.items():
            plot_backtest(df_bt, interactive=True, title=f"Backtest {model_name}")

        additional_plots = [
            (plot_volatility,          df_garch["Volatility"],                "Volatility Estimate"),
            (plot_var_series,          df_asset_normal,                       "Diversified vs Undiversified VaR"),
            (plot_risk_contribution_bar,   component_var_df,                   "Average Component VaR"),
            (plot_risk_contribution_lines, component_var_df,                   "Component VaR Over Time"),
            (plot_correlation_matrix,  positions_df,                          "Return Correlation Matrix"),
        ]
        for fn, data, title in additional_plots:
            fn(data, interactive=True, title=title)


    # 8) SYNTHESIS – EQUITY VaR & ES
    # define raw VaR and ES mappings
    var_table = {
        "Asset-Normal":       var_asset_normal,
        "Sharpe-Factor":      var_sharpe_model,
        "FF3":                var_ff3,
        "Monte Carlo":        var_mc_eq,
        "Hist Sim":           var_hist_sim_eq,
        "MA":                  var_ma,
        "EWMA":                var_ewma,
        "EVT":                 var_evt,
        "GARCH":               var_garch
    }

    es_table = {
        "Monte Carlo":        es_mc_eq,
        "Hist Sim":           es_hist_sim_eq,
        "Sharpe-Factor":      es_sharpe_model,
        "FF3":                es_ff3,
        "MA":                  es_ma,
        "EWMA":                es_ewma,
        "EVT":                 es_evt,
        "GARCH":               es_garch
    }

    # build the final metrics_eq
    metrics_eq = {
        f"{name} VaR": value
        for name, value in var_table.items()
    }
    metrics_eq.update({
        f"{name} ES": value
        for name, value in es_table.items()
    })

    # 8b) SYNTHESIS PRINTOUTS

    # Equity metrics DataFrame
    df_metrics = (
    pd.DataFrame.from_dict(metrics_eq, orient='index', columns=['Value'])
      .assign(
          Type=lambda df: np.where(df.index.str.contains('VaR'), 'VaR', 'ES'),
          Pct_of_Port=lambda df: df['Value'] / portfolio_value
      )
    )

    # Print VaR and ES tables
    for metric_type in ['VaR', 'ES']:
        title = f"\n===== SYNTHESIS – EQUITY {metric_type} =====\n"
        print(title)
        df_sub = df_metrics[df_metrics['Type'] == metric_type] \
                    .sort_values('Value', ascending=False)
        print(
            df_sub[['Value','Pct_of_Port']]
            .to_string(
                float_format=lambda x: f"{x:,.2f}"
            )
        )

    # Options metrics, if any
    if options_list:
        metrics_opt = {
            "Monte Carlo VaR (EQ+OPT)": var_mc_opt,
            "Hist Sim VaR (EQ+OPT)":     var_hist_sim_opt,
            "Monte Carlo ES (EQ+OPT)":   es_mc_opt,
            "Hist Sim ES (EQ+OPT)":      es_hist_sim_opt
        }
        df_opt = (
            pd.DataFrame.from_dict(metrics_opt, orient='index', columns=['Value'])
            .assign(Pct_of_Port=lambda df: df['Value'] / total_value)
            .sort_values('Value', ascending=False)
        )
        print("\n===== SYNTHESIS – EQUITY + OPTIONS =====\n")
        print(
            df_opt[['Value','Pct_of_Port']]
            .to_string(
                float_format=lambda x: f"{x:,.2f}"
            )
        )


    # --- 9) BACKTEST RESULTS: violations, rates & p-values ---

    # Build a DataFrame directly from the summaries
    results_df = pd.DataFrame.from_dict(
        {
            name: summarize_backtest(df_bt)
            for name, df_bt in backtest_data.items()
        },
        orient="index",
        columns=[
            "Violations",
            "Violation Rate",
            "Kupiec p-value",
            "Christoffersen p-value",
            "Joint p-value"
        ]
    )

    print("\n===== BACKTEST RESULTS =====\n")
    print(results_df.to_string(float_format=lambda x: f"{x:.3f}"))



    # --- BUILD SUMMARY TEXT FOR PROMPT LLM (ONLY for VaR) ---

    # Precompute backtest summaries by model name
    backtest_summaries = {
        model: (
            f"backtest showed {int(row['Violations'])} violations "
            f"({row['Violation Rate']:.3f}), "
            f"Kupiec p={row['Kupiec p-value']:.3f}, "
            f"Christoffersen p={row['Christoffersen p-value']:.3f}, "
            f"Joint p={row['Joint p-value']:.3f}"
        )
        for model, row in results_df.iterrows()
    }

    # Build the summary lines in one pass
    summary_lines = []
    for metric_name, value in metrics_eq.items():
        if not metric_name.endswith(" VaR"):
            continue
        base_key = metric_name[:-4]  # strip " VaR"
        # find the first backtest model whose name contains this key
        match = next((m for m in backtest_summaries if base_key in m), None)
        stats = backtest_summaries[match] if match else "backtest not performed"
        summary_lines.append(
            f"{metric_name} has a value of {value:.2f} {BASE}, {stats}."
        )

    summary_text = "\n".join(summary_lines)

    print("\n===== SUMMARY TEXT =====\n" + summary_text + "\n")




    # 10) LLM INTERPRETATION & PDF REPORT (only if RUN_LLM_INTERPRETATION)
    if RUN_LLM_INTERPRETATION:
        import llm.llm_rag as rag
        from llm.pdf_reporting import open_report_as_pdf

        rag.LMSTUDIO_ENDPOINT = LMSTUDIO_ENDPOINT
        rag.API_PATH          = API_PATH
        rag.MODEL_NAME        = MODEL_NAME


        vector_store = rag.get_vectorstore(r"llm\knowledge_base.pdf")
        combined_content = {
            "VaR & ES Metrics": metrics_eq,
            "Backtest Summary": results_df.to_dict(orient="index")
        }
        prompt = rag.build_rag_prompt(
            combined=combined_content,
            vectordb=vector_store,
            portfolio_value=portfolio_value,
            base=BASE,
            confidence_level=CONFIDENCE_LEVEL,
            summary_text=summary_text
        )
        interpretation = rag.ask_llm(prompt, max_tokens=1000, temperature=0.1)
        print("===== LLM INTERPRETATION =====")
        print(interpretation)

        open_report_as_pdf(
            metrics=metrics_eq,
            weights=weights,
            interpretation=interpretation,
            opt_list=options_list,
            backtest_results=results_df
        )
        print("!! PDF report generated !!")