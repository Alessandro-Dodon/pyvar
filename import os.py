import os
import subprocess
import datetime

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, t as student_t
import matplotlib.pyplot as plt
import statsmodels.api as sm
from zipfile import ZipFile
from io import BytesIO
import requests
from gpt4all import GPT4All

# ────────── Utility ──────────
def get_raw_prices(tickers, start="2024-01-01"):
    return (
        yf.download(" ".join(tickers), start=start,
                    auto_adjust=False, progress=False)["Close"]
        .ffill()
    )

def convert_to_base(raw, cur_map, base="EUR"):
    needed = {cur for cur in cur_map.values() if cur not in {base, "UNKNOWN"}}
    fx_pairs = [f"{base}{cur}=X" for cur in needed]
    fx = (
        yf.download(" ".join(fx_pairs),
                    start=raw.index[0], auto_adjust=True, progress=False)["Close"]
        .ffill() if fx_pairs else pd.DataFrame()
    )
    out = pd.DataFrame(index=raw.index)
    for t in raw.columns:
        p = raw[t] * (0.01 if cur_map[t] in {"GBp","GBX","ZAc"} else 1)
        if cur_map[t] not in {base, "UNKNOWN"}:
            rate = fx[f"{base}{cur_map[t]}=X"]
            p = p / rate
        out[t] = p
    return out

def compute_returns_stats(prices):
    rets = prices.pct_change().dropna()
    return rets, rets.mean(), rets.cov()

# ────────── VaR/CVaR ──────────
def historical_var(port_ret, alpha=5):
    varp = np.percentile(port_ret, alpha)
    cvar = port_ret[port_ret <= varp].mean()
    return -varp, -cvar

def parametric_var_cvar(mu_p, sigma_p, dist="normal", alpha=5, dof=6):
    a = alpha/100
    if dist=="normal":
        z    = norm.ppf(1 - a)
        var  = z*sigma_p - mu_p
        cvar = sigma_p * norm.pdf(norm.ppf(a)) / a - mu_p
    else:
        q    = student_t.ppf(a, dof)
        f    = student_t.pdf(q, dof)
        var  = np.sqrt((dof-2)/dof)*student_t.ppf(1 - a, dof)*sigma_p - mu_p
        cvar = -mu_p + sigma_p*(f*(dof+q**2))/(a*(dof-1))
    return var, cvar

def mc_var_cvar(mu, cov, port_val, weights, sims=5000, alpha=5):
    L = np.linalg.cholesky(cov)
    Z = np.random.standard_normal((len(mu), sims))
    sim_rets = mu.values.reshape(-1,1) + L@Z
    port_sims = port_val*(1 + weights.values@sim_rets)
    var  = port_val - np.percentile(port_sims, alpha)
    cvar = port_val - port_sims[port_sims <= np.percentile(port_sims, alpha)].mean()
    return var, cvar, port_sims

def sharpe_model_cov(returns, market):
    σm2    = market.var(ddof=0)
    cov_im = returns.apply(lambda x: x.cov(market))
    betas  = cov_im/σm2
    idio   = returns.var(ddof=0) - betas.pow(2)*σm2
    tickers= returns.columns
    outer  = np.outer(betas, betas)*σm2
    Sigma  = pd.DataFrame(outer, index=tickers, columns=tickers)
    for t in tickers:
        Sigma.at[t,t] += idio[t]
    return Sigma, betas, idio, σm2

def compute_ff3factor_var(tickers, shares, base="EUR", start="2024-01-01", alpha=0.95):
    raw    = get_raw_prices(tickers, start=start)
    cur_map= {t: yf.Ticker(t).fast_info.get("currency", base) or base for t in tickers}
    for t in tickers:
        if cur_map[t] in {"GBp","GBX"}: cur_map[t] = "GBP"
        if cur_map[t]=="ZAc":            cur_map[t] = "ZAR"
    prices = convert_to_base(raw, cur_map, base)
    rets   = prices.pct_change().dropna()

    url  = ("https://mba.tuck.dartmouth.edu/pages/faculty/"
            "ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip")
    zf   = ZipFile(BytesIO(requests.get(url).content))
    csvf = next(n for n in zf.namelist() if n.endswith(".CSV"))
    ff   = pd.read_csv(zf.open(csvf), skiprows=3, index_col=0)
    mask = ff.index.astype(str).str.match(r"^\d{8}$")
    ff   = ff.loc[mask].astype(float)/100
    ff.index = pd.to_datetime(ff.index.astype(str), format="%Y%m%d")
    ff.columns= ["Mkt_RF","SMB","HML","RF"]
    ff        = ff.sort_index().loc[start:]
    ff        = ff.reindex(rets.index).ffill()

    excess = rets.sub(ff["RF"], axis=0)
    X      = sm.add_constant(ff[["Mkt_RF","SMB","HML"]])
    betas, idio = {}, {}
    for t in tickers:
        tmp = pd.concat([excess[t], X], axis=1).dropna()
        res = sm.OLS(tmp.iloc[:,0], tmp.iloc[:,1:]).fit()
        betas[t] = res.params.drop("const")
        idio[t]  = res.resid.var(ddof=0)
    B       = pd.DataFrame(betas).T.values
    Σf      = ff[["Mkt_RF","SMB","HML"]].cov().values
    Σ       = B@Σf@B.T + np.diag(pd.Series(idio).values)

    last    = prices.iloc[-1]
    port_val= (last*pd.Series(shares)).sum()
    w       = (last*pd.Series(shares))/port_val
    var_p   = w.values@Σ@w.values
    sigma_p = np.sqrt(var_p)
    z       = norm.ppf(alpha)
    VaR     = z*sigma_p*port_val
    tail    = 1-alpha
    CVaR    = sigma_p*norm.pdf(norm.ppf(tail))/tail*port_val
    return VaR, CVaR



# ────────── PDF Report ──────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles   import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums    import TA_CENTER, TA_LEFT
from reportlab.platypus     import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units    import cm
from reportlab.lib          import colors
import os
import datetime


def save_report_as_pdf(metrics: dict,
                       weights: pd.Series,
                       interpretation: str,
                       filename: str = "interpretation_report.pdf"):

    # setup documento
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    if "RptTitle" not in styles:
        styles.add(ParagraphStyle("RptTitle",
                                  fontName="Times-Roman",
                                  fontSize=24,
                                  alignment=TA_CENTER,
                                  spaceAfter=12))
    if "SectHead" not in styles:
        styles.add(ParagraphStyle("SectHead",
                                  fontName="Times-Roman",
                                  fontSize=18,
                                  spaceAfter=6))
    if "BodyTxt" not in styles:
        styles.add(ParagraphStyle("BodyTxt",
                                  fontName="Times-Roman",
                                  fontSize=12,
                                  leading=14,
                                  spaceAfter=4))

    story = []
    # titolo
    story.append(Paragraph("Interpretation Report", styles["RptTitle"]))
    story.append(Paragraph(f"Date: {datetime.date.today():%d %B %Y}", styles["BodyTxt"]))
    story.append(Spacer(1, 0.7*cm))

    # --- Metrics ---
    story.append(Paragraph("Portfolio Risk Metrics", styles["SectHead"]))
    story.append(Spacer(1, 0.3*cm))  # extra spazio
    data = [["Metric", "Value"]]
    for k, v in metrics.items():
        data.append([k, f"{v:,.2f}"])
    tbl = Table(data, colWidths=[8*cm, 6*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN",       (1,1), (-1,-1), "RIGHT"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME",    (0,0), (-1,-1), "Times-Roman"),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.7*cm))

    # --- Weights ---
    story.append(Paragraph("Portfolio Weights", styles["SectHead"]))
    story.append(Spacer(1, 0.3*cm))
    wdata = [["Ticker", "Weight (%)"]]
    for t, wt in weights.items():
        wdata.append([t, f"{wt*100:.2f}"])
    wtbl = Table(wdata, colWidths=[4*cm, 4*cm])
    wtbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("GRID",  (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Times-Roman"),
    ]))
    story.append(wtbl)
    story.append(Spacer(1, 0.7*cm))

    

    # --- LLM Interpretation ---
    story.append(Paragraph("LLM Interpretation", styles["SectHead"]))
    story.append(Spacer(1, 0.3*cm))
    # rimuovo eventuale prima riga "Correlation Matrix:..."
    lines = interpretation.splitlines()
    if lines and lines[0].startswith("Correlation Matrix"):
        lines.pop(0)
    clean_interp = "\n".join(lines).strip()
    for para in clean_interp.split("\n\n"):
        story.append(Paragraph(para.replace("\n"," "), styles["BodyTxt"]))
        story.append(Spacer(1, 0.3*cm))

    # costruisco PDF e lo apro
    doc.build(story)
    path = os.path.abspath(filename)
    os.startfile(path)
    print(f"✅ Report PDF generato e aperto: {path}")





# ────────── Main ──────────
if __name__ == "__main__":
    BASE   = input("Base currency (EUR): ").upper()
    TKS    = input("Tickers (sep space): ").upper().split()
    SHARES = pd.Series({t: float(input(f"Shares of {t}: ")) for t in TKS})
    START  = "2024-01-01"

    # download, convert, stats
    raw      = get_raw_prices(TKS, START)
    cur_map  = {t:(yf.Ticker(t).fast_info or {}).get("currency", BASE) for t in TKS}
    prices   = convert_to_base(raw, cur_map, BASE)
    rets, mu, cov = compute_returns_stats(prices)

    # portfolio
    last     = prices.iloc[-1]
    port_val = (last*SHARES).sum()
    w        = (last*SHARES)/port_val

    # VaR/CVaR
    hVar,  hCVar  = historical_var(rets.dot(w), alpha=5)
    nVar,  nCVar  = parametric_var_cvar(w.dot(mu),
                                        np.sqrt(w.values@cov.values@w.values),
                                        "normal", alpha=5)
    tVar,  tCVar  = parametric_var_cvar(w.dot(mu),
                                        np.sqrt(w.values@cov.values@w.values),
                                        "t-distribution", alpha=5, dof=6)
    mcVar, mcCVar, sims = mc_var_cvar(mu, cov, port_val, w, sims=5000, alpha=5)

    spy_price = convert_to_base(get_raw_prices(["SPY"], START),
                                {"SPY":"USD"}, BASE)["SPY"]
    market    = spy_price.pct_change().dropna()
    Sigma_sh, betas_sh, idio_sh, _ = sharpe_model_cov(rets, market)
    sigma_sh  = np.sqrt(w.values@Sigma_sh.values@w.values)
    sVar      = norm.ppf(0.95)*sigma_sh*port_val
    sCVar     = sigma_sh*norm.pdf(norm.ppf(0.05))/0.05*port_val

    ffVar, ffCVar = compute_ff3factor_var(TKS, SHARES, BASE, START, alpha=0.95)

    # stampa a console
    print(f"\nPortfolio Value: {port_val:,.2f} {BASE}")
    for label,value in [
        ("Historical VaR 95%",   hVar*port_val),
        ("Parametric Normal VaR",nVar*port_val),
        ("Parametric t VaR",     tVar*port_val),
        ("MC VaR 95%",           mcVar),
        ("Sharpe-model VaR 95%", sVar),
        ("FF-3factor VaR 95%",   ffVar)
    ]:
        print(f"  {label:25s}: {value:,.2f}")
    print()
    for label,value in [
        ("Historical CVaR 95%",   hCVar*port_val),
        ("Parametric Normal CVaR",nCVar*port_val),
        ("Parametric t CVaR",     tCVar*port_val),
        ("MC CVaR 95%",           mcCVar),
        ("Sharpe-model CVaR 95%", sCVar),
        ("FF-3factor CVaR 95%",   ffCVar)
    ]:
        print(f"  {label:25s}: {value:,.2f}")

    # prepare metrics dict
    metrics = {
        "Historical VaR 95%":      hVar*port_val,
        "Historical CVaR 95%":     hCVar*port_val,
        "Parametric Normal VaR":   nVar*port_val,
        "Parametric Normal CVaR":  nCVar*port_val,
        "Parametric t VaR":        tVar*port_val,
        "Parametric t CVaR":       tCVar*port_val,
        "MC VaR 95%":              mcVar,
        "MC CVaR 95%":             mcCVar,
        "Sharpe-Model VaR 95%":    sVar,
        "Sharpe-Model CVaR 95%":   sCVar,
        "FF-3factor VaR 95%":      ffVar,
        "FF-3factor CVaR 95%":     ffCVar,
    }





    # load LLM offline
    print("\nLoading LLM model…")
    model = GPT4All(model_name="DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",  #QUA METTETE IL MODELLO CHE AVETE SCARICATO
                    model_path=r"C:/Users/nickl/AppData/Local/nomic.ai/GPT4All", #QUA IL PERCORSO DEL MODELLO 
                    allow_download=False, verbose=False)

    prompt = (f"You are a senior financial analyst. Interpret these metrics: {metrics}") #modifica del prompt

    response = model.generate(prompt, max_tokens=10) #Lunghezza massima della risposta
    print("\n--- LLM Interpretation ---\n")
    print(response)


    # genera e apre il PDF
    save_report_as_pdf(metrics, w, response)

    # infine mostra istogramma Monte Carlo
    plt.hist(sims, bins=50, edgecolor='k')
    plt.axvline(np.percentile(sims,5), color='r', linestyle='--', label='VaR95')
    plt.title('MC Simulated Portfolio Values')
    plt.xlabel('Portfolio Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()