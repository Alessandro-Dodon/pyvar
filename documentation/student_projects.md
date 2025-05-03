# Student Projects Programming in Finance II
peter.gruber@usi.ch, 2025-04-16

## Python package for VaR
Dodon, Gasparetti, Lecce (13:39)
- flexible VaR calculation
- focus on equity
- modules
    - backtest
    - VaR models (historical, parametric  e.g. GARCH, Monte Carlo)
    - metrics: violations
    - LLM for explanation
    - tutorial
- TODO
    - knowledge base for LLM
    - implementation of existing standard approaches ("JP Morgan method")
    - measures beyond VaR (e.g. CVaR, ES)

## Traffic concerns in Ticino
Group "I"
- how does traffic impact environment to society in Ticino?
- policy implications
- create Python script
- Data sources
- modules
    - data
    - data scraping, cleaing etc.
    - data analysis and visualization
    - predictions
    - LLM interpretation
- TODO
    - modular code with functions
    - google traffic data (?)
    - google maps layer (?)
    - predictions --> scenarios
    - be considerate about data analysis and presentation

## Earnings Watcher (UPDATE?)
Group "H"
- earnings reports are often too long and have low information content
- Automated retrieval plus LLM to summarize earnings reports
- Better than ChatGPT
- modules
    - watch companies
    - RAG pipeline (needed?)
    - clean human readable summary
    - Progressive web app
- TODO
    - customization of summary ("I am interested in ...", style, ...)
    - design?
    - knowledge base for LLM, including company descriptions
    - RLHF?
    - train LLM --> fine-tuning or in-context learning
    - cron daily -> every 15min
    - email alert
    - Chatbot may be too much

## Marc, the virtual financial assistant
(presented later)

## Port Balance – LLM-assisted smart portfolio (UPDATE)
Acquistapace et al
- chatbot for 
- modules
    - web scraping for data (why scrape?)
    - python package for risk analyis
    - LLM API **usage**
    - Chatbot
    - web interface streamlit/flask
- TODO
    - what is the goal? people may not know which question to ask
    - user journey?
    - risk profile?
    - what do you intend with scalability?

## LLM-assisted strategy builder and backtester
Ferrario, Scorpioni, Pizzuto
- Problem: people cannot code
- Solution: LLM-assisted strategy builder
- modules
    - Chatbot takes description of a strategy
    - LLM generates code
    - backtester
    - LLM generates report including metrics
    - web app
- TODO
    - design, visualization
    - how to treat coding errors
    - Maybe the LLM should ask questions
    - knowledge base for LLM, especially terms

## LLM portfolio advisor (UPDATE)
Bjarami, Verneva, et al
- People construct portfolios without rules
- modules
    - web based, AI assisted PF management tool
      1. take a list of stocks from user
      2. do mean-variance optimization with constraints
      3. Chart of historical trend -> verbal comment by LLM 
      4. Interactive dashboard
    - Python
- TODO
    - LLM input?
    - User interface - not everything with LLM
    - should user select the tickers?
    - customization of summary ("I am interested in ...", style, ...)
    - alterntive optimization methods
    - clear user journey and 

## Hybrid trading bot
Filograsso, Gallina, La Rocca, Semerar
- Trading based on sentiment and technical analysis
- modules
    - data from Yfinance + sentiment data from X, Reddit etc
    - sentiment analysis via LLM
    - technical indicators
    - hybrid signal (logical conditions)
    - backtester, stress testing, 
    - Python package
- TODO
    - how to formulate the strategy?
    - handling of API keys
    - make a nice documentation on GitHub

## Trading bot blue chips
De Cani, Fossati, Linossi
- modules
    - python package for trading, check, backtest*
    - Reorting
    - LLM verbal summary of trade
- TODO
    - how to formulate the strategy?
    - handling of API keys

## Financial news summarizer and forecaster
Celeste, Ferchichi, Jones
- analyze daily news 
- create news index, correlate sentiment with S&P500
- outcome: show correlation between news and market, understand impact of news sentiment of market
- modules
    - data: financial news + prices
    - LLM for sentiment analysis
    - web dashboard
    - predictive analytics with ML model 

## AI analyst
Poma, Schiatti, Marzoli, Zoccolillo
- analyze earnings reports via RAG
- modules
    - data: earnings reports
    - LLM for sentiment analysis
    - web dashboard
    - predictive analytics with ML model
    - RAG pipeline
- TODO
    - allow for hybrid queries: "year =xx" (SQL) and "content is about yyy" (vector search)

## Sustainable finance literacy
Fumi et al
- households have low level of sustainable finance literacy and cannot make informed decisions about durable goods
- minimize lifetime cost
- modules
    - life time cost calculator
    
- TODO
    - allow for hybrid queries: "year =xx" (SQL) and "content is about yyy" (vector search)
