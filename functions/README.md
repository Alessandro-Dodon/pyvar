### TODO (Ale)

- Once finished, make tutorial

- Check if some matrix algebra operation fails with the portfolio VaR/ ES or dynamic corr models

- Check crazy short positions on the functions that use x monetary, in case add a failsafe/ warning

- Take away multiday MC?

- Uniform across all models and py files the params names (z, etc)

- Add NaN checker/ blocker in each function

- Add MC and simulations graphs to the plots?

- consider data check for na for downloading data function

- Check log returns vs normal returns on everything

- Use term "filtered historical innovations"/ check it out (shoould be the same as standardized residuals)

- Dynamic corr models could work with standardized innovations also, but volatility must be normalized by portfolio value. This is equivalent to using weights in the volatility calculations.

- Explain buy and hold strategy (always assumed in our calculations) in the pdf report

- Check w formula in factor models, it is delicate with extreme positions (also it is delicate in the dynamic corr models if put there)

- Check positive semidefinite matrix when crafting portfolio in data download functions

- Specify in docstrings that options can be empty list (equity only portfolios) and longer VaR horizon can be done by cumulative T day period returns for HS

- Update data download function to work also with weekly? Does it work with non param VaR?

- In basic VaR, scaling by sqrt(h) is incompatible with the df and violation logic (check and add a warning maybe)... Maybe remove scaling by sqrt(h)? It can be done trivially and doesn't complicate the logic.

- Yes, factor models use normal distribution

- Add a note on unstable VCV matrix when too few columns compared to rows. This is not our case, as data downloading from YF makes it rare that someone downloads portfolios of hundreds of positions (maybe add a note or warning from data_download)

- Check pd series vs pd df for univariate VaR models and data download functions (maybe add in the data download function if only a ticker is selected it returns the pd series with date index so runs automatically with univariate models)