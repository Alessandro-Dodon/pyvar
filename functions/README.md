### TODO (Ale)

- Once finished, make tutorial

- Check if some matrix algebra operation fails with the portfolio VaR/ ES or dynamic corr models

- Check crazy short positions on the functions that use x monetary, in case add a failsafe/ warning

- Take away multiday MC?

- Uniform across all models and py files the params names (z, etc)

- Add NaN checker/ blocker in each function

- Add MC and simulations graphs to the plots?

- consider data check for na for downloading data function

- Check conversion/ different results on different days

- Check log returns vs normal returns on everything

- Use term "filtered historical innovations"/ check it out (shoould be the same as standardized residuals)

- Dynamic corr models could work with standardized innovations also, but volatility must be normalized by portfolio value. This is equivalent to using weights in the volatility calculations.

- Explain buy and hold strategy (always assumed in our calculations) in the pdf report

- Check w formula in factor models, it is delicate with extreme positions (also it is delicate in the dynamic corr models if put there)

- Check positive semidefinite matrix when crafting portfolio in data download functions