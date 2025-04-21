### TODO (Ale)

- Empirical z is correct for ARCH/ GARCH model, but maybe not for EWMA and MA, so assuming a distribution for those may be better. This could explain the very similar violations rate between models

- Maybe add possibility of using z not empirical but drawn from distributions with volatility models

- Maybe add different GARCH models (TGARCH etc)

- Maybe add GARCH based simulations (it is complex)

- In the (volatility) functions, add the possibility to use wealth (W) as input, so it is even more customizable

- Continue with portfolio methods/ EVT (when Plazzi explains it)

- Maybe add a function to download data directly from YF to use the portfolio VaR in an easy way

- Maybe meta functions (like for plotting etc)

- Modify plotting functions so that the user may select static versions and not interactive if desired (easier for pdf etc)

- Better styling when printing results for portfolio functions