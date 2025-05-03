### TODO (Ale)

- Finish check on corr models

- Check returns vs log returns on everything (and % change returns)

- After those, check the shorting for the functions that receive x positions as inputs... (and shorting on others?)

- For portfolio metrics, is AN VaR the base or is portfolio normal? Should be the first, but double check...

- For displaying the tables with PortfolioVaR add head and tail only, it may be too long otherwise (years of data...)

- Check correlations plots and shorting cases

- EVT needs to have wealth input? so worth separating it (into ES and VaR)? Likely not the best, both ES and VaR need the same params...

- Add wealth scaling to basic VaR

- **kwargs hides parameters from users (fix that)

- Remove callers, add wealth scaling directly into each function for VaR and ES (as extra column)