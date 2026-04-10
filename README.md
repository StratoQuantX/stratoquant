# StratoQuant

**A quantitative finance library for options pricing, Greeks, volatility modeling, backtesting, and market data analysis.**

Built by [Yassine Housseine](https://github.com/Yaskoi) as part of StratoQuant, a student quantitative research association.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-2.0.0-green)](setup.py)

---

## Overview

StratoQuant covers the full quantitative workflow — from fetching market data to pricing options, computing Greeks, calibrating stochastic vol models, running backtests, and building implied volatility surfaces.

```python
import stratoquant as sq
import numpy as np

# Price a European call
sq.black_scholes_price(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type='call')
# → 10.4506

# Full Greek surface over a strike grid
K_grid = np.linspace(80, 120, 41)
sq.delta(100, K_grid, 1.0, 0.05, 0.2, 'call')   # → array of shape (41,)

# Backtest a strategy
bt = sq.Backtest(ohlcv, sq.MACrossStrategy, params={'fast': 20, 'slow': 50})
bt.run()
bt.summary()
```

---

## Installation

```bash
pip install stratoquant
```

Or from source:

```bash
git clone https://github.com/StratoQuantX/stratoquant.git
cd stratoquant
pip install -e .
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`, `statsmodels`, `arch`, `yfinance`, `seaborn`

Optional extras:

```bash
pip install stratoquant[dev]       # pytest, black, mypy
pip install stratoquant[notebook]  # jupyterlab, plotly
```

---

## Modules

| Module | Description |
|---|---|
| `pricing` | Black-Scholes, Binomial CRR, Monte Carlo, Heston |
| `bs_greeks` | 10 Greeks (Delta → Zomma), vectorized over arrays |
| `volatility` | Implied vol, historical vol, realized vol |
| `vol_surface` | IV surface construction, SVI fit, arbitrage checks |
| `calibration` | Heston calibration, GARCH/EGARCH/GJR fitting, OHLCV vol estimators |
| `stats` | Stationarity tests (ADF, KPSS, PP), Granger causality, normality tests |
| `portfolio` | Portfolio returns, Sharpe, Sortino, drawdown |
| `backtesting` | Event-driven backtesting framework + 4 built-in strategies |
| `data` | Yahoo Finance data fetching, option chains, synthetic GBM data |
| `plots` | 8 visualization functions (surfaces, smiles, payoffs, dashboards) |
| `tech_analysis` | 13 technical indicators (RSI, MACD, Bollinger, Ichimoku, ...) |

---

## Usage

### Pricing

```python
import stratoquant as sq

# Black-Scholes — scalar
sq.black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')   # 10.4506

# Black-Scholes — vectorized over strikes
import numpy as np
K_grid = np.linspace(80, 120, 41)
prices = sq.black_scholes_price(100, K_grid, 1.0, 0.05, 0.2, 'call')  # shape (41,)

# Price surface (K × T grid)
T_grid = np.array([0.25, 0.5, 1.0, 2.0])
surface = sq.bs_price_surface(100, K_grid, T_grid, 0.05, 0.2, 'call')
# → pd.DataFrame, index=K_grid, columns=T_grid

# Binomial tree (CRR)
sq.binomial_tree_pricing(100, 100, 1.0, 0.05, 0.2, n=500, option_type='call')

# Monte Carlo with antithetic variates + 95% CI
price, ci_low, ci_high = sq.monte_carlo_pricing(
    100, 100, 1.0, 0.05, 0.2, n_simulations=50_000,
    option_type='call', antithetic=True, return_ci=True, seed=42
)

# Heston stochastic volatility
sq.heston_price(S=100, K=100, T=1.0, r=0.05,
                v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
                option_type='call')
```

### Greeks

All Greeks accept scalars, numpy arrays, or pd.Series and broadcast natively.

```python
# Scalar
sq.delta(100, 100, 1.0, 0.05, 0.2, 'call')   # 0.6368

# Vectorized over strikes
sq.gamma(100, K_grid, 1.0, 0.05, 0.2)         # shape (41,)

# All 10 Greeks at once
g = sq.all_greeks(100, K_grid, 1.0, 0.05, 0.2, 'call')
# g['delta'], g['vanna'], g['zomma'], ...   all shape (41,)

# Full Greek surface (K × T) as MultiIndex DataFrame
df = sq.greeks_surface(100, K_grid, T_grid, 0.05, 0.2, 'call')
df.loc[(100.0, 1.0), 'delta']       # ATM 1Y delta
df['vega'].unstack('T')              # vega surface as matrix
```

Available Greeks: `delta`, `gamma`, `vega`, `theta`, `rho`, `volga`, `charm`, `vanna`, `speed`, `zomma`.

### Volatility

```python
# Implied volatility from market price (Brent's method)
sq.implied_volatility(market_price=10.5, S=100, K=100, T=1.0, r=0.05, option_type='call')

# Historical vol (rolling window, annualized)
sq.historical_volatility(prices, window=252)

# Realized vol (same, with explicit window)
sq.realized_volatility(prices, window=21)
```

### Implied Volatility Surface

```python
import numpy as np

K = np.linspace(85, 115, 7)
T = np.array([0.25, 0.5, 1.0, 2.0])
IV = ...  # your (n_K, n_T) IV matrix

vs = sq.VolSurface(K, T, IV, spot=100, r=0.05, method='spline')

# Interpolate at any (K, T)
vs.get_iv(K=103, T=0.75)

# Full smile for a given maturity
smile = vs.get_smile(T=1.0)

# Build from option chain
chain = sq.fetch_option_chain('SPY', expiry='2025-12-19', option_type='call')
vs = sq.VolSurface.from_chain(chain, spot=chain.attrs['spot'], r=0.05)

# Arbitrage checks (calendar + butterfly + density)
vs.check_arbitrage()

# Plot
vs.plot(kind='surface')
vs.plot(kind='smile', T_list=[0.5, 1.0, 2.0])
```

### Calibration

```python
# Heston calibration from market prices
result = sq.calibrate_heston(
    market_prices,          # (n_K, n_T) array
    S=100, K_grid=K_grid, T_grid=T_grid, r=0.05,
    use_vega_weights=True,  # focus on near-ATM options
    popsize=12, maxiter=300,
)
print(result['params'])      # {'v0': ..., 'kappa': ..., 'theta': ..., 'sigma': ..., 'rho': ...}
print(result['rmse'])
print(result['feller_ok'])   # 2κθ > σ² check

# GARCH family
result = sq.fit_garch(returns, model='gjr', dist='studentst', verbose=True)
result['cond_vol'].plot()    # annualized conditional volatility
result['persistence']        # α + β (+ 0.5γ for GJR)
result['half_life']          # vol half-life in trading days

# OHLCV realized vol estimators
vol = sq.realized_vol_estimators(ohlcv, window=21)
# columns: close_to_close, parkinson, garman_klass, yang_zhang
```

### Statistics

```python
# Stationarity
sq.adf_test(returns)                    # Augmented Dickey-Fuller
sq.kpss_test(returns)                   # KPSS (opposite H0 to ADF)
sq.pp_test(returns)                     # Phillips-Perron

# Cointegration
sq.cointegration_test(series1, series2)

# Granger causality (returns all lags)
result = sq.granger_causality_test(data[['y', 'x']], max_lags=10)
result['all_pvalues']   # {lag: p-value} for all tested lags

# Normality
sq.jarque_bera_test(returns, plot=True)
sq.shapiro_wilk_test(returns)
```

### Portfolio Analysis

```python
returns = sq.fetch_returns(['AAPL', 'MSFT', 'GOOG'], start='2022-01-01')

result = sq.portfolio_analysis(
    returns,
    weights=[0.4, 0.35, 0.25],
    risk_free_rate=0.05,
    freq='daily'
)
result['sharpe_ratio']
result['max_drawdown']
result['cumulative_index'].plot()
```

### Backtesting

```python
ohlcv = sq.fetch_prices('SPY', period='5y')

# Built-in strategies
bt = sq.Backtest(ohlcv, sq.MACrossStrategy,    params={'fast': 20, 'slow': 50})
bt = sq.Backtest(ohlcv, sq.RSIMeanReversion,   params={'oversold': 30, 'overbought': 70})
bt = sq.Backtest(ohlcv, sq.BollingerBreakout,  params={'mode': 'reversion', 'k': 2.0})
bt = sq.Backtest(ohlcv, sq.MomentumStrategy,   params={'lookback': 126, 'skip': 5})

# Run and inspect
results = bt.run()
bt.summary()     # 14 metrics: Sharpe, Sortino, Calmar, VaR, CVaR, ...
fig = bt.plot()  # equity curve, drawdown, positions, daily returns

# Walk-forward validation
oos = bt.walk_forward(train_periods=252, test_periods=63)

# Custom strategy — just a function with this signature:
def my_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
    # return pd.Series of signals: 1=long, -1=short, 0=flat
    ...

bt = sq.Backtest(ohlcv, my_strategy, params={...},
                 cost_bps=10, slippage_bps=5, allow_short=True)
```

### Data

```python
# OHLCV — single ticker
ohlcv = sq.fetch_prices('BNP.PA', start='2020-01-01')
ohlcv = sq.fetch_prices('AAPL', period='2y')

# Close prices — multiple tickers
prices = sq.fetch_prices(['AAPL', 'MSFT', 'GOOG'], start='2022-01-01')

# Log or simple returns
returns = sq.fetch_returns(['SAN.PA', 'BNP.PA'], start='2021-01-01', method='log')

# Option chain
chain = sq.fetch_option_chain('SPY', expiry='2025-12-19',
                               option_type='call', moneyness_range=(0.9, 1.1))
expiries = sq.fetch_option_expiries('SPY')

# Live risk-free rate (US T-bills/T-notes via Yahoo Finance)
r = sq.fetch_risk_free_rate('3m')   # 13-week T-bill
r = sq.fetch_risk_free_rate('10y')  # 10-year T-note

# Synthetic GBM data (offline testing)
ohlcv  = sq.simulate_prices(S0=100, sigma=0.20, n_days=252, seed=42)
prices = sq.simulate_prices(n_assets=4, ticker_names=['A','B','C','D'], seed=0)
```

### Visualization

```python
# OHLCV candlestick + volume + moving averages
sq.plot_price(ohlcv, ticker='SPY', ma_periods=[20, 50])

# Return distribution (histogram + KDE + normal + QQ plot)
sq.plot_returns(returns['SPY'], label='SPY')

# Greek profile vs spot (call + put)
sq.plot_greeks_profile(K=100, T=1.0, r=0.05, sigma=0.2, greek='vanna')

# Greek 3D surface
sq.plot_greeks_surface(100, K_grid, T_grid, 0.05, 0.2, greek='volga')

# Implied vol smile with optional SVI fit
sq.plot_vol_smile(strikes, ivs, spot=100, expiry='2025-12-19', fit_svi=True)

# Payoff diagram — single or multi-leg
sq.plot_payoff([
    {'K': 100, 'type': 'call', 'position': 'long',  'premium': 5.0, 'label': 'Long Call'},
    {'K': 100, 'type': 'put',  'position': 'long',  'premium': 4.0, 'label': 'Long Put'},
])

# BS price surface 3D
sq.plot_bs_surface(100, K_grid, T_grid, 0.05, 0.2, 'call')

# Portfolio dashboard (equity curve + drawdown + rolling Sharpe)
sq.plot_portfolio(returns, label='My Portfolio')
```

### Technical Analysis

```python
close  = ohlcv['close']

sq.SMA(close, period=20)
sq.RSI(close, period=14)
macd, signal, hist = sq.MACD(close)
upper, mid, lower  = sq.BBands(close, period=20, k=2)
sq.ATR(ohlcv, period=14)
sq.KAMA(close, n=14)
adx, di_up, di_dn  = sq.ADX(ohlcv, period=14)
sq.Parabolic_SAR(ohlcv)
K_stoch, D_stoch   = sq.stoch_oscillator(ohlcv, period=14)
sq.CCI(ohlcv, n=20)
sq.VWAP(ohlcv)
sq.VWAP_intraday(ohlcv)
tenkan, kijun, span_a, span_b, chikou = sq.Ichimoku(ohlcv)
```

---

## Project Structure

```
stratoquant/
├── LICENSE
├── README.md
├── setup.py
└── stratoquant/
    ├── __init__.py
    ├── pricing.py        # BS, Binomial, MC, Heston
    ├── bs_greeks.py      # 10 Greeks, vectorized
    ├── volatility.py     # Implied vol, historical vol
    ├── vol_surface.py    # IV surface, SVI, arbitrage checks
    ├── calibration.py    # Heston calibration, GARCH, OHLCV estimators
    ├── stats.py          # ADF, KPSS, PP, Granger, JB, SW
    ├── portfolio.py      # Portfolio performance analysis
    ├── backtesting.py    # Backtesting engine + strategies
    ├── data.py           # Market data (Yahoo Finance) + synthetic
    ├── plots.py          # Visualization layer
    └── tech_analysis.py  # Technical indicators
```

---

## License

Licensed under the [Apache License 2.0](LICENSE).

Copyright 2025 Yassine Housseine
