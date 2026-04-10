# stratoquant/__init__.py

from .pricing import black_scholes_price, binomial_tree_pricing, monte_carlo_pricing, heston_price
from .bs_greeks import delta, gamma, vega, theta, rho, volga, charm
from .hedging import simulate_delta_hedge, compute_hedging_error, hedging_summary, simulate_greeks_over_time
from .volatility import implied_volatility, historical_volatility, realized_volatility
from .stats import cointegration_test, adf_test, pp_test, kpss_test, granger_causality_test, jarque_bera_test, shapiro_wilk_test
from .tech_analysis import *


__all__ = [
    'black_scholes_price', 'binomial_tree_pricing', 'monte_carlo_pricing', 'heston_price',
    'delta', 'gamma', 'vega', 'theta', 'rho', 'volga', 'charm',
    'simulate_delta_hedge', 'compute_hedging_error', 'hedging_summary',
    'implied_volatility', 'historical_volatility', 'realized_volatility',
    'cointegration_test', 'adf_test', 'pp_test', 'kpss_test', 'granger_causality_test', 'jarque_bera_test', 'shapiro_wilk_test',
    'SMA', 'RSI', 'MACD', 'BBands', 'ATR', 'KAMA', 'ADX', 'Parabolic_SAR', 'stoch_oscillator', 'CCI', 'VWAP', 'VWAP_intraday', 'Ichimoku'
    ]

# Version of the package
__version__ = "1.0.0"

# Author of the package
__author__ = "Yassine Housseine"

# End of file __init__.py