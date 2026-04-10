# stratoquant/__init__.py
"""
StratoQuant — Quantitative Finance Library
===========================================
Usage:
    import stratoquant as sq

    sq.black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')
    sq.delta(100, 100, 1.0, 0.05, 0.2, 'call')
    sq.greeks_surface(100, K_grid, T_grid, 0.05, 0.2, 'call')
    sq.fit_garch(returns, model='gjr', dist='studentst')
    sq.VolSurface(K, T, IV, spot=100, r=0.05)
    sq.Backtest(ohlcv, sq.MACrossStrategy, params={'fast': 20, 'slow': 50})
"""

# ── Pricing ───────────────────────────────────────────────────────────────────────
from .pricing import (
    black_scholes_price,
    bs_price_surface,
    binomial_tree_pricing,
    monte_carlo_pricing,
    heston_price,
    heston_price_surface,
)

# ── Greeks ────────────────────────────────────────────────────────────────────────
from .bs_greeks import (
    delta,
    gamma,
    vega,
    theta,
    rho,
    volga,
    charm,
    vanna,
    speed,
    zomma,
    all_greeks,
    greeks_surface,
)

# ── Volatility ────────────────────────────────────────────────────────────────────
from .volatility import (
    implied_volatility,
    historical_volatility,
    realized_volatility,
)

# ── Vol surface ───────────────────────────────────────────────────────────────────
from .vol_surface import (
    VolSurface,
    svi_raw,
    fit_svi_slice,
    check_calendar_arbitrage,
    check_butterfly_arbitrage,
)

# ── Calibration ───────────────────────────────────────────────────────────────────
from .calibration import (
    calibrate_heston,
    fit_garch,
    realized_vol_estimators,
)

# ── Statistics ────────────────────────────────────────────────────────────────────
from .stats import (
    cointegration_test,
    adf_test,
    kpss_test,
    pp_test,
    granger_causality_test,
    jarque_bera_test,
    shapiro_wilk_test,
)

# ── Portfolio ─────────────────────────────────────────────────────────────────────
from .portfolio import portfolio_analysis

# ── Backtesting ───────────────────────────────────────────────────────────────────
from .backtesting import (
    Backtest,
    MACrossStrategy,
    RSIMeanReversion,
    BollingerBreakout,
    MomentumStrategy,
)

# ── Data ──────────────────────────────────────────────────────────────────────────
from .data import (
    fetch_prices,
    fetch_returns,
    fetch_option_chain,
    fetch_option_expiries,
    fetch_risk_free_rate,
    fetch_ticker_info,
    simulate_prices,
)

# ── Technical analysis ────────────────────────────────────────────────────────────
from .tech_analysis import (
    SMA,
    RSI,
    MACD,
    BBands,
    ATR,
    KAMA,
    ADX,
    Parabolic_SAR,
    stoch_oscillator,
    CCI,
    VWAP,
    VWAP_intraday,
    Ichimoku,
)

# ── Plots ─────────────────────────────────────────────────────────────────────────
from .plots import (
    plot_price,
    plot_returns,
    plot_greeks_profile,
    plot_greeks_surface,
    plot_vol_smile,
    plot_payoff,
    plot_bs_surface,
    plot_portfolio,
)

# ── Package metadata ──────────────────────────────────────────────────────────────
__version__ = "1.0.0"
__author__  = "Yassine Housseine"
__all__ = [
    # Pricing
    "black_scholes_price", "bs_price_surface",
    "binomial_tree_pricing", "monte_carlo_pricing",
    "heston_price", "heston_price_surface",
    # Greeks
    "delta", "gamma", "vega", "theta", "rho",
    "volga", "charm", "vanna", "speed", "zomma",
    "all_greeks", "greeks_surface",
    # Volatility
    "implied_volatility", "historical_volatility", "realized_volatility",
    # Vol surface
    "VolSurface", "svi_raw", "fit_svi_slice",
    "check_calendar_arbitrage", "check_butterfly_arbitrage",
    # Calibration
    "calibrate_heston", "fit_garch", "realized_vol_estimators",
    # Stats
    "cointegration_test", "adf_test", "kpss_test", "pp_test",
    "granger_causality_test", "jarque_bera_test", "shapiro_wilk_test",
    # Portfolio
    "portfolio_analysis",
    # Backtesting
    "Backtest",
    "MACrossStrategy", "RSIMeanReversion",
    "BollingerBreakout", "MomentumStrategy",
    # Data
    "fetch_prices", "fetch_returns", "fetch_option_chain",
    "fetch_option_expiries", "fetch_risk_free_rate",
    "fetch_ticker_info", "simulate_prices",
    # Technical analysis
    "SMA", "RSI", "MACD", "BBands", "ATR", "KAMA",
    "ADX", "Parabolic_SAR", "stoch_oscillator", "CCI",
    "VWAP", "VWAP_intraday", "Ichimoku",
    # Plots
    "plot_price", "plot_returns", "plot_greeks_profile",
    "plot_greeks_surface", "plot_vol_smile", "plot_payoff",
    "plot_bs_surface", "plot_portfolio",
]