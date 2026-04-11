import numpy as np
import pandas as pd


def portfolio_analysis(returns, weights=None, risk_free_rate=0.01, freq='daily'):
    """
    Portfolio performance and risk analysis.

    Parameters:
    -----------
    returns         : pd.DataFrame — asset return time series (rows = dates, cols = assets)
    weights         : array-like   — portfolio weights (default: equal weights).
                                     Must sum to 1 for standard interpretation.
    risk_free_rate  : float        — annualized risk-free rate (default: 0.01 = 1%).
                                     Always pass an annualized rate; the function
                                     converts it to per-period before computing Sharpe.
    freq            : str          — return frequency for annualization.
                                     One of: 'daily' (252), 'weekly' (52),
                                     'monthly' (12), 'annual' (1).

    Returns:
    --------
    dict with:
        returns            : pd.Series  — portfolio return time series
        volatility         : float      — annualized portfolio volatility
        sharpe_ratio       : float      — annualized Sharpe ratio
        cumulative_returns : pd.Series  — cumulative return (0-based, e.g. 0.15 = +15%)
        cumulative_index   : pd.Series  — cumulative wealth index (starts at 1.0)
        drawdown           : pd.Series  — drawdown series (always <= 0)
        max_drawdown       : float      — maximum drawdown (worst peak-to-trough)

    Notes:
    ------
    - Sharpe ratio is annualized: (mean_period_return - rf_period) / std_period * sqrt(periods_per_year)
    - risk_free_rate must be annualized regardless of return frequency.
    """
    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12, 'annual': 1}
    if freq not in freq_map:
        raise ValueError(f"freq must be one of {list(freq_map.keys())}, got '{freq}'.")

    periods_per_year = freq_map[freq]

    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]
    else:
        weights = np.asarray(weights, dtype=float)

    if returns.shape[1] != weights.shape[0]:
        raise ValueError(
            f"Weight vector length ({weights.shape[0]}) must match "
            f"number of assets ({returns.shape[1]})."
        )

    # Portfolio return series
    portfolio_returns = returns.dot(weights)

    # Annualized stats
    mean_period   = portfolio_returns.mean()
    std_period    = portfolio_returns.std()
    rf_per_period = risk_free_rate / periods_per_year

    volatility = std_period * np.sqrt(periods_per_year)

    if std_period == 0:
        sharpe_ratio = np.nan
    else:
        sharpe_ratio = (mean_period - rf_per_period) / std_period * np.sqrt(periods_per_year)

    # Cumulative returns
    cumulative_index   = (1 + portfolio_returns).cumprod()
    cumulative_returns = cumulative_index - 1

    # Drawdown
    cumulative_max = cumulative_index.cummax()
    drawdown       = (cumulative_index - cumulative_max) / cumulative_max
    max_drawdown   = drawdown.min()

    return {
        'returns':            portfolio_returns,
        'volatility':         volatility,
        'sharpe_ratio':       sharpe_ratio,
        'cumulative_returns': cumulative_returns,
        'cumulative_index':   cumulative_index,
        'drawdown':           drawdown,
        'max_drawdown':       max_drawdown,
    }
