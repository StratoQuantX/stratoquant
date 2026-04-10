import numpy as np
import pandas as pd

def PortfolioAnalysis(returns, weights=None, risk_free_rate=0.01):
    """
    Perform portfolio performance and risk analysis.

    Parameters:
        returns (pd.DataFrame): DataFrame of asset returns.
        weights (np.array or list): Portfolio weights (default: equal weights).
        risk_free_rate (float): Risk-free rate per period (e.g., daily or annualized).

    Returns:
        dict: Dictionary with returns, volatility, Sharpe ratio, cumulative returns, and drawdown.
    """
    if weights is None:
        weights = np.ones(returns.shape[1]) / returns.shape[1]
    else:
        weights = np.asarray(weights)

    # Check weight dimensions
    assert returns.shape[1] == weights.shape[0], "Weight vector length must match number of assets"

    # Portfolio returns time series
    portfolio_returns = returns.dot(weights)

    # Cumulative returns
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # Drawdown
    cumulative_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - cumulative_max) / cumulative_max

    # Portfolio stats
    avg_return = portfolio_returns.mean()
    volatility = portfolio_returns.std()
    sharpe_ratio = (avg_return - risk_free_rate) / volatility

    return {
        'returns': portfolio_returns,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'cumulative_returns': cumulative_returns - 1,  # to match your format
        'drawdown': drawdown
    }

# End of file portfolio.py
