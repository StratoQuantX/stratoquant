import numpy as np
from scipy.optimize import brentq
from .pricing import black_scholes_price
from .bs_greeks import vega


def implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculate the implied volatility using the Black-Scholes model.

    """

    # Define the error function to minimize
    def error_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    # Use Brent's method to find the root of the error function    
    implied_vol = brentq(error_function, 1e-6, 5.0)  # Volatility bounds: [0.000001, 5.0]

    print(f"Implied Volatility of {option_type} option: {implied_vol}")  # Debug print statement


def historical_volatility(prices, window=252):
    """
    Calculate the historical volatility of a series of prices.

    """

    returns = np.log(prices / prices.shift(1))  # Log returns
    result = np.std(returns, ddof=1) * np.sqrt(window)  # Historical volatility over the window
    result_2 = np.std(returns, ddof=1) * np.sqrt(252)  # Annualized historical volatility over the window
    print(f"Historical Volatility of {prices.name}: {result}")  # Debug print statement
    print(f"Annualized Historical Volatility of {prices.name}: {result_2}")  # Debug print statement


def realized_volatility(prices, window=252):
    """
    Calculate the realized volatility of a series of prices over a specified window.
    """

    returns = np.log(prices / prices.shift(1))  # Log returns
    result = np.std(returns[-window:], ddof=1) * np.sqrt(window)  # Annualized volatility over the window
    
    print(f"Realized Volatility of {prices.name}: {result}")  # Debug print statement
