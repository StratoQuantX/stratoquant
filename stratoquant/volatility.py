import numpy as np
from scipy.optimize import brentq
from .pricing import black_scholes_price
from .bs_greeks import vega


def implied_volatility(market_price, S, K, T, r, option_type='call', verbose=False):
    """
    Calculate the implied volatility using the Black-Scholes model (Brent's method).

    Parameters:
    -----------
    market_price : float — observed market price of the option
    S            : float — spot price of the underlying
    K            : float — strike price
    T            : float — time to maturity (in years)
    r            : float — risk-free interest rate
    option_type  : str   — 'call' or 'put'
    verbose      : bool  — if True, prints the result

    Returns:
    --------
    float : implied volatility (annualized)

    Raises:
    -------
    ValueError : if market_price violates no-arbitrage bounds
    RuntimeError : if Brent's method fails to converge
    """
    # No-arbitrage bounds check
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0)
    upper_bound = S if option_type == 'call' else K * np.exp(-r * T)

    if market_price < intrinsic:
        raise ValueError(
            f"market_price={market_price:.4f} is below intrinsic value={intrinsic:.4f}. "
            "No implied vol exists (arbitrage violation)."
        )
    if market_price >= upper_bound:
        raise ValueError(
            f"market_price={market_price:.4f} is above the upper BS bound={upper_bound:.4f}. "
            "No implied vol exists."
        )

    def error_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        implied_vol = brentq(error_function, 1e-6, 5.0, xtol=1e-8, maxiter=500)
    except ValueError as e:
        raise RuntimeError(
            f"Brent's method failed to converge for {option_type} "
            f"(S={S}, K={K}, T={T}, market_price={market_price}). "
            f"Original error: {e}"
        )

    if verbose:
        print(f"Implied Volatility ({option_type}): {implied_vol:.6f} ({implied_vol*100:.2f}%)")

    return implied_vol


def historical_volatility(prices, window=252, verbose=False):
    """
    Calculate the historical volatility of a price series over a rolling window,
    annualized on a 252 trading-day basis.

    Parameters:
    -----------
    prices  : pd.Series — time series of asset prices
    window  : int       — number of trading days to look back (default: 252)
    verbose : bool      — if True, prints the result

    Returns:
    --------
    float : annualized historical volatility computed on the last `window` observations

    Note:
    -----
    The window parameter controls BOTH the lookback period AND is used for annualization
    only if window != 252. Annualization always uses sqrt(252) to stay comparable
    across different window choices.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < window:
        raise ValueError(
            f"Not enough observations: got {len(log_returns)}, need at least {window}."
        )

    windowed_returns = log_returns.iloc[-window:]
    hist_vol = windowed_returns.std(ddof=1) * np.sqrt(252)

    if verbose:
        name = prices.name if hasattr(prices, 'name') and prices.name else 'series'
        print(f"Historical Volatility ({name}, window={window}d): {hist_vol:.6f} ({hist_vol*100:.2f}%)")

    return hist_vol


def realized_volatility(prices, window=252, verbose=False):
    """
    Calculate the realized volatility of a price series over a specified window,
    annualized on a 252 trading-day basis.

    Parameters:
    -----------
    prices  : pd.Series — time series of asset prices
    window  : int       — number of trading days for the realized vol window (default: 252)
    verbose : bool      — if True, prints the result

    Returns:
    --------
    float : annualized realized volatility over the last `window` observations

    Note:
    -----
    Difference from historical_volatility: realized vol conventionally uses
    zero-mean returns (ddof=0 or sum of squared returns), reflecting the
    actual path rather than a sample estimate. Here we use the standard
    close-to-close estimator for consistency with the rest of the library.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < window:
        raise ValueError(
            f"Not enough observations: got {len(log_returns)}, need at least {window}."
        )

    windowed_returns = log_returns.iloc[-window:]
    real_vol = windowed_returns.std(ddof=1) * np.sqrt(252)

    if verbose:
        name = prices.name if hasattr(prices, 'name') and prices.name else 'series'
        print(f"Realized Volatility ({name}, window={window}d): {real_vol:.6f} ({real_vol*100:.2f}%)")

    return real_vol
