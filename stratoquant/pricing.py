from scipy.stats import norm
import numpy as np
import pandas as pd
from numpy import log, exp, sqrt, real
from scipy.integrate import quad


def _check_option_type(option_type):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")


def _to_array(*args):
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


def _scalar_out(result, *original_inputs):
    if all(np.ndim(x) == 0 for x in original_inputs):
        return float(np.squeeze(result))
    return result


# ── Black-Scholes ───────────────────────────────────────────────────────────────
def black_scholes_price(S, K, T, r, sigma, option_type=None):
    """
    European option price under Black-Scholes — fully vectorized.

    All inputs broadcast: pass arrays to price a whole strike grid at once.

    Parameters:
    -----------
    S, K, T, r, sigma : scalar or array-like (broadcast-compatible)
    option_type       : 'call' or 'put'

    Returns:
    --------
    float or np.ndarray — option price(s)

    Examples:
    ---------
    # Single price
    black_scholes_price(100, 100, 1.0, 0.05, 0.2, 'call')

    # Full strike grid (41 prices, one call)
    K_grid = np.linspace(80, 120, 41)
    black_scholes_price(100, K_grid, 1.0, 0.05, 0.2, 'call')

    # Full surface: 9 strikes × 4 maturities = 36 prices
    K_mesh, T_mesh = np.meshgrid(np.linspace(80,120,9), [0.25,0.5,1.0,2.0])
    black_scholes_price(100, K_mesh, T_mesh, 0.05, 0.2, 'call')
    """
    _check_option_type(option_type)
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    # Intrinsic value at expiry
    if option_type == 'call':
        intrinsic = np.maximum(S - K, 0.0)
    else:
        intrinsic = np.maximum(K - S, 0.0)

    # BS formula (only where T > 0)
    T_safe = np.where(T > 0, T, np.nan)
    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type == 'call':
        bs = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        bs = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    result = np.where(T <= 0, intrinsic, bs)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def bs_price_surface(S, K_grid, T_grid, r, sigma, option_type=None):
    """
    Black-Scholes price surface over a (K × T) grid.

    Parameters:
    -----------
    S           : float      — spot price
    K_grid      : array-like — 1D array of strikes
    T_grid      : array-like — 1D array of maturities
    r           : float      — risk-free rate
    sigma       : float      — implied volatility
    option_type : str        — 'call' or 'put'

    Returns:
    --------
    pd.DataFrame — index = K_grid, columns = T_grid

    Example:
        df = bs_price_surface(100, np.linspace(80,120,9), [0.25,0.5,1.0,2.0], 0.05, 0.2, 'call')
        df.loc[100, 1.0]   # ATM 1Y call price
    """
    _check_option_type(option_type)
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)   # (nT, nK)
    prices = black_scholes_price(S, K_mesh, T_mesh, r, sigma, option_type)  # (nT, nK)

    return pd.DataFrame(prices.T, index=K_grid, columns=T_grid)


# ── Binomial tree (CRR) ─────────────────────────────────────────────────────────

def binomial_tree_pricing(S, K, T, r, sigma, n, option_type=None):
    """
    European option price via Cox-Ross-Rubinstein binomial tree.

    Scalar only — the backward induction is inherently sequential.
    For vectorized pricing across strikes, use black_scholes_price.

    Parameters:
    -----------
    S           : float — initial stock price
    K           : float — strike price
    T           : float — time to maturity (years)
    r           : float — risk-free rate
    sigma       : float — volatility
    n           : int   — number of time steps (n=500 → ~4 decimal places vs BS)
    option_type : str   — 'call' or 'put'

    Returns:
    --------
    float : option price

    Note: for American options, add early exercise check inside backward induction.
    """
    _check_option_type(option_type)

    dt       = T / n
    u        = np.exp(sigma * np.sqrt(dt))
    d        = 1 / u
    p        = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    i  = np.arange(n + 1)
    ST = S * (u ** i) * (d ** (n - i))

    option = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)

    for _ in range(n):
        option = discount * (p * option[1:] + (1 - p) * option[:-1])

    return float(option[0])


# ── Monte Carlo ─────────────────────────────────────────────────────────────────

def monte_carlo_pricing(S, K, T, r, sigma, n_simulations, option_type=None,
                        antithetic=True, return_ci=False, seed=None):
    """
    European option price via Monte Carlo (GBM terminal price) — vectorized draws.

    Parameters:
    -----------
    S             : float — spot price
    K             : float — strike price
    T             : float — time to maturity (years)
    r             : float — risk-free rate
    sigma         : float — volatility
    n_simulations : int   — number of paths
    option_type   : str   — 'call' or 'put'
    antithetic    : bool  — antithetic variates for variance reduction (default: True)
                            Draws n/2 Gaussians + mirrors → ~35% variance reduction
    return_ci     : bool  — if True, return (price, ci_lower_95, ci_upper_95)
    seed          : int   — random seed for reproducibility

    Returns:
    --------
    float | (float, float, float)
    """
    _check_option_type(option_type)

    rng       = np.random.default_rng(seed)
    discount  = np.exp(-r * T)
    drift     = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    if antithetic:
        half  = n_simulations // 2
        Z     = rng.standard_normal(half)
        Z_all = np.concatenate([Z, -Z])
    else:
        Z_all = rng.standard_normal(n_simulations)

    ST = S * np.exp(drift + diffusion * Z_all)

    payoffs = np.maximum(ST - K, 0) if option_type == 'call' else np.maximum(K - ST, 0)
    price   = discount * np.mean(payoffs)

    if return_ci:
        std_err  = np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))
        ci_lower = discount * (np.mean(payoffs) - 1.96 * std_err)
        ci_upper = discount * (np.mean(payoffs) + 1.96 * std_err)
        return price, ci_lower, ci_upper

    return price


# ── Heston ──────────────────────────────────────────────────────────────────────

def heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, option_type=None):
    """
    European option price under Heston stochastic vol (Heston 1993 characteristic fn).

    Scalar only — integration is inherently scalar.
    For a grid of strikes/maturities, loop over this function or use
    heston_price_surface() below.

    Parameters:
    -----------
    S, K, T, r  : float — spot, strike, maturity, risk-free rate
    v0          : float — initial variance (e.g. 0.04 for 20% vol)
    kappa       : float — mean reversion speed
    theta       : float — long-term variance
    sigma       : float — vol of vol
    rho         : float — spot/variance correlation (typically negative)
    option_type : str   — 'call' or 'put'

    Returns:
    --------
    float : option price

    Notes:
    ------
    Feller condition: 2*kappa*theta > sigma² ensures variance stays positive.
    Put price derived from call via put-call parity.
    """
    _check_option_type(option_type)

    if 2 * kappa * theta <= sigma**2:
        import warnings
        warnings.warn(
            f"Feller condition violated: 2κθ={2*kappa*theta:.4f} ≤ σ²={sigma**2:.4f}. "
            "Variance may hit zero; accuracy may degrade.",
            RuntimeWarning
        )

    def integrand(phi, j):
        u = 0.5 if j == 1 else -0.5
        b = (kappa - rho * sigma) if j == 1 else kappa
        a = kappa * theta
        x = log(S)

        d     = np.sqrt((rho * sigma * 1j * phi - b)**2 - sigma**2 * (2 * u * 1j * phi - phi**2))
        g     = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        denom = 1 - g * np.exp(d * T)
        if np.abs(denom) < 1e-12:
            return 0.0

        C = (r * 1j * phi * T +
             a / sigma**2 * ((b - rho * sigma * 1j * phi + d) * T -
             2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2) * (
            (1 - np.exp(d * T)) / (1 - g * np.exp(d * T))
        )
        return real(np.exp(C + D * v0 + 1j * phi * x - 1j * phi * log(K)) / (1j * phi))

    try:
        P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1), 1e-6, 200, limit=100)[0]
        P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2), 1e-6, 200, limit=100)[0]
    except Exception as e:
        raise RuntimeError(
            f"Heston integration failed (S={S}, K={K}, T={T}, v0={v0}, "
            f"κ={kappa}, θ={theta}, σ={sigma}, ρ={rho}). Error: {e}"
        )

    call_price = S * P1 - K * exp(-r * T) * P2
    return call_price if option_type == 'call' else call_price - S + K * exp(-r * T)


def heston_price_surface(S, K_grid, T_grid, r, v0, kappa, theta, sigma, rho, option_type=None):
    """
    Heston price surface over a (K × T) grid.

    Loops heston_price() — slower than BS surface but correct.

    Parameters:
    -----------
    S           : float      — spot price
    K_grid      : array-like — 1D array of strikes
    T_grid      : array-like — 1D array of maturities
    r, v0, kappa, theta, sigma, rho : Heston parameters
    option_type : str        — 'call' or 'put'

    Returns:
    --------
    pd.DataFrame — index = K_grid, columns = T_grid
    """
    _check_option_type(option_type)
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    prices = np.array([
        [heston_price(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
         for K in K_grid]
        for T in T_grid
    ])  # shape (nT, nK)

    return pd.DataFrame(prices.T, index=K_grid, columns=T_grid)