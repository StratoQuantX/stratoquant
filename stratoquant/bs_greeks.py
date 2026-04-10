from scipy.stats import norm
import numpy as np
import pandas as pd


# ── Internal utilities ───────────────────────────────────────────────────────────

def _to_array(*args):
    """Broadcast all inputs to compatible numpy arrays."""
    return np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in args])


def _scalar_out(result, *original_inputs):
    """Return float if all original inputs were scalar, else return array."""
    if all(np.ndim(x) == 0 for x in original_inputs):
        return float(np.squeeze(result))
    return result


def _d1d2(S, K, T, r, sigma):
    """
    Compute d1 and d2 — vectorized.
    All inputs may be arrays of compatible shapes.
    To extend to continuous dividends q: replace r with (r - q) in d1.
    """
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def _check_option_type(option_type):
    if option_type not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'.")


# ── Convention note ─────────────────────────────────────────────────────────────
#
# Vega, Rho, Volga, Vanna, Zomma are ÷100 (per 1% move convention).
#
# All functions are fully vectorized:
#   scalar in  → scalar out
#   array in   → array out  (numpy broadcasting)
#   pd.Series  → np.ndarray (index not preserved — wrap manually if needed)
#
# Example — entire strike grid at once:
#   K_grid = np.linspace(80, 120, 41)
#   d = delta(100, K_grid, 1.0, 0.05, 0.2, 'call')   # shape (41,)
#
# ────────────────────────────────────────────────────────────────────────────────


# ── First-order Greeks ──────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, option_type=None):
    """
    Delta — ∂V/∂S. Range [0,1] calls, [-1,0] puts.
    At T=0: binary (1/-1 if ITM, 0 otherwise).
    """
    _check_option_type(option_type)
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, _ = _d1d2(S, K, T, r, sigma)

    if option_type == 'call':
        result = np.where(T <= 0, np.where(S > K, 1.0, 0.0), norm.cdf(d1))
    else:
        result = np.where(T <= 0, np.where(S < K, -1.0, 0.0), norm.cdf(d1) - 1.0)

    return _scalar_out(result, S0, K0, T0, r0, s0)


def gamma(S, K, T, r, sigma):
    """
    Gamma — ∂²V/∂S². Identical for calls and puts. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, _ = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    active = norm.pdf(d1) / (S * sigma * np.sqrt(T_safe))
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def vega(S, K, T, r, sigma):
    """
    Vega — ∂V/∂σ / 100 (per 1% vol move). Identical for calls and puts. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, _ = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    active = S * norm.pdf(d1) * np.sqrt(T_safe) / 100
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def theta(S, K, T, r, sigma, option_type=None):
    """
    Theta — ∂V/∂t / 365 (per calendar day). Typically negative. At T=0: 0.
    """
    _check_option_type(option_type)
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    decay = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T_safe))

    if option_type == 'call':
        active = (decay - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        active = (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def rho(S, K, T, r, sigma, option_type=None):
    """
    Rho — ∂V/∂r / 100 (per 1% rate move). At T=0: 0.
    """
    _check_option_type(option_type)
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    _, d2 = _d1d2(S, K, T, r, sigma)

    if option_type == 'call':
        active = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        active = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


# ── Second-order Greeks ─────────────────────────────────────────────────────────

def volga(S, K, T, r, sigma):
    """
    Volga (Vomma) — ∂²V/∂σ². Convention: ÷100. Identical for calls and puts. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    vega_raw = S * norm.pdf(d1) * np.sqrt(T_safe)
    active = vega_raw * d1 * d2 / sigma / 100
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def charm(S, K, T, r, sigma, option_type=None):
    """
    Charm (Delta decay) — ∂²V/∂S∂t. Overnight delta drift. At T=0: 0.
    """
    _check_option_type(option_type)
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    sqrt_T = np.sqrt(T_safe)
    common = -norm.pdf(d1) * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T_safe * sigma * sqrt_T)
    active = common if option_type == 'call' else -common
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def vanna(S, K, T, r, sigma):
    """
    Vanna — ∂²V/∂S∂σ. Delta sensitivity to vol. Convention: ÷100. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    active = -norm.pdf(d1) * d2 / sigma / 100
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


# ── Third-order Greeks ──────────────────────────────────────────────────────────

def speed(S, K, T, r, sigma):
    """
    Speed — ∂³V/∂S³. Rate of gamma change w.r.t. spot. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, _ = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    gam = norm.pdf(d1) / (S * sigma * np.sqrt(T_safe))
    active = -gam / S * (d1 / (sigma * np.sqrt(T_safe)) + 1)
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


def zomma(S, K, T, r, sigma):
    """
    Zomma — ∂³V/∂S²∂σ. Gamma sensitivity to vol. Convention: ÷100. At T=0: 0.
    """
    S0, K0, T0, r0, s0 = S, K, T, r, sigma
    S, K, T, r, sigma = _to_array(S, K, T, r, sigma)

    d1, d2 = _d1d2(S, K, T, r, sigma)
    T_safe = np.where(T > 0, T, np.nan)
    gam = norm.pdf(d1) / (S * sigma * np.sqrt(T_safe))
    active = gam * (d1 * d2 - 1) / sigma / 100
    result = np.where(T <= 0, 0.0, active)
    return _scalar_out(result, S0, K0, T0, r0, s0)


# ── Convenience functions ───────────────────────────────────────────────────────

def all_greeks(S, K, T, r, sigma, option_type=None):
    """
    All Greeks in one call — returns a dict of arrays.

    Example:
        g = all_greeks(100, np.linspace(80,120,41), 1.0, 0.05, 0.2, 'call')
        g['delta']  # shape (41,)
    """
    _check_option_type(option_type)
    return {
        'delta' : delta(S, K, T, r, sigma, option_type),
        'gamma' : gamma(S, K, T, r, sigma),
        'vega'  : vega(S, K, T, r, sigma),
        'theta' : theta(S, K, T, r, sigma, option_type),
        'rho'   : rho(S, K, T, r, sigma, option_type),
        'volga' : volga(S, K, T, r, sigma),
        'charm' : charm(S, K, T, r, sigma, option_type),
        'vanna' : vanna(S, K, T, r, sigma),
        'speed' : speed(S, K, T, r, sigma),
        'zomma' : zomma(S, K, T, r, sigma),
    }


def greeks_surface(S, K_grid, T_grid, r, sigma, option_type=None):
    """
    Full Greeks surface over a (K × T) grid — returns a MultiIndex DataFrame.

    Parameters:
    -----------
    S           : float      — spot price
    K_grid      : array-like — 1D array of strikes
    T_grid      : array-like — 1D array of maturities (years)
    r           : float      — risk-free rate
    sigma       : float      — implied vol
    option_type : str        — 'call' or 'put'

    Returns:
    --------
    pd.DataFrame  MultiIndex (K, T), columns = greek names

    Example:
        df = greeks_surface(100, np.linspace(80,120,9), [0.25,0.5,1.0,2.0], 0.05, 0.2, 'call')
        df.loc[(100, 1.0), 'delta']   # ATM 1Y delta
        df['vega'].unstack('T')        # vega surface as matrix
    """
    _check_option_type(option_type)
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)

    K_mesh, T_mesh = np.meshgrid(K_grid, T_grid)   # shapes (nT, nK)
    g = all_greeks(S, K_mesh, T_mesh, r, sigma, option_type)

    index = pd.MultiIndex.from_product([K_grid, T_grid], names=['K', 'T'])
    return pd.DataFrame(
        {name: vals.T.ravel() for name, vals in g.items()},
        index=index
    ).sort_index()