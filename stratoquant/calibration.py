"""
stratoquant.calibration
=======================
Model calibration for StratoQuant.

Three calibrators:

1. calibrate_heston(market_prices, S, K_grid, T_grid, r)
   Fits Heston (v0, kappa, theta, sigma, rho) to a grid of market option
   prices by minimizing vega-weighted RMSE via differential evolution +
   L-BFGS-B local polish.

2. fit_garch(returns, model, dist)
   Fits a GARCH-family model (GARCH, EGARCH, GJR-GARCH) to a return series
   via maximum likelihood. Returns parameters, conditional volatility, and
   full diagnostics.

3. realized_vol_estimators(ohlcv)
   Parkinson, Garman-Klass, and Yang-Zhang realized vol estimators from
   OHLCV data — more efficient than close-to-close.

Dependencies: scipy, numpy, pandas, arch
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy.optimize import differential_evolution, minimize
from scipy.integrate import quad
from numpy import log, exp, sqrt, real


# ── Internal helpers ──────────────────────────────────────────────────────────────

def _check_option_type(t):
    if t not in ('call', 'put'):
        raise ValueError(f"option_type must be 'call' or 'put', got '{t}'.")


def _bs_price_scalar(S, K, T, r, sigma, option_type):
    """Scalar BS pricer used internally during calibration loops."""
    from scipy.stats import norm
    if T <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _bs_vega_scalar(S, K, T, r, sigma):
    """Scalar BS vega (raw, not /100) for weighting."""
    from scipy.stats import norm
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def _heston_price_scalar(S, K, T, r, v0, kappa, theta, sigma, rho, option_type):
    """Scalar Heston pricer via characteristic function (Heston 1993)."""
    def integrand(phi, j):
        u = 0.5 if j == 1 else -0.5
        b = (kappa - rho * sigma) if j == 1 else kappa
        a = kappa * theta
        x = log(max(S, 1e-10))

        d = np.sqrt((rho * sigma * 1j * phi - b)**2
                    - sigma**2 * (2 * u * 1j * phi - phi**2))
        g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)
        denom = 1 - g * np.exp(d * T)
        if np.abs(denom) < 1e-12:
            return 0.0

        C = (r * 1j * phi * T
             + a / sigma**2 * ((b - rho * sigma * 1j * phi + d) * T
             - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g))))
        D = ((b - rho * sigma * 1j * phi + d) / sigma**2
             * (1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return real(np.exp(C + D * v0 + 1j * phi * x - 1j * phi * log(K))
                    / (1j * phi))

    try:
        P1 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 1),
                                        1e-6, 200, limit=100)[0]
        P2 = 0.5 + (1 / np.pi) * quad(lambda phi: integrand(phi, 2),
                                        1e-6, 200, limit=100)[0]
    except Exception:
        # Fall back to BS if integration fails during optimization
        iv = np.sqrt(max(v0, 1e-6))
        return _bs_price_scalar(S, K, T, r, iv, option_type)

    call = S * P1 - K * exp(-r * T) * P2
    return call if option_type == 'call' else call - S + K * exp(-r * T)


# ── 1. Heston calibration ─────────────────────────────────────────────────────────

def calibrate_heston(
    market_prices: np.ndarray,
    S:             float,
    K_grid:        np.ndarray,
    T_grid:        np.ndarray,
    r:             float,
    option_type:   str   = 'call',
    weights:       Optional[np.ndarray] = None,
    use_vega_weights: bool = True,
    popsize:       int   = 12,
    maxiter:       int   = 300,
    seed:          Optional[int] = 42,
    verbose:       bool  = True,
) -> dict:
    """
    Calibrate Heston model parameters to market option prices.

    Minimizes vega-weighted RMSE between Heston model prices and market prices
    over a (K × T) grid, using differential evolution for global search
    followed by L-BFGS-B for local polish.

    Parameters:
    -----------
    market_prices    : np.ndarray  — observed market prices, shape (n_K, n_T) or flat (n,)
                                     Must align with K_grid × T_grid meshgrid.
    S                : float       — current spot price
    K_grid           : np.ndarray  — 1D array of strikes
    T_grid           : np.ndarray  — 1D array of maturities (years)
    r                : float       — risk-free rate
    option_type      : str         — 'call' or 'put' (default: 'call')
    weights          : np.ndarray  — custom weights (overrides use_vega_weights)
    use_vega_weights : bool        — weight each option by its BS vega (default: True)
                                     Vega-weighting focuses calibration on liquid
                                     near-ATM options rather than far wings.
    popsize          : int         — differential evolution population size (default: 12)
    maxiter          : int         — max iterations for differential evolution (default: 300)
    seed             : int         — random seed for differential evolution (default: 42)
    verbose          : bool        — print calibration progress (default: True)

    Returns:
    --------
    dict with:
        params       : dict  — {'v0', 'kappa', 'theta', 'sigma', 'rho'}
        rmse         : float — root mean squared error of fitted prices
        rmse_rel     : float — relative RMSE (RMSE / mean market price)
        model_prices : np.ndarray — Heston model prices at calibrated params
        residuals    : np.ndarray — (market_price - model_price) per option
        feller_ok    : bool  — whether Feller condition 2κθ > σ² is satisfied
        success      : bool  — optimizer convergence flag

    Notes:
    ------
    - Calibration is slow (~30–120 seconds depending on grid size and maxiter).
      For a quick check, reduce popsize=6, maxiter=50.
    - Parameter bounds:
        v0    : (1e-4, 1.0)  — initial variance
        kappa : (0.1, 20.0)  — mean reversion speed
        theta : (1e-4, 1.0)  — long-term variance
        sigma : (1e-4, 2.0)  — vol of vol
        rho   : (-0.99, 0.0) — correlation (typically negative for equities)
    - The Feller condition 2κθ > σ² ensures the variance process stays positive.
      The optimizer does not enforce it — check feller_ok in the output.

    Example:
    --------
    # Simulate "market prices" from known Heston params, then recover them
    K_grid = np.array([90, 95, 100, 105, 110], dtype=float)
    T_grid = np.array([0.25, 0.5, 1.0], dtype=float)
    true_params = dict(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
    mkt = np.array([[_heston_price_scalar(100, K, T, 0.05, **true_params, option_type='call')
                     for T in T_grid] for K in K_grid])
    result = calibrate_heston(mkt, S=100, K_grid=K_grid, T_grid=T_grid, r=0.05)
    print(result['params'])
    """
    _check_option_type(option_type)
    K_grid = np.asarray(K_grid, dtype=float)
    T_grid = np.asarray(T_grid, dtype=float)
    market_prices = np.asarray(market_prices, dtype=float).reshape(len(K_grid), len(T_grid))

    # Build flat lists of (K, T, market_price) triplets
    K_flat = np.repeat(K_grid, len(T_grid))
    T_flat = np.tile(T_grid, len(K_grid))
    P_flat = market_prices.ravel()

    # Vega weights
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
    elif use_vega_weights:
        iv_approx = 0.20  # rough vol for initial weighting
        w = np.array([_bs_vega_scalar(S, K, T, r, iv_approx)
                      for K, T in zip(K_flat, T_flat)])
        w = w / w.sum() * len(w)  # normalize so weights average to 1
    else:
        w = np.ones(len(P_flat))

    n_options = len(P_flat)

    # Bounds: v0, kappa, theta, sigma, rho
    bounds = [
        (1e-4, 1.0),    # v0
        (0.1,  20.0),   # kappa
        (1e-4, 1.0),    # theta
        (1e-4, 2.0),    # sigma (vol of vol)
        (-0.99, 0.0),   # rho (negative for equities)
    ]

    call_count = [0]

    def objective(params):
        v0, kappa, theta, sigma, rho = params
        call_count[0] += 1

        model_prices = np.array([
            _heston_price_scalar(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
            for K, T in zip(K_flat, T_flat)
        ])
        diff = P_flat - model_prices
        return np.sqrt(np.mean(w * diff**2))

    if verbose:
        print(f"Calibrating Heston on {n_options} options "
              f"({len(K_grid)} strikes × {len(T_grid)} maturities)...")
        print(f"Method: differential evolution (popsize={popsize}, maxiter={maxiter})"
              f" + L-BFGS-B polish")

    # Global search
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        de_result = differential_evolution(
            objective, bounds,
            seed=seed, popsize=popsize, maxiter=maxiter,
            tol=1e-6, polish=False,
            workers=1,
        )

    # Local polish
    local_result = minimize(
        objective, de_result.x,
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 500}
    )

    best_params = local_result.x if local_result.fun < de_result.fun else de_result.x
    v0, kappa, theta, sigma, rho = best_params

    # Final model prices and diagnostics
    model_prices = np.array([
        _heston_price_scalar(S, K, T, r, v0, kappa, theta, sigma, rho, option_type)
        for K, T in zip(K_flat, T_flat)
    ])
    residuals = P_flat - model_prices
    rmse      = np.sqrt(np.mean(residuals**2))
    rmse_rel  = rmse / np.mean(P_flat) if np.mean(P_flat) > 0 else np.nan
    feller_ok = 2 * kappa * theta > sigma**2

    params_dict = {
        'v0':    round(float(v0),    6),
        'kappa': round(float(kappa), 6),
        'theta': round(float(theta), 6),
        'sigma': round(float(sigma), 6),
        'rho':   round(float(rho),   6),
    }

    if verbose:
        print(f"\nCalibration complete ({call_count[0]} evaluations)")
        print(f"  v0    = {v0:.6f}  (initial variance, impl. vol ≈ {np.sqrt(v0):.2%})")
        print(f"  kappa = {kappa:.6f}  (mean reversion speed)")
        print(f"  theta = {theta:.6f}  (long-term variance, impl. vol ≈ {np.sqrt(theta):.2%})")
        print(f"  sigma = {sigma:.6f}  (vol of vol)")
        print(f"  rho   = {rho:.6f}  (spot/vol correlation)")
        print(f"  RMSE        = {rmse:.6f}")
        print(f"  Rel. RMSE   = {rmse_rel:.2%}")
        print(f"  Feller cond.: {'✅ satisfied (2κθ > σ²)' if feller_ok else '❌ violated (2κθ ≤ σ²)'}")

    return {
        'params':       params_dict,
        'rmse':         rmse,
        'rmse_rel':     rmse_rel,
        'model_prices': model_prices.reshape(len(K_grid), len(T_grid)),
        'residuals':    residuals.reshape(len(K_grid), len(T_grid)),
        'feller_ok':    feller_ok,
        'success':      local_result.success or de_result.success,
    }


# ── 2. GARCH calibration ──────────────────────────────────────────────────────────

def fit_garch(
    returns:    Union[pd.Series, np.ndarray],
    model:      str = 'garch',
    p:          int = 1,
    q:          int = 1,
    dist:       str = 'normal',
    mean:       str = 'constant',
    verbose:    bool = True,
) -> dict:
    """
    Fit a GARCH-family model to a return series via maximum likelihood.

    Supported models:
        'garch'    — standard GARCH(p,q)
        'egarch'   — EGARCH(p,q) — asymmetric, captures leverage effect
        'gjr'      — GJR-GARCH(p,q) — asymmetric, captures negative shocks

    Supported error distributions:
        'normal'   — Gaussian innovations
        'studentst'— Student-t (fat tails)
        'skewt'    — Skewed Student-t (fat tails + asymmetry)

    Parameters:
    -----------
    returns : pd.Series or np.ndarray — return series (log returns preferred)
              Should be in percentage points (e.g. 1.5 for +1.5%) OR
              decimals (e.g. 0.015). The function auto-scales if std < 0.01.
    model   : str  — 'garch', 'egarch', or 'gjr' (default: 'garch')
    p       : int  — ARCH lag order (default: 1)
    q       : int  — GARCH lag order (default: 1)
    dist    : str  — error distribution (default: 'normal')
    mean    : str  — mean model: 'constant', 'zero', 'ar' (default: 'constant')
    verbose : bool — print fit summary (default: True)

    Returns:
    --------
    dict with:
        params          : dict   — fitted parameters (omega, alpha, beta, etc.)
        cond_vol        : pd.Series — conditional volatility time series (annualized)
        cond_vol_raw    : pd.Series — conditional volatility (model units)
        residuals       : pd.Series — standardized residuals
        log_likelihood  : float  — log-likelihood at optimum
        aic             : float  — Akaike Information Criterion
        bic             : float  — Bayesian Information Criterion
        persistence     : float  — alpha + beta (GARCH) — proximity to 1 = long memory
        half_life       : float  — vol half-life in periods
        model_summary   : str    — arch model summary string
        diagnostics     : dict   — Ljung-Box test on residuals²

    Example:
    --------
    from stratoquant.data import fetch_returns
    ret = fetch_returns('SPY', period='5y')['SPY']
    result = fit_garch(ret, model='gjr', dist='studentst')
    result['cond_vol'].plot(title='GJR-GARCH Conditional Vol')
    """
    try:
        from arch import arch_model
        from arch.__future__ import reindexing
    except ImportError:
        raise ImportError(
            "arch is required for GARCH fitting. "
            "Install with:  pip install arch"
        )

    if isinstance(returns, np.ndarray):
        returns = pd.Series(returns.ravel(), name='returns')

    returns = returns.dropna().copy()

    # Auto-scale: arch expects returns in % (e.g. 1.0 = 1%), not decimals
    if returns.std() < 0.01:
        returns = returns * 100
        scaled = True
    else:
        scaled = False

    model_map = {
        'garch': 'GARCH',
        'egarch': 'EGARCH',
        'gjr': 'GJR-GARCH',
    }
    if model not in model_map:
        raise ValueError(f"model must be one of {list(model_map.keys())}, got '{model}'.")

    vol_model = {'garch': 'GARCH', 'egarch': 'EGARCH', 'gjr': 'GARCH'}[model]
    power     = 2.0
    o         = 1 if model == 'gjr' else 0

    am = arch_model(
        returns,
        mean=mean,
        vol=vol_model,
        p=p, o=o, q=q,
        power=power,
        dist=dist,
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = am.fit(disp='off', show_warning=False)

    # Conditional vol — annualize
    cond_vol_raw = fit.conditional_volatility  # in model units (% or decimal)
    scale        = 100 if scaled else 1
    ann_factor   = np.sqrt(252)
    cond_vol_ann = cond_vol_raw / scale * ann_factor

    # Params
    params = dict(fit.params)

    # Persistence & half-life (GARCH and GJR only)
    alpha_keys = [k for k in params if k.startswith('alpha')]
    beta_keys  = [k for k in params if k.startswith('beta')]
    gamma_keys = [k for k in params if k.startswith('gamma')]

    alpha = sum(params[k] for k in alpha_keys)
    beta  = sum(params[k] for k in beta_keys)
    gamma = sum(params.get(k, 0) for k in gamma_keys)

    persistence = alpha + beta + 0.5 * gamma  # GJR adjustment
    half_life   = np.log(0.5) / np.log(persistence) if 0 < persistence < 1 else np.inf

    # Standardized residuals
    std_resid = pd.Series(fit.std_resid, index=returns.index, name='std_resid')

    # Ljung-Box on squared residuals (test for remaining ARCH effects)
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(std_resid**2, lags=[5, 10, 20], return_df=True)
        lb_diag = {
            f'lb_stat_lag{lag}':    float(lb.loc[lag, 'lb_stat'])
            for lag in [5, 10, 20]
        } | {
            f'lb_pvalue_lag{lag}':  float(lb.loc[lag, 'lb_pvalue'])
            for lag in [5, 10, 20]
        }
    except Exception:
        lb_diag = {}

    if verbose:
        print(f"\n{'='*50}")
        print(f"  {model_map[model]}({p},{q}) — dist={dist} — mean={mean}")
        print(f"{'='*50}")
        print(f"  Log-likelihood : {fit.loglikelihood:.4f}")
        print(f"  AIC            : {fit.aic:.4f}")
        print(f"  BIC            : {fit.bic:.4f}")
        print(f"  Persistence    : {persistence:.6f}")
        print(f"  Half-life      : {half_life:.1f} periods")
        print(f"\n  Parameters:")
        for k, v in params.items():
            print(f"    {k:20s}: {v:.6f}")
        if lb_diag:
            print(f"\n  Ljung-Box (squared residuals):")
            for lag in [5, 10, 20]:
                pv = lb_diag.get(f'lb_pvalue_lag{lag}', np.nan)
                flag = '✅' if pv > 0.05 else '⚠️ '
                print(f"    lag {lag:>2}: p-value = {pv:.4f}  {flag}")
        print(f"{'='*50}\n")

    return {
        'params':         params,
        'cond_vol':       cond_vol_ann,
        'cond_vol_raw':   cond_vol_raw,
        'residuals':      std_resid,
        'log_likelihood': fit.loglikelihood,
        'aic':            fit.aic,
        'bic':            fit.bic,
        'persistence':    persistence,
        'half_life':      half_life,
        'model_summary':  str(fit.summary()),
        'diagnostics':    lb_diag,
        'fit_object':     fit,   # full arch fit object for advanced use
    }


# ── 3. Realized volatility estimators ────────────────────────────────────────────

def realized_vol_estimators(
    ohlcv:    pd.DataFrame,
    window:   int  = 21,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Compute realized volatility estimators from OHLCV data.

    Three estimators, each more efficient than close-to-close:

    Close-to-close (baseline):
        σ² = (1/n) Σ (log(C_t / C_{t-1}))²
        Efficiency ratio: 1x (reference)

    Parkinson (1980):
        σ² = (1 / 4n·ln2) Σ (log(H_t / L_t))²
        Uses intraday high-low range. Efficiency ratio: ~5x vs c-to-c.
        Assumes no drift and no overnight jumps.

    Garman-Klass (1980):
        σ² = (1/n) Σ [0.5(log(H/L))² − (2ln2−1)(log(C/O))²]
        Uses OHLC. Efficiency ratio: ~8x vs c-to-c.
        Assumes no drift.

    Yang-Zhang (2000):
        σ² = σ_overnight² + k·σ_open² + (1−k)·σ_close²
        Uses overnight gaps + open-to-close. Most robust.
        Efficiency ratio: ~14x vs c-to-c. Handles drift and jumps.

    Parameters:
    -----------
    ohlcv     : pd.DataFrame — OHLCV with columns [open, high, low, close]
    window    : int          — rolling window in periods (default: 21)
    annualize : bool         — annualize using sqrt(252) (default: True)

    Returns:
    --------
    pd.DataFrame with columns:
        close_to_close  — classic close-to-close estimator
        parkinson       — Parkinson estimator
        garman_klass    — Garman-Klass estimator
        yang_zhang      — Yang-Zhang estimator

    Example:
    --------
    from stratoquant.data import fetch_prices
    ohlcv = fetch_prices('SPY', period='2y')
    vol = realized_vol_estimators(ohlcv, window=21)
    vol.plot(title='Realized Vol Estimators — SPY')
    """
    required = ['open', 'high', 'low', 'close']
    missing  = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    O = np.log(ohlcv['open'])
    H = np.log(ohlcv['high'])
    L = np.log(ohlcv['low'])
    C = np.log(ohlcv['close'])
    C_prev = C.shift(1)

    ann = np.sqrt(252) if annualize else 1.0

    # ── Close-to-close ──
    cc = (C - C_prev)
    cc_vol = cc.rolling(window).std() * ann

    # ── Parkinson ──
    hl    = (H - L) ** 2
    pk_var = hl.rolling(window).mean() / (4 * np.log(2))
    pk_vol = np.sqrt(pk_var) * ann

    # ── Garman-Klass ──
    gk_term = 0.5 * hl - (2 * np.log(2) - 1) * (C - O) ** 2
    gk_var  = gk_term.rolling(window).mean()
    gk_vol  = np.sqrt(gk_var.clip(lower=1e-12)) * ann

    # ── Yang-Zhang ──
    # Overnight return (close-to-open)
    overnight = O - C_prev
    # Open-to-close return
    oc        = C - O
    k         = 0.34 / (1.34 + (window + 1) / (window - 1))

    sigma_ov  = overnight.rolling(window).var()
    sigma_oc  = oc.rolling(window).var()

    # Rogers-Satchell variance (drift-free open-to-close)
    rs_term   = (H - C) * (H - O) + (L - C) * (L - O)
    sigma_rs  = rs_term.rolling(window).mean()

    yz_var    = sigma_ov + k * sigma_oc + (1 - k) * sigma_rs
    yz_vol    = np.sqrt(yz_var.clip(lower=1e-12)) * ann

    result = pd.DataFrame({
        'close_to_close': cc_vol,
        'parkinson':      pk_vol,
        'garman_klass':   gk_vol,
        'yang_zhang':     yz_vol,
    }, index=ohlcv.index)

    return result.dropna(how='all')
