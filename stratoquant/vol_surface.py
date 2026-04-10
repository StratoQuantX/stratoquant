"""
stratoquant.vol_surface
=======================
Implied volatility surface construction and analysis.

Two interpolation methods:
  - Cubic spline (per-maturity smile interpolation)
  - SVI parametric (Gatheral's stochastic vol inspired model)

Arbitrage checks:
  - Calendar spread arbitrage (total variance must be increasing in T)
  - Butterfly arbitrage    (density must be non-negative — g(k) >= 0)
  - Static replication bounds (IV must be >= 0)

Usage
-----
    from stratoquant.vol_surface import VolSurface

    # From option chain (fetch_option_chain output)
    chain = fetch_option_chain('SPY', expiry='2025-06-20')
    vs = VolSurface.from_chain(chain, spot=500, r=0.05)

    # From a (K, T) → IV grid directly
    vs = VolSurface(strikes, maturities, ivs, spot=500, r=0.05)

    iv = vs.get_iv(K=510, T=0.5)          # interpolated IV
    smile = vs.get_smile(T=0.5)           # full smile for one maturity
    fig = vs.plot()                        # surface plot
    arb = vs.check_arbitrage()            # arbitrage report

Dependencies: numpy, pandas, scipy, matplotlib
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize_scalar, minimize


# ── SVI parametric model ──────────────────────────────────────────────────────────

def svi_raw(k: np.ndarray, a: float, b: float, rho: float,
            m: float, sigma: float) -> np.ndarray:
    """
    Gatheral's raw SVI parametrization for total implied variance.

    w(k) = a + b * (rho*(k - m) + sqrt((k - m)^2 + sigma^2))

    where k = log(K/F) is log-moneyness (F = forward price).

    Parameters:
    -----------
    k     : np.ndarray — log-moneyness values
    a     : float      — overall variance level (vertical shift)
    b     : float      — angle between left and right asymptotes (>= 0)
    rho   : float      — rotation of the smile (-1 < rho < 1)
    m     : float      — horizontal translation (ATM shift)
    sigma : float      — curvature of the smile (> 0)

    Returns:
    --------
    np.ndarray — total implied variance w(k) = sigma_implied^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_var:     np.ndarray,
    weights:       Optional[np.ndarray] = None,
    n_restarts:    int = 5,
) -> dict:
    """
    Fit SVI parameters to a single maturity slice.

    Parameters:
    -----------
    log_moneyness : np.ndarray — k = log(K/F) for each strike
    total_var     : np.ndarray — market total variance = IV^2 * T
    weights       : np.ndarray — optional weights (e.g. vega weights)
    n_restarts    : int        — number of random restarts for robustness

    Returns:
    --------
    dict with: params (a,b,rho,m,sigma), rmse, success
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_var, dtype=float)
    if weights is None:
        wt = np.ones(len(k))
    else:
        wt = np.asarray(weights, dtype=float)

    # SVI arbitrage-free constraints (Gatheral & Jacquier 2014):
    # b >= 0, |rho| < 1, sigma > 0, a + b*sigma*sqrt(1-rho^2) >= 0
    def loss(params):
        a, b, rho, m, sigma = params
        if b < 0 or abs(rho) >= 1 or sigma <= 0:
            return 1e10
        w_model = svi_raw(k, a, b, rho, m, sigma)
        if np.any(w_model < 0):
            return 1e10
        return np.sum(wt * (w - w_model)**2)

    best_result = None
    best_loss   = np.inf

    # Multiple restarts with different initial guesses
    rng = np.random.default_rng(42)
    for i in range(n_restarts):
        if i == 0:
            x0 = [w.mean(), 0.1, -0.3, k.mean(), 0.2]
        else:
            x0 = [
                rng.uniform(0.001, w.max()),
                rng.uniform(0.01, 0.5),
                rng.uniform(-0.8, 0.0),
                rng.uniform(k.min(), k.max()),
                rng.uniform(0.05, 0.5),
            ]

        bounds = [
            (0.0,  w.max() * 2),  # a
            (0.0,  2.0),          # b
            (-0.999, 0.999),      # rho
            (k.min() - 0.5, k.max() + 0.5),  # m
            (1e-4, 1.0),          # sigma
        ]

        try:
            res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000})
            if res.fun < best_loss:
                best_loss   = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None:
        return {'params': None, 'rmse': np.inf, 'success': False}

    a, b, rho, m, sigma = best_result.x
    w_fit = svi_raw(k, a, b, rho, m, sigma)
    rmse  = np.sqrt(np.mean((w - w_fit)**2))

    return {
        'params':  {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma},
        'rmse':    rmse,
        'success': best_result.success,
    }


# ── Arbitrage checks ──────────────────────────────────────────────────────────────

def check_calendar_arbitrage(
    maturities: np.ndarray,
    total_var_matrix: np.ndarray,
) -> dict:
    """
    Check calendar spread arbitrage: total variance must be non-decreasing in T
    for each fixed log-moneyness.

    Parameters:
    -----------
    maturities        : np.ndarray — 1D array of maturities (years), sorted ascending
    total_var_matrix  : np.ndarray — shape (n_K, n_T), total variance w(k, T)

    Returns:
    --------
    dict with: violations (list of (k_idx, T_idx) pairs), is_clean (bool)
    """
    violations = []
    n_K, n_T = total_var_matrix.shape
    for i in range(n_K):
        for j in range(n_T - 1):
            if total_var_matrix[i, j] > total_var_matrix[i, j + 1] + 1e-8:
                violations.append({
                    'K_idx': i,
                    'T_from': maturities[j],
                    'T_to':   maturities[j + 1],
                    'excess': total_var_matrix[i, j] - total_var_matrix[i, j + 1],
                })
    return {'violations': violations, 'is_clean': len(violations) == 0}


def check_butterfly_arbitrage(
    log_moneyness: np.ndarray,
    total_var:     np.ndarray,
    n_points:      int = 200,
) -> dict:
    """
    Check butterfly arbitrage for a single smile slice.

    The risk-neutral density g(k) must be non-negative everywhere:
        g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2*(1/4 + 1/w) + w''/2

    where w'  = dw/dk, w'' = d²w/dk².

    Parameters:
    -----------
    log_moneyness : np.ndarray — k values (sorted)
    total_var     : np.ndarray — total variance w(k) for this maturity
    n_points      : int        — interpolation resolution for derivative check

    Returns:
    --------
    dict with: min_density, violations (array), is_clean (bool)
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_var, dtype=float)

    sort_idx = np.argsort(k)
    k, w = k[sort_idx], w[sort_idx]

    if len(k) < 4:
        return {'min_density': np.nan, 'violations': [], 'is_clean': True}

    # Cubic spline interpolation for smooth derivatives
    cs   = CubicSpline(k, w)
    k_fine = np.linspace(k[0], k[-1], n_points)
    w_f  = cs(k_fine)
    w1   = cs(k_fine, 1)   # first derivative
    w2   = cs(k_fine, 2)   # second derivative

    # Avoid division by zero
    w_safe = np.where(w_f > 1e-10, w_f, 1e-10)
    g = ((1 - k_fine * w1 / (2 * w_safe))**2
         - (w1**2 / 4) * (1 / w_safe + 0.25)
         + w2 / 2)

    min_g      = float(g.min())
    violations = k_fine[g < -1e-6].tolist()

    return {
        'min_density': min_g,
        'violations':  violations,
        'is_clean':    len(violations) == 0,
    }


# ── VolSurface class ──────────────────────────────────────────────────────────────

class VolSurface:
    """
    Implied volatility surface with interpolation and arbitrage checks.

    Supports two interpolation backends:
      - 'spline'  : cubic spline per maturity, bivariate spline across surface
      - 'svi'     : SVI parametric fit per maturity, interpolated across T

    Parameters:
    -----------
    strikes    : np.ndarray  — 1D array of unique strikes
    maturities : np.ndarray  — 1D array of unique maturities (years)
    ivs        : np.ndarray  — implied vol surface, shape (n_strikes, n_maturities)
                               in decimal form (e.g. 0.20 = 20%)
    spot       : float       — current spot price
    r          : float       — risk-free rate
    method     : str         — 'spline' or 'svi' (default: 'spline')
    forward    : np.ndarray  — optional forward prices per maturity
                               (default: S * exp(r * T))

    Example:
    --------
    # Build from a known IV surface
    K = np.linspace(90, 110, 5)
    T = np.array([0.25, 0.5, 1.0])
    IV = np.array([[0.25, 0.22, 0.20],
                   [0.22, 0.20, 0.19],
                   [0.20, 0.19, 0.18],
                   [0.22, 0.20, 0.19],
                   [0.25, 0.22, 0.20]])
    vs = VolSurface(K, T, IV, spot=100, r=0.05)
    print(vs.get_iv(K=100, T=0.5))
    """

    def __init__(
        self,
        strikes:    np.ndarray,
        maturities: np.ndarray,
        ivs:        np.ndarray,
        spot:       float,
        r:          float,
        method:     str = 'spline',
        forward:    Optional[np.ndarray] = None,
    ):
        self.strikes    = np.asarray(strikes,    dtype=float)
        self.maturities = np.asarray(maturities, dtype=float)
        self.ivs        = np.asarray(ivs,        dtype=float)  # (n_K, n_T)
        self.spot       = float(spot)
        self.r          = float(r)
        self.method     = method

        if forward is not None:
            self.forward = np.asarray(forward, dtype=float)
        else:
            self.forward = spot * np.exp(r * self.maturities)

        # Total variance surface w(k, T) = IV^2 * T
        self.log_moneyness = np.log(
            self.strikes[:, None] / self.forward[None, :]
        )  # shape (n_K, n_T)
        self.total_var = self.ivs**2 * self.maturities[None, :]  # (n_K, n_T)

        self._interpolator = None
        self._svi_fits     = {}
        self._build_interpolator()

    # ── Constructors ─────────────────────────────────────────────────────────────

    @classmethod
    def from_chain(
        cls,
        chain:       pd.DataFrame,
        spot:        float,
        r:           float,
        method:      str = 'spline',
        min_iv:      float = 0.001,
        min_volume:  int   = 0,
    ) -> 'VolSurface':
        """
        Build VolSurface from a fetch_option_chain() output DataFrame.

        Parameters:
        -----------
        chain      : pd.DataFrame — option chain with columns [strike, expiry, implied_volatility, ...]
        spot       : float        — current spot price
        r          : float        — risk-free rate
        method     : str          — 'spline' or 'svi'
        min_iv     : float        — minimum IV filter (default: 0.001)
        min_volume : int          — minimum volume filter (default: 0)

        Returns:
        --------
        VolSurface instance
        """
        df = chain.copy()

        # Filter
        df = df[df['implied_volatility'] > min_iv]
        if 'volume' in df.columns and min_volume > 0:
            df = df[df['volume'].fillna(0) >= min_volume]

        # Compute time to maturity
        if 'expiry' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['expiry']):
                today = pd.Timestamp.today().normalize()
                df['T'] = (df['expiry'] - today).dt.days / 365.0
            else:
                df['T'] = df['expiry'].astype(float)
        else:
            raise ValueError("chain must have an 'expiry' column.")

        df = df[df['T'] > 0]

        # Pivot to (K × T) IV matrix
        pivot = df.pivot_table(
            index='strike', columns='T',
            values='implied_volatility', aggfunc='mean'
        ).dropna(how='all')

        strikes    = pivot.index.values.astype(float)
        maturities = pivot.columns.values.astype(float)
        ivs        = pivot.values  # (n_K, n_T), NaNs where data missing

        # Fill NaNs via linear interpolation per row (per strike)
        for i in range(len(strikes)):
            row = ivs[i]
            if np.any(np.isfinite(row)):
                mask = np.isfinite(row)
                if mask.sum() >= 2:
                    ivs[i] = np.interp(
                        maturities, maturities[mask], row[mask]
                    )

        return cls(strikes, maturities, ivs, spot, r, method=method)

    # ── Build interpolator ────────────────────────────────────────────────────────

    def _build_interpolator(self):
        """Build the internal interpolator based on self.method."""
        K = self.strikes
        T = self.maturities
        w = self.total_var  # (n_K, n_T)

        if self.method == 'spline':
            if len(K) >= 3 and len(T) >= 3:
                try:
                    self._interpolator = RectBivariateSpline(K, T, w, kx=3, ky=3)
                except Exception:
                    # Fallback to linear if cubic fails
                    self._interpolator = RectBivariateSpline(K, T, w, kx=1, ky=1)
            elif len(K) >= 2 and len(T) >= 2:
                self._interpolator = RectBivariateSpline(K, T, w, kx=1, ky=1)

        elif self.method == 'svi':
            # Fit SVI per maturity slice
            for j, T_j in enumerate(T):
                F_j = self.forward[j]
                k_j = np.log(K / F_j)
                w_j = w[:, j]

                valid = np.isfinite(w_j) & (w_j > 0)
                if valid.sum() < 4:
                    continue

                fit = fit_svi_slice(k_j[valid], w_j[valid])
                if fit['success'] and fit['params'] is not None:
                    self._svi_fits[T_j] = fit['params']

    # ── IV interpolation ──────────────────────────────────────────────────────────

    def get_iv(self, K: float, T: float) -> float:
        """
        Get interpolated implied volatility at (K, T).

        Parameters:
        -----------
        K : float — strike price
        T : float — maturity (years)

        Returns:
        --------
        float : implied volatility (decimal)
        """
        K = float(K)
        T = float(T)

        # Clamp to surface bounds
        K = np.clip(K, self.strikes.min(), self.strikes.max())
        T = np.clip(T, self.maturities.min(), self.maturities.max())

        if self.method == 'spline' and self._interpolator is not None:
            w = float(self._interpolator(K, T)[0, 0])
            w = max(w, 1e-10)
            return float(np.sqrt(w / T))

        elif self.method == 'svi' and self._svi_fits:
            # Interpolate SVI across maturities
            F = self.spot * np.exp(self.r * T)
            k = np.log(K / F)
            T_fits = sorted(self._svi_fits.keys())

            if T <= T_fits[0]:
                p = self._svi_fits[T_fits[0]]
                w = svi_raw(np.array([k]), **p)[0]
            elif T >= T_fits[-1]:
                p = self._svi_fits[T_fits[-1]]
                w = svi_raw(np.array([k]), **p)[0]
            else:
                # Linear interpolation in T between two SVI fits
                j = np.searchsorted(T_fits, T) - 1
                T0, T1 = T_fits[j], T_fits[j + 1]
                alpha  = (T - T0) / (T1 - T0)
                p0, p1 = self._svi_fits[T0], self._svi_fits[T1]
                w0 = svi_raw(np.array([k]), **p0)[0]
                w1 = svi_raw(np.array([k]), **p1)[0]
                w  = (1 - alpha) * w0 + alpha * w1

            return float(np.sqrt(max(w, 1e-10) / T))

        # Fallback: nearest-neighbor
        i = np.argmin(np.abs(self.strikes    - K))
        j = np.argmin(np.abs(self.maturities - T))
        return float(self.ivs[i, j])

    def get_smile(
        self,
        T:        float,
        K_range:  Optional[tuple] = None,
        n_points: int = 100,
    ) -> pd.Series:
        """
        Get the interpolated IV smile for a given maturity.

        Parameters:
        -----------
        T        : float — maturity (years)
        K_range  : tuple — (K_min, K_max) for smile range
                           (default: surface strike range)
        n_points : int   — number of strike points

        Returns:
        --------
        pd.Series — IV smile indexed by strike
        """
        if K_range is None:
            K_range = (self.strikes.min(), self.strikes.max())
        K_arr = np.linspace(K_range[0], K_range[1], n_points)
        ivs   = np.array([self.get_iv(K, T) for K in K_arr])
        return pd.Series(ivs, index=K_arr, name=f'T={T:.2f}')

    def get_surface(
        self,
        K_grid:   Optional[np.ndarray] = None,
        T_grid:   Optional[np.ndarray] = None,
        n_K:      int = 50,
        n_T:      int = 20,
    ) -> pd.DataFrame:
        """
        Get the full interpolated IV surface as a DataFrame.

        Parameters:
        -----------
        K_grid : np.ndarray — custom strike grid (default: 50 points over surface range)
        T_grid : np.ndarray — custom maturity grid (default: 20 points over surface range)
        n_K    : int        — number of strike points if K_grid not provided
        n_T    : int        — number of maturity points if T_grid not provided

        Returns:
        --------
        pd.DataFrame — index = strikes, columns = maturities
        """
        if K_grid is None:
            K_grid = np.linspace(self.strikes.min(), self.strikes.max(), n_K)
        if T_grid is None:
            T_grid = np.linspace(self.maturities.min(), self.maturities.max(), n_T)

        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid, indexing='ij')
        iv_mesh = np.vectorize(self.get_iv)(K_mesh, T_mesh)

        return pd.DataFrame(iv_mesh, index=K_grid, columns=T_grid)

    # ── Arbitrage checks ──────────────────────────────────────────────────────────

    def check_arbitrage(self, verbose: bool = True) -> dict:
        """
        Run full arbitrage check on the surface.

        Checks:
        1. Calendar spread arbitrage  (total var non-decreasing in T)
        2. Butterfly arbitrage        (risk-neutral density >= 0) per slice
        3. IV non-negativity

        Parameters:
        -----------
        verbose : bool — print report (default: True)

        Returns:
        --------
        dict with:
            calendar   : dict — calendar arbitrage result
            butterfly  : dict — butterfly results per maturity {T: result}
            iv_ok      : bool — all IVs positive
            is_clean   : bool — True if no arbitrage detected
        """
        # 1. Calendar
        cal = check_calendar_arbitrage(self.maturities, self.total_var)

        # 2. Butterfly per maturity
        bfly = {}
        for j, T_j in enumerate(self.maturities):
            F_j = self.forward[j]
            k_j = np.log(self.strikes / F_j)
            w_j = self.total_var[:, j]
            valid = np.isfinite(w_j) & (w_j > 0)
            if valid.sum() >= 4:
                bfly[T_j] = check_butterfly_arbitrage(k_j[valid], w_j[valid])
            else:
                bfly[T_j] = {'min_density': np.nan, 'violations': [], 'is_clean': True}

        # 3. IV positivity
        iv_ok = bool(np.all(self.ivs[np.isfinite(self.ivs)] > 0))

        bfly_clean = all(v['is_clean'] for v in bfly.values())
        is_clean   = cal['is_clean'] and bfly_clean and iv_ok

        if verbose:
            print("=" * 50)
            print("  Arbitrage Check Report")
            print("=" * 50)
            cal_flag = '✅' if cal['is_clean'] else f"❌ ({len(cal['violations'])} violations)"
            print(f"  Calendar spread : {cal_flag}")

            for T_j, res in bfly.items():
                if not np.isnan(res['min_density']):
                    flag = '✅' if res['is_clean'] else f"❌ ({len(res['violations'])} violations)"
                    print(f"  Butterfly T={T_j:.2f} : {flag}  (min density={res['min_density']:.4f})")

            iv_flag = '✅' if iv_ok else '❌ (negative IVs found)'
            print(f"  IV non-negative : {iv_flag}")
            print(f"{'=' * 50}")
            print(f"  Overall         : {'✅ Clean' if is_clean else '❌ Arbitrage detected'}")
            print("=" * 50)

        return {
            'calendar':  cal,
            'butterfly': bfly,
            'iv_ok':     iv_ok,
            'is_clean':  is_clean,
        }

    # ── Plots ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        kind:   str = 'surface',
        T_list: Optional[list] = None,
        cmap:   str = 'RdYlBu_r',
    ) -> 'plt.Figure':
        """
        Plot the implied volatility surface or smile slices.

        Parameters:
        -----------
        kind   : str  — 'surface' (3D surface) or 'smile' (2D smile per maturity)
        T_list : list — for kind='smile', list of maturities to plot
                        (default: all available maturities)
        cmap   : str  — colormap for surface plot

        Returns:
        --------
        matplotlib.Figure

        Example:
        --------
        fig = vs.plot(kind='surface')
        fig = vs.plot(kind='smile', T_list=[0.25, 0.5, 1.0])
        plt.show()
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        _BLUE = '#2563EB'; _RED = '#DC2626'; _DARK = '#111827'; _GRAY = '#6B7280'
        colors = [_BLUE, _RED, '#16A34A', '#D97706', '#7C3AED',
                  '#0891B2', '#BE185D', '#B45309']

        if kind == 'surface':
            surf_df = self.get_surface(n_K=50, n_T=20)
            K_u = surf_df.index.values
            T_u = surf_df.columns.values
            Z   = surf_df.values * 100  # to %

            K_mesh, T_mesh = np.meshgrid(K_u, T_u, indexing='ij')

            fig = plt.figure(figsize=(13, 7), facecolor='white')
            ax  = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')

            surf = ax.plot_surface(K_mesh, T_mesh, Z, cmap=cmap,
                                   alpha=0.88, linewidth=0, antialiased=True)

            # ATM smile
            atm_idx = np.argmin(np.abs(K_u - self.spot))
            ax.plot(np.full(len(T_u), K_u[atm_idx]), T_u, Z[atm_idx, :],
                    color=_DARK, linewidth=1.5, linestyle='--',
                    alpha=0.7, label='ATM')

            cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
            cbar.set_label('Implied Vol (%)', fontsize=9)
            ax.set_xlabel('Strike K', fontsize=9, color=_GRAY)
            ax.set_ylabel('Maturity T (y)', fontsize=9, color=_GRAY)
            ax.set_zlabel('IV (%)', fontsize=9, color=_GRAY)
            ax.set_title(
                f'Implied Volatility Surface  |  S={self.spot}, r={self.r:.1%}',
                fontsize=11, fontweight='500', pad=12, color=_DARK
            )
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8)

        elif kind == 'smile':
            if T_list is None:
                T_list = list(self.maturities)

            fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')

            for i, T_j in enumerate(T_list):
                smile = self.get_smile(T_j) * 100
                ax.plot(smile.index, smile.values,
                        linewidth=1.8, color=colors[i % len(colors)],
                        label=f'T = {T_j:.2f}y', zorder=3)
                # Mark market data points
                j_idx = np.argmin(np.abs(self.maturities - T_j))
                ax.scatter(self.strikes,
                           self.ivs[:, j_idx] * 100,
                           color=colors[i % len(colors)],
                           s=20, alpha=0.7, zorder=4)

            ax.axvline(self.spot, color=_GRAY, linewidth=1.0,
                       linestyle=':', alpha=0.6, label=f'Spot={self.spot}')
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
            ax.set_xlabel('Strike K', fontsize=9, color=_GRAY)
            ax.set_ylabel('Implied Vol (%)', fontsize=9, color=_GRAY)
            ax.set_title(
                f'Vol Smiles  |  S={self.spot}, r={self.r:.1%}',
                fontsize=11, fontweight='500', color=_DARK
            )
            ax.tick_params(labelsize=8, colors=_GRAY)
            for sp in ax.spines.values():
                sp.set_color('#E5E7EB'); sp.set_linewidth(0.8)
            ax.grid(True, color='#E5E7EB', linewidth=0.5,
                    linestyle='--', alpha=0.7)
            ax.set_facecolor('white')
            ax.legend(fontsize=9)

        else:
            raise ValueError(f"kind must be 'surface' or 'smile', got '{kind}'.")

        fig.tight_layout()
        return fig

    # ── Repr ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"VolSurface("
            f"strikes=[{self.strikes.min():.1f}, {self.strikes.max():.1f}], "
            f"maturities=[{self.maturities.min():.2f}, {self.maturities.max():.2f}], "
            f"spot={self.spot}, method='{self.method}', "
            f"shape={self.ivs.shape})"
        )
