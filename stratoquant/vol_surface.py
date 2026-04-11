"""
stratoquant.vol_surface
=======================
Implied volatility surface construction and analysis.

Two interpolation methods:
  - Spline  : adaptive order (cubic when enough data, linear otherwise)
  - SVI     : Gatheral's raw parametrization per maturity slice

Arbitrage checks:
  - Calendar spread  (total variance non-decreasing in T)
  - Butterfly        (risk-neutral density >= 0)
  - IV non-negativity

Usage
-----
    from stratoquant.vol_surface import VolSurface

    # From raw option chain (fetch_option_chain output)
    chain = sq.fetch_option_chain('SPY', expiry='2025-06-20')
    vs = VolSurface.from_chain(chain, spot=500, r=0.05)

    # From compute_iv_dataframe output (multiple expiries combined)
    vs = VolSurface.from_chain(combined_iv_df, spot=spot, r=r)

    # From a (K, T) → IV grid directly
    vs = VolSurface(strikes, maturities, ivs, spot=500, r=0.05)

    iv    = vs.get_iv(K=510, T=0.5)      # interpolated IV
    smile = vs.get_smile(T=0.5)          # full smile at one maturity
    surf  = vs.get_surface()             # full IV surface as DataFrame
    arb   = vs.check_arbitrage()         # calendar + butterfly checks
    fig   = vs.plot(kind='surface')      # 3D surface plot
    fig   = vs.plot(kind='smile')        # 2D smile slices

Dependencies: numpy, pandas, scipy, matplotlib
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union
from scipy.interpolate import CubicSpline, RectBivariateSpline
from scipy.optimize import minimize


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
    rho   : float      — smile rotation (-1 < rho < 1)
    m     : float      — ATM shift (horizontal translation)
    sigma : float      — smile curvature (> 0)

    Returns:
    --------
    np.ndarray — total implied variance w(k) = IV^2 * T
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))


def fit_svi_slice(
    log_moneyness: np.ndarray,
    total_var:     np.ndarray,
    weights:       Optional[np.ndarray] = None,
    n_restarts:    int = 5,
) -> dict:
    """
    Fit SVI parameters to a single maturity slice via L-BFGS-B with restarts.

    Parameters:
    -----------
    log_moneyness : np.ndarray — k = log(K/F) for each strike
    total_var     : np.ndarray — market total variance = IV^2 * T
    weights       : np.ndarray — optional per-strike weights (e.g. vega weights)
    n_restarts    : int        — number of random restarts (default: 5)

    Returns:
    --------
    dict with: params (a,b,rho,m,sigma), rmse, success
    """
    k  = np.asarray(log_moneyness, dtype=float)
    w  = np.asarray(total_var,     dtype=float)
    wt = np.ones(len(k)) if weights is None else np.asarray(weights, dtype=float)

    def loss(params):
        a, b, rho, m, sigma = params
        if b < 0 or abs(rho) >= 1 or sigma <= 0:
            return 1e10
        w_model = svi_raw(k, a, b, rho, m, sigma)
        if np.any(w_model < 0):
            return 1e10
        return float(np.sum(wt * (w - w_model)**2))

    bounds = [
        (0.0,   w.max() * 2),
        (0.0,   2.0),
        (-0.999, 0.999),
        (k.min() - 0.5, k.max() + 0.5),
        (1e-4,  1.0),
    ]

    best_result = None
    best_loss   = np.inf
    rng = np.random.default_rng(42)

    for i in range(n_restarts):
        x0 = ([w.mean(), 0.1, -0.3, float(k.mean()), 0.2] if i == 0 else [
            float(rng.uniform(0.001, w.max())),
            float(rng.uniform(0.01,  0.5)),
            float(rng.uniform(-0.8,  0.0)),
            float(rng.uniform(k.min(), k.max())),
            float(rng.uniform(0.05,  0.5)),
        ])
        try:
            res = minimize(loss, x0, method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 1000})
            if res.fun < best_loss:
                best_loss, best_result = res.fun, res
        except Exception:
            continue

    if best_result is None:
        return {'params': None, 'rmse': np.inf, 'success': False}

    a, b, rho, m, sigma = best_result.x
    rmse = float(np.sqrt(np.mean((w - svi_raw(k, a, b, rho, m, sigma))**2)))
    return {
        'params':  {'a': a, 'b': b, 'rho': rho, 'm': m, 'sigma': sigma},
        'rmse':    rmse,
        'success': best_result.success,
    }


# ── Arbitrage checks ──────────────────────────────────────────────────────────────

def check_calendar_arbitrage(
    maturities:       np.ndarray,
    total_var_matrix: np.ndarray,
) -> dict:
    """
    Calendar spread check: total variance must be non-decreasing in T.

    Parameters:
    -----------
    maturities        : np.ndarray — sorted 1D array of maturities (years)
    total_var_matrix  : np.ndarray — shape (n_K, n_T), w(k, T) = IV^2 * T

    Returns:
    --------
    dict: violations (list), is_clean (bool)
    """
    violations = []
    n_K, n_T = total_var_matrix.shape
    for i in range(n_K):
        for j in range(n_T - 1):
            if total_var_matrix[i, j] > total_var_matrix[i, j + 1] + 1e-8:
                violations.append({
                    'K_idx':  i,
                    'T_from': float(maturities[j]),
                    'T_to':   float(maturities[j + 1]),
                    'excess': float(total_var_matrix[i, j] - total_var_matrix[i, j + 1]),
                })
    return {'violations': violations, 'is_clean': len(violations) == 0}


def check_butterfly_arbitrage(
    log_moneyness: np.ndarray,
    total_var:     np.ndarray,
    n_points:      int = 200,
) -> dict:
    """
    Butterfly arbitrage check: risk-neutral density g(k) must be >= 0.

    g(k) = (1 - k*w'/(2w))^2 - (w'/2)^2*(1/4 + 1/w) + w''/2

    Parameters:
    -----------
    log_moneyness : np.ndarray — k values (sorted)
    total_var     : np.ndarray — total variance w(k) for this maturity
    n_points      : int        — resolution for derivative check (default: 200)

    Returns:
    --------
    dict: min_density (float), violations (list of k values), is_clean (bool)
    """
    k = np.asarray(log_moneyness, dtype=float)
    w = np.asarray(total_var,     dtype=float)

    idx  = np.argsort(k)
    k, w = k[idx], w[idx]

    if len(k) < 4:
        return {'min_density': np.nan, 'violations': [], 'is_clean': True}

    cs     = CubicSpline(k, w)
    k_fine = np.linspace(k[0], k[-1], n_points)
    w_f    = cs(k_fine)
    w1     = cs(k_fine, 1)
    w2     = cs(k_fine, 2)

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


# ── Internal helpers ──────────────────────────────────────────────────────────────

def _build_iv_matrix(
    df:     pd.DataFrame,
    iv_col: str,
) -> tuple:
    """
    Build a (strikes × maturities) IV matrix from a flat DataFrame.

    Handles irregular grids by pivoting then interpolating NaNs per strike
    using linear interpolation in the maturity direction.

    Returns: (strikes, maturities, ivs) — all numpy arrays, ivs shape (nK, nT)
    """
    pivot = df.pivot_table(
        index='strike', columns='T',
        values=iv_col, aggfunc='mean',
    ).dropna(how='all')

    strikes    = pivot.index.values.astype(float)
    maturities = pivot.columns.values.astype(float)
    ivs        = pivot.values.copy()   # writeable copy — critical fix

    # Fill NaNs per strike via linear interpolation across maturities
    for i in range(len(strikes)):
        row  = ivs[i].copy()
        mask = np.isfinite(row)
        if mask.sum() == 0:
            continue
        if mask.sum() == 1:
            ivs[i] = row[mask][0]   # constant fill
        elif mask.sum() >= 2:
            ivs[i] = np.interp(maturities, maturities[mask], row[mask])

    return strikes, maturities, ivs


# ── VolSurface class ──────────────────────────────────────────────────────────────

class VolSurface:
    """
    Implied volatility surface with interpolation and arbitrage checks.

    Interpolation methods:
      - 'spline' : RectBivariateSpline — adaptive order (cubic if enough data,
                   linear if few maturities). Stable on real market data.
      - 'svi'    : SVI parametric fit per maturity slice, linear interpolation
                   in T. Arbitrage-aware but slower.

    Parameters:
    -----------
    strikes    : np.ndarray — 1D array of unique strikes
    maturities : np.ndarray — 1D array of unique maturities (years)
    ivs        : np.ndarray — IV surface, shape (n_strikes, n_maturities), decimal
    spot       : float      — current spot price
    r          : float      — risk-free rate
    method     : str        — 'spline' or 'svi' (default: 'spline')
    forward    : np.ndarray — optional forward prices per maturity
                              (default: S * exp(r * T))

    Example:
    --------
    K  = np.linspace(90, 110, 5)
    T  = np.array([0.25, 0.5, 1.0])
    IV = np.array([[0.25, 0.22, 0.20],
                   [0.22, 0.20, 0.19],
                   [0.20, 0.19, 0.18],
                   [0.22, 0.20, 0.19],
                   [0.25, 0.22, 0.20]])
    vs = VolSurface(K, T, IV, spot=100, r=0.05)
    vs.get_iv(K=100, T=0.5)
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
        self.ivs        = np.asarray(ivs,        dtype=float)
        self.spot       = float(spot)
        self.r          = float(r)
        self.method     = method

        self.forward = (
            np.asarray(forward, dtype=float)
            if forward is not None
            else spot * np.exp(r * self.maturities)
        )

        self.log_moneyness = np.log(
            self.strikes[:, None] / self.forward[None, :]
        )
        self.total_var = self.ivs**2 * self.maturities[None, :]

        self._interpolator = None
        self._svi_fits     = {}
        self._build_interpolator()

    # ── Constructors ─────────────────────────────────────────────────────────────

    @classmethod
    def from_chain(
        cls,
        chain:      pd.DataFrame,
        spot:       float,
        r:          float,
        method:     str   = 'spline',
        min_iv:     float = 0.001,
        min_volume: int   = 0,
    ) -> 'VolSurface':
        """
        Build a VolSurface from a fetch_option_chain() or compute_iv_dataframe() output.

        Accepts two column name conventions:
          - 'implied_volatility' : raw fetch_option_chain output
          - 'implied_vol'        : compute_iv_dataframe output

        Accepts two maturity column conventions:
          - 'maturity' : pre-computed T in years (compute_iv_dataframe pipeline)
          - 'expiry'   : expiry date (fetch_option_chain output)

        Handles irregular strike grids across expiries by interpolating NaNs
        per strike in the maturity direction before building the surface.

        Parameters:
        -----------
        chain      : pd.DataFrame — flat option data, one row per option
        spot       : float        — current spot price
        r          : float        — risk-free rate
        method     : str          — 'spline' or 'svi' (default: 'spline')
        min_iv     : float        — minimum IV filter (default: 0.001)
        min_volume : int          — minimum volume filter (default: 0)

        Returns:
        --------
        VolSurface

        Example:
        --------
        # From raw chain
        chain = sq.fetch_option_chain('SPY', expiry='2025-12-19')
        vs = VolSurface.from_chain(chain, spot=chain.attrs['spot'], r=0.05)

        # From compute_iv_dataframe (multiple expiries)
        iv_df = sq.compute_iv_dataframe(chain, S=spot, r=r)
        vs    = VolSurface.from_chain(iv_df, spot=spot, r=r)
        """
        df = chain.copy()

        # Resolve IV column
        if 'implied_volatility' in df.columns:
            iv_col = 'implied_volatility'
        elif 'implied_vol' in df.columns:
            iv_col = 'implied_vol'
        else:
            raise ValueError(
                "chain must have an 'implied_volatility' or 'implied_vol' column. "
                "Run compute_iv_dataframe() first if starting from raw prices."
            )

        # Filter
        df = df[df[iv_col] > min_iv].copy()
        if 'volume' in df.columns and min_volume > 0:
            df = df[df['volume'].fillna(0) >= min_volume]

        # Resolve maturity column → column 'T' in years
        if 'maturity' in df.columns:
            df['T'] = df['maturity'].astype(float)
        elif 'expiry' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['expiry']):
                today  = pd.Timestamp.today().normalize()
                df['T'] = (df['expiry'] - today).dt.days / 365.0
            else:
                df['T'] = pd.to_datetime(df['expiry']).apply(
                    lambda d: (d - pd.Timestamp.today()).days / 365.0
                )
        else:
            raise ValueError(
                "chain must have a 'maturity' or 'expiry' column."
            )

        df = df[df['T'] > 0].copy()

        if df.empty:
            raise ValueError("No valid options after filtering. Check min_iv and T > 0.")

        strikes, maturities, ivs = _build_iv_matrix(df, iv_col)
        return cls(strikes, maturities, ivs, spot, r, method=method)

    # ── Interpolator ─────────────────────────────────────────────────────────────

    def _build_interpolator(self):
        """
        Build the internal interpolator.

        Spline order adapts to data density:
          - kx=3, ky=3  if >= 5 strikes and >= 4 maturities
          - kx=3, ky=1  if >= 5 strikes but < 4 maturities (common with real data)
          - kx=1, ky=1  fallback for sparse grids
        """
        K = self.strikes
        T = self.maturities
        w = self.total_var

        if self.method == 'spline':
            kx = 3 if len(K) >= 5 else 1
            ky = 3 if len(T) >= 4 else 1
            try:
                self._interpolator = RectBivariateSpline(K, T, w, kx=kx, ky=ky)
            except Exception:
                try:
                    self._interpolator = RectBivariateSpline(K, T, w, kx=1, ky=1)
                except Exception:
                    self._interpolator = None

        elif self.method == 'svi':
            for j, T_j in enumerate(T):
                F_j   = self.forward[j]
                k_j   = np.log(K / F_j)
                w_j   = w[:, j]
                valid = np.isfinite(w_j) & (w_j > 0)
                if valid.sum() < 4:
                    continue
                fit = fit_svi_slice(k_j[valid], w_j[valid])
                if fit['success'] and fit['params'] is not None:
                    self._svi_fits[T_j] = fit['params']

    # ── IV interpolation ──────────────────────────────────────────────────────────

    def get_iv(self, K: float, T: float) -> float:
        """
        Interpolated implied volatility at (K, T).

        Parameters:
        -----------
        K : float — strike price
        T : float — maturity (years)

        Returns:
        --------
        float : implied volatility (decimal, e.g. 0.20 = 20%)
        """
        K = float(np.clip(K, self.strikes.min(),    self.strikes.max()))
        T = float(np.clip(T, self.maturities.min(), self.maturities.max()))

        if self.method == 'spline' and self._interpolator is not None:
            w = float(self._interpolator(K, T)[0, 0])
            return float(np.sqrt(max(w, 1e-10) / T))

        if self.method == 'svi' and self._svi_fits:
            F      = self.spot * np.exp(self.r * T)
            k      = np.log(K / F)
            T_fits = sorted(self._svi_fits.keys())
            if T <= T_fits[0]:
                w = float(svi_raw(np.array([k]), **self._svi_fits[T_fits[0]])[0])
            elif T >= T_fits[-1]:
                w = float(svi_raw(np.array([k]), **self._svi_fits[T_fits[-1]])[0])
            else:
                j      = np.searchsorted(T_fits, T) - 1
                T0, T1 = T_fits[j], T_fits[j + 1]
                alpha  = (T - T0) / (T1 - T0)
                w0     = float(svi_raw(np.array([k]), **self._svi_fits[T0])[0])
                w1     = float(svi_raw(np.array([k]), **self._svi_fits[T1])[0])
                w      = (1 - alpha) * w0 + alpha * w1
            return float(np.sqrt(max(w, 1e-10) / T))

        # Fallback: nearest neighbour
        i = int(np.argmin(np.abs(self.strikes    - K)))
        j = int(np.argmin(np.abs(self.maturities - T)))
        return float(self.ivs[i, j])

    def get_smile(
        self,
        T:        float,
        K_range:  Optional[tuple] = None,
        n_points: int = 100,
    ) -> pd.Series:
        """
        Interpolated IV smile at a given maturity.

        Parameters:
        -----------
        T        : float — maturity (years)
        K_range  : tuple — (K_min, K_max), default: surface range
        n_points : int   — number of strike points (default: 100)

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
        K_grid: Optional[np.ndarray] = None,
        T_grid: Optional[np.ndarray] = None,
        n_K:    int = 50,
        n_T:    int = 20,
    ) -> pd.DataFrame:
        """
        Full interpolated IV surface as a DataFrame.

        Parameters:
        -----------
        K_grid : np.ndarray — custom strike grid (default: n_K evenly spaced)
        T_grid : np.ndarray — custom maturity grid (default: n_T evenly spaced)
        n_K    : int        — number of strike points (default: 50)
        n_T    : int        — number of maturity points (default: 20)

        Returns:
        --------
        pd.DataFrame — index = strikes, columns = maturities, values = IV (decimal)
        """
        if K_grid is None:
            K_grid = np.linspace(self.strikes.min(),    self.strikes.max(),    n_K)
        if T_grid is None:
            T_grid = np.linspace(self.maturities.min(), self.maturities.max(), n_T)

        K_mesh, T_mesh = np.meshgrid(K_grid, T_grid, indexing='ij')
        iv_mesh        = np.vectorize(self.get_iv)(K_mesh, T_mesh)
        return pd.DataFrame(iv_mesh, index=K_grid, columns=T_grid)

    # ── Arbitrage checks ──────────────────────────────────────────────────────────

    def check_arbitrage(self, verbose: bool = True) -> dict:
        """
        Full arbitrage check: calendar spread + butterfly + IV positivity.

        Parameters:
        -----------
        verbose : bool — print report (default: True)

        Returns:
        --------
        dict: calendar, butterfly, iv_ok, is_clean
        """
        cal  = check_calendar_arbitrage(self.maturities, self.total_var)

        bfly = {}
        for j, T_j in enumerate(self.maturities):
            F_j   = self.forward[j]
            k_j   = np.log(self.strikes / F_j)
            w_j   = self.total_var[:, j]
            valid = np.isfinite(w_j) & (w_j > 0)
            if valid.sum() >= 4:
                bfly[T_j] = check_butterfly_arbitrage(k_j[valid], w_j[valid])
            else:
                bfly[T_j] = {'min_density': np.nan, 'violations': [], 'is_clean': True}

        iv_ok      = bool(np.all(self.ivs[np.isfinite(self.ivs)] > 0))
        bfly_clean = all(v['is_clean'] for v in bfly.values())
        is_clean   = cal['is_clean'] and bfly_clean and iv_ok

        if verbose:
            print("=" * 52)
            print("  Arbitrage Check Report")
            print("=" * 52)
            print(f"  Calendar spread : {'✅' if cal['is_clean'] else f'❌ ({len(cal[chr(118)+chr(105)+chr(111)+chr(108)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(115)])} violations)'}")
            for T_j, res in bfly.items():
                if not np.isnan(res['min_density']):
                    flag = '✅' if res['is_clean'] else f"❌ ({len(res['violations'])} violations)"
                    print(f"  Butterfly T={T_j:.2f} : {flag}  "
                          f"(min density={res['min_density']:.4f})")
            print(f"  IV non-negative : {'✅' if iv_ok else '❌ (negative IVs found)'}")
            print("=" * 52)
            print(f"  Overall         : {'✅ Clean' if is_clean else '❌ Arbitrage detected'}")
            print("=" * 52)

        return {'calendar': cal, 'butterfly': bfly, 'iv_ok': iv_ok, 'is_clean': is_clean}

    # ── Plots ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        kind:   str = 'surface',
        T_list: Optional[list] = None,
        cmap:   str = 'RdYlBu_r',
    ) -> 'plt.Figure':
        """
        Plot the IV surface (3D) or smile slices (2D).

        Parameters:
        -----------
        kind   : str  — 'surface' or 'smile' (default: 'surface')
        T_list : list — maturities to plot for kind='smile'
                        (default: all available maturities)
        cmap   : str  — matplotlib colormap (default: 'RdYlBu_r')

        Returns:
        --------
        matplotlib.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from mpl_toolkits.mplot3d import Axes3D  # noqa

        _BLUE  = '#2563EB'
        _RED   = '#DC2626'
        _DARK  = '#111827'
        _GRAY  = '#6B7280'
        _LIGHT = '#F3F4F6'
        COLORS = [_BLUE, _RED, '#16A34A', '#D97706',
                  '#7C3AED', '#0891B2', '#BE185D', '#B45309']

        if kind == 'surface':
            surf_df = self.get_surface(n_K=60, n_T=25)
            K_u     = surf_df.index.values
            T_u     = surf_df.columns.values

            # Clip to physical range before plotting
            Z = np.clip(surf_df.values * 100, 0.5, 150)

            K_mesh, T_mesh = np.meshgrid(K_u, T_u, indexing='ij')

            fig = plt.figure(figsize=(13, 7), facecolor='white')
            ax  = fig.add_subplot(111, projection='3d')
            ax.set_facecolor('white')

            surf = ax.plot_surface(K_mesh, T_mesh, Z, cmap=cmap,
                                   alpha=0.90, linewidth=0, antialiased=True)

            # ATM line
            atm_idx = int(np.argmin(np.abs(K_u - self.spot)))
            ax.plot(np.full(len(T_u), K_u[atm_idx]), T_u, Z[atm_idx, :],
                    color=_DARK, linewidth=1.8, linestyle='--',
                    alpha=0.8, label='ATM')

            cbar = fig.colorbar(surf, ax=ax, shrink=0.45, pad=0.08)
            cbar.set_label('Implied Vol (%)', fontsize=9)
            ax.set_xlabel('Strike K',       fontsize=9, color=_GRAY)
            ax.set_ylabel('Maturity T (y)', fontsize=9, color=_GRAY)
            ax.set_zlabel('IV (%)',          fontsize=9, color=_GRAY)
            ax.set_title(
                f'Implied Volatility Surface  |  '
                f'S={self.spot:.2f}, r={self.r:.1%}',
                fontsize=11, fontweight='500', pad=12, color=_DARK,
            )
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=8)

        elif kind == 'smile':
            if T_list is None:
                T_list = list(self.maturities)

            fig, ax = plt.subplots(figsize=(11, 6), facecolor='white')

            for i, T_j in enumerate(T_list):
                smile = self.get_smile(T_j) * 100
                color = COLORS[i % len(COLORS)]
                ax.plot(smile.index, smile.values,
                        linewidth=1.8, color=color,
                        label=f'T = {T_j:.2f}y', zorder=3)
                # Overlay raw market data points
                j_idx = int(np.argmin(np.abs(self.maturities - T_j)))
                ax.scatter(self.strikes, self.ivs[:, j_idx] * 100,
                           color=color, s=22, alpha=0.75, zorder=4)

            ax.axvline(self.spot, color=_GRAY, linewidth=1.0,
                       linestyle=':', alpha=0.6, label=f'Spot = {self.spot:.2f}')
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
            ax.set_xlabel('Strike K',       fontsize=9, color=_GRAY)
            ax.set_ylabel('Implied Vol (%)', fontsize=9, color=_GRAY)
            ax.set_title(
                f'Vol Smiles  |  S={self.spot:.2f}, r={self.r:.1%}',
                fontsize=11, fontweight='500', color=_DARK,
            )
            ax.tick_params(labelsize=8, colors=_GRAY)
            for sp in ax.spines.values():
                sp.set_color('#E5E7EB')
                sp.set_linewidth(0.8)
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
            f"maturities=[{self.maturities.min():.3f}, {self.maturities.max():.3f}], "
            f"spot={self.spot:.2f}, "
            f"method='{self.method}', "
            f"shape={self.ivs.shape})"
        )
