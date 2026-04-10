"""
stratoquant.plots
=================
Visualization layer for StratoQuant.

All functions return matplotlib Figure objects — ready to display in a notebook
(plt.show()) or save to file (fig.savefig('output.png', dpi=150)).

Functions:
    plot_price          — OHLCV candlestick + volume
    plot_returns        — return distribution (histogram + KDE + QQ)
    plot_greeks_profile — Greek curves vs spot (call + put on same chart)
    plot_greeks_surface — 3D surface of any Greek over (K, T) grid
    plot_vol_smile      — implied vol smile for a given expiry
    plot_payoff         — option payoff diagram at expiry
    plot_bs_surface     — BS price surface (3D) over (K, T)
    plot_portfolio      — cumulative returns + drawdown dashboard

Style: clean dark-on-white academic style, no seaborn dependency for core plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from scipy.stats import probplot, gaussian_kde
from typing import Optional, Union


# ── Style defaults ────────────────────────────────────────────────────────────────

_BLUE   = '#2563EB'
_RED    = '#DC2626'
_GREEN  = '#16A34A'
_AMBER  = '#D97706'
_GRAY   = '#6B7280'
_LIGHT  = '#F3F4F6'
_DARK   = '#111827'

def _apply_style(ax, title='', xlabel='', ylabel='', grid=True):
    """Apply consistent academic style to an Axes."""
    ax.set_title(title, fontsize=11, fontweight='500', pad=8, color=_DARK)
    ax.set_xlabel(xlabel, fontsize=9, color=_GRAY)
    ax.set_ylabel(ylabel, fontsize=9, color=_GRAY)
    ax.tick_params(labelsize=8, colors=_GRAY)
    for spine in ax.spines.values():
        spine.set_color('#E5E7EB')
        spine.set_linewidth(0.8)
    if grid:
        ax.grid(True, color='#E5E7EB', linewidth=0.5, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    ax.set_facecolor('white')
    return ax


def _fig(w=12, h=6):
    fig = plt.figure(figsize=(w, h), facecolor='white')
    fig.patch.set_facecolor('white')
    return fig


# ── 1. Price chart ────────────────────────────────────────────────────────────────

def plot_price(
    ohlcv:      pd.DataFrame,
    ticker:     str = '',
    ma_periods: list = [20, 50],
    show_volume: bool = True,
) -> plt.Figure:
    """
    OHLCV price chart with moving averages and volume bar chart.

    Parameters:
    -----------
    ohlcv       : pd.DataFrame — OHLCV with columns [open, high, low, close, volume]
                                 (output of fetch_prices or simulate_prices)
    ticker      : str          — ticker name for title
    ma_periods  : list         — list of SMA periods to overlay (default: [20, 50])
    show_volume : bool         — show volume subplot (default: True)

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    from stratoquant.data import fetch_prices
    ohlcv = fetch_prices('AAPL', period='1y')
    fig = plot_price(ohlcv, ticker='AAPL')
    plt.show()
    """
    rows = 2 if show_volume else 1
    fig  = _fig(14, 7 if show_volume else 5)
    gs   = GridSpec(rows, 1, figure=fig, hspace=0.06,
                    height_ratios=[3, 1] if show_volume else [1])

    ax_price = fig.add_subplot(gs[0])

    close = ohlcv['close']
    x = np.arange(len(ohlcv))

    # Candlestick-style: color by up/down day
    up   = ohlcv['close'] >= ohlcv['open']
    down = ~up
    w    = 0.6

    # Bodies
    ax_price.bar(x[up],   ohlcv['close'][up]   - ohlcv['open'][up],
                 bottom=ohlcv['open'][up],   width=w, color=_GREEN, alpha=0.85, zorder=3)
    ax_price.bar(x[down], ohlcv['close'][down] - ohlcv['open'][down],
                 bottom=ohlcv['open'][down], width=w, color=_RED,   alpha=0.85, zorder=3)

    # Wicks
    ax_price.vlines(x[up],   ohlcv['low'][up],   ohlcv['high'][up],
                    color=_GREEN, linewidth=0.8, zorder=2)
    ax_price.vlines(x[down], ohlcv['low'][down], ohlcv['high'][down],
                    color=_RED,   linewidth=0.8, zorder=2)

    # Moving averages
    colors_ma = [_BLUE, _AMBER, _GRAY]
    for i, p in enumerate(ma_periods):
        if len(close) > p:
            ma = close.rolling(p).mean()
            ax_price.plot(x, ma.values, linewidth=1.2,
                          color=colors_ma[i % len(colors_ma)],
                          label=f'SMA {p}', zorder=4)

    # X-axis ticks as dates
    n = len(ohlcv)
    step = max(1, n // 8)
    tick_idx = list(range(0, n, step))
    ax_price.set_xticks(tick_idx)
    ax_price.set_xticklabels(
        [ohlcv.index[i].strftime('%Y-%m-%d') for i in tick_idx],
        rotation=30, ha='right', fontsize=7
    )
    _apply_style(ax_price,
                 title=f'{ticker} — Price' if ticker else 'Price',
                 ylabel='Price')
    if ma_periods:
        ax_price.legend(fontsize=8, framealpha=0.8)

    if show_volume and 'volume' in ohlcv.columns:
        ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
        ax_vol.bar(x[up],   ohlcv['volume'][up],   width=w, color=_GREEN, alpha=0.6)
        ax_vol.bar(x[down], ohlcv['volume'][down], width=w, color=_RED,   alpha=0.6)
        ax_vol.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v/1e6:.1f}M' if v >= 1e6 else f'{v/1e3:.0f}K'
        ))
        plt.setp(ax_price.get_xticklabels(), visible=False)
        _apply_style(ax_vol, ylabel='Volume')

    fig.suptitle(ticker, fontsize=13, fontweight='600', color=_DARK, y=1.01)
    fig.tight_layout()
    return fig


# ── 2. Returns distribution ───────────────────────────────────────────────────────

def plot_returns(
    returns:  pd.Series,
    label:    str = '',
    bins:     int = 60,
) -> plt.Figure:
    """
    Return distribution: histogram + KDE + normal overlay + QQ plot + key stats.

    Parameters:
    -----------
    returns : pd.Series — return series (log or simple)
    label   : str       — series name for title
    bins    : int       — histogram bins (default: 60)

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    fig = plot_returns(ohlcv['close'].pct_change().dropna(), label='SPY')
    plt.show()
    """
    r = returns.dropna().values
    fig = _fig(13, 5)
    gs  = GridSpec(1, 2, figure=fig, wspace=0.3)

    # ── Left: histogram + KDE + normal ──
    ax1 = fig.add_subplot(gs[0])
    ax1.hist(r, bins=bins, density=True, color=_BLUE, alpha=0.35,
             edgecolor='white', linewidth=0.3, zorder=2)

    # KDE
    kde = gaussian_kde(r)
    x_kde = np.linspace(r.min(), r.max(), 300)
    ax1.plot(x_kde, kde(x_kde), color=_BLUE, linewidth=1.8, label='KDE', zorder=3)

    # Normal overlay
    from scipy.stats import norm as spnorm
    mu_r, std_r = r.mean(), r.std()
    ax1.plot(x_kde, spnorm.pdf(x_kde, mu_r, std_r),
             color=_RED, linewidth=1.5, linestyle='--', label='Normal', zorder=3)

    # Stats annotation
    skew = pd.Series(r).skew()
    kurt = pd.Series(r).kurt()
    ann_vol = std_r * np.sqrt(252)
    stats_txt = (f'μ = {mu_r:.4f}\nσ = {std_r:.4f}\n'
                 f'Ann. vol = {ann_vol:.2%}\n'
                 f'Skew = {skew:.3f}\nKurt = {kurt:.3f}')
    ax1.text(0.02, 0.97, stats_txt, transform=ax1.transAxes,
             fontsize=8, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=_LIGHT, alpha=0.9))

    ax1.legend(fontsize=8)
    _apply_style(ax1, title=f'Return Distribution — {label}' if label else 'Return Distribution',
                 xlabel='Return', ylabel='Density')

    # ── Right: QQ plot ──
    ax2 = fig.add_subplot(gs[1])
    (osm, osr), (slope, intercept, _) = probplot(r, dist='norm')
    ax2.scatter(osm, osr, s=8, color=_BLUE, alpha=0.5, zorder=3)
    x_line = np.array([osm[0], osm[-1]])
    ax2.plot(x_line, slope * x_line + intercept,
             color=_RED, linewidth=1.5, linestyle='--', label='Normal line')
    ax2.legend(fontsize=8)
    _apply_style(ax2, title='QQ Plot vs Normal',
                 xlabel='Theoretical Quantiles', ylabel='Sample Quantiles')

    fig.tight_layout()
    return fig


# ── 3. Greek profiles vs spot ─────────────────────────────────────────────────────

def plot_greeks_profile(
    K:     float,
    T:     float,
    r:     float,
    sigma: float,
    greek: str = 'delta',
    S_range: Optional[tuple] = None,
    n_points: int = 200,
) -> plt.Figure:
    """
    Plot a Greek profile (call + put) as a function of spot price.

    Parameters:
    -----------
    K        : float       — strike price
    T        : float       — time to maturity (years)
    r        : float       — risk-free rate
    sigma    : float       — implied volatility
    greek    : str         — greek to plot: 'delta','gamma','vega','theta','rho',
                             'volga','charm','vanna','speed','zomma' (default: 'delta')
    S_range  : tuple       — (S_min, S_max) for x-axis (default: K*0.5, K*1.5)
    n_points : int         — number of spot points (default: 200)

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    fig = plot_greeks_profile(K=100, T=1.0, r=0.05, sigma=0.2, greek='gamma')
    plt.show()
    """
    from .bs_greeks import all_greeks  # relative import in lib context

    if S_range is None:
        S_range = (K * 0.5, K * 1.5)
    S_arr = np.linspace(S_range[0], S_range[1], n_points)

    g_call = all_greeks(S_arr, K, T, r, sigma, 'call')[greek]
    g_put  = all_greeks(S_arr, K, T, r, sigma, 'put')[greek]  \
             if greek in ('delta', 'theta', 'rho', 'charm') else g_call

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    ax.plot(S_arr, g_call, color=_BLUE, linewidth=2.0, label='Call', zorder=3)
    if greek in ('delta', 'theta', 'rho', 'charm'):
        ax.plot(S_arr, g_put, color=_RED, linewidth=2.0,
                linestyle='--', label='Put', zorder=3)

    # Mark current ATM (K)
    ax.axvline(K, color=_GRAY, linewidth=1.0, linestyle=':', alpha=0.7, label=f'K = {K}')
    ax.axhline(0, color=_DARK, linewidth=0.5, alpha=0.3)

    # Mark spot = K greek value
    atm_idx = np.argmin(np.abs(S_arr - K))
    ax.scatter([K], [g_call[atm_idx]], color=_BLUE, s=60, zorder=5)
    ax.annotate(f'{g_call[atm_idx]:.4f}',
                xy=(K, g_call[atm_idx]),
                xytext=(8, 8), textcoords='offset points',
                fontsize=8, color=_BLUE)

    ax.legend(fontsize=9)
    _apply_style(ax,
                 title=f'{greek.capitalize()} profile  |  K={K}, T={T}y, σ={sigma:.0%}, r={r:.1%}',
                 xlabel='Spot price (S)',
                 ylabel=greek.capitalize())
    fig.tight_layout()
    return fig


# ── 4. Greeks 3D surface ──────────────────────────────────────────────────────────

def plot_greeks_surface(
    S:     float,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r:     float,
    sigma: float,
    greek: str = 'delta',
    option_type: str = 'call',
    cmap:  str = 'RdYlBu_r',
) -> plt.Figure:
    """
    3D surface plot of a Greek over a (K × T) grid.

    Parameters:
    -----------
    S           : float      — current spot price
    K_grid      : array-like — 1D array of strikes
    T_grid      : array-like — 1D array of maturities (years)
    r           : float      — risk-free rate
    sigma       : float      — implied volatility
    greek       : str        — greek name (default: 'delta')
    option_type : str        — 'call' or 'put'
    cmap        : str        — matplotlib colormap (default: 'RdYlBu_r')

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    K_grid = np.linspace(80, 120, 30)
    T_grid = np.linspace(0.1, 2.0, 20)
    fig = plot_greeks_surface(100, K_grid, T_grid, 0.05, 0.2, greek='vega')
    plt.show()
    """
    from .bs_greeks import greeks_surface as _gs

    df   = _gs(S, K_grid, T_grid, r, sigma, option_type)
    K_u  = df.index.get_level_values('K').unique().values
    T_u  = df.index.get_level_values('T').unique().values
    Z    = df[greek].values.reshape(len(K_u), len(T_u))

    K_mesh, T_mesh = np.meshgrid(K_u, T_u, indexing='ij')

    fig = plt.figure(figsize=(12, 7), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    surf = ax.plot_surface(K_mesh, T_mesh, Z, cmap=cmap, alpha=0.88,
                           linewidth=0, antialiased=True)

    # ATM line
    atm_idx = np.argmin(np.abs(K_u - S))
    ax.plot(np.full(len(T_u), K_u[atm_idx]), T_u, Z[atm_idx, :],
            color=_DARK, linewidth=1.5, linestyle='--', alpha=0.7, label='ATM')

    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08,
                 label=greek.capitalize())
    ax.set_xlabel('Strike K', fontsize=9, color=_GRAY)
    ax.set_ylabel('Maturity T (y)', fontsize=9, color=_GRAY)
    ax.set_zlabel(greek.capitalize(), fontsize=9, color=_GRAY)
    ax.set_title(
        f'{greek.capitalize()} surface — {option_type}  |  S={S}, σ={sigma:.0%}, r={r:.1%}',
        fontsize=11, fontweight='500', pad=12, color=_DARK
    )
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── 5. Vol smile ──────────────────────────────────────────────────────────────────

def plot_vol_smile(
    strikes: Union[np.ndarray, pd.Series],
    ivs:     Union[np.ndarray, pd.Series],
    spot:    Optional[float] = None,
    expiry:  str = '',
    ticker:  str = '',
    option_type: str = 'call',
    fit_svi: bool = False,
) -> plt.Figure:
    """
    Plot implied volatility smile for a given expiry.

    Parameters:
    -----------
    strikes     : array-like — strike prices
    ivs         : array-like — implied volatilities (as decimals, e.g. 0.20 = 20%)
    spot        : float      — current spot price (marks ATM if provided)
    expiry      : str        — expiry label for title
    ticker      : str        — ticker label for title
    option_type : str        — 'call' or 'put' (for title only)
    fit_svi     : bool       — fit and overlay a raw SVI parametric smile (default: False)

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    # From fetch_option_chain output:
    chain = fetch_option_chain('SPY', expiry='2025-06-20', option_type='call')
    fig = plot_vol_smile(chain['strike'], chain['implied_volatility'],
                         spot=chain.attrs['spot'], expiry='2025-06-20', ticker='SPY')
    plt.show()
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs     = np.asarray(ivs, dtype=float)

    # Drop zeros/NaNs (common in low-liquidity wings)
    mask = (ivs > 0.001) & np.isfinite(ivs)
    strikes, ivs = strikes[mask], ivs[mask]
    sort_idx = np.argsort(strikes)
    strikes, ivs = strikes[sort_idx], ivs[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    ax.scatter(strikes, ivs * 100, color=_BLUE, s=28, zorder=4, alpha=0.8, label='Market IV')
    ax.plot(strikes, ivs * 100, color=_BLUE, linewidth=1.2, alpha=0.5, zorder=3)

    if spot is not None:
        ax.axvline(spot, color=_RED, linewidth=1.2, linestyle='--',
                   alpha=0.7, label=f'Spot = {spot:.2f}')

    if fit_svi and len(strikes) >= 5:
        try:
            from scipy.optimize import minimize
            # Raw SVI: σ²(k) = a + b*(ρ*(k-m) + sqrt((k-m)² + σ²))
            # k = log(K/S)
            k = np.log(strikes / spot) if spot else np.log(strikes / strikes.mean())
            y = ivs**2  # total variance

            def svi(params, k):
                a, b, rho, m, s = params
                return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + s**2))

            def loss(params):
                a, b, rho, m, s = params
                if b < 0 or s < 0 or abs(rho) >= 1 or a < 0:
                    return 1e10
                return np.mean((svi(params, k) - y)**2)

            x0 = [y.mean(), 0.1, -0.5, 0.0, 0.1]
            res = minimize(loss, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8})

            if res.success:
                k_fine = np.linspace(k.min(), k.max(), 300)
                iv_svi = np.sqrt(np.maximum(svi(res.x, k_fine), 0)) * 100
                strikes_fine = np.exp(k_fine) * (spot if spot else strikes.mean())
                ax.plot(strikes_fine, iv_svi, color=_AMBER, linewidth=2.0,
                        linestyle='-', label='SVI fit', zorder=5)
        except Exception:
            pass  # SVI fit failed silently — still show market points

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.1f}%'))
    ax.legend(fontsize=9)
    title = f'Vol Smile — {ticker} {option_type.capitalize()}' if ticker else 'Vol Smile'
    if expiry:
        title += f'  |  Expiry {expiry}'
    _apply_style(ax, title=title, xlabel='Strike', ylabel='Implied Vol (%)')
    fig.tight_layout()
    return fig


# ── 6. Payoff diagram ─────────────────────────────────────────────────────────────

def plot_payoff(
    legs:    list,
    S_range: Optional[tuple] = None,
    n_points: int = 300,
) -> plt.Figure:
    """
    Option payoff diagram at expiry for single or multi-leg strategies.

    Parameters:
    -----------
    legs    : list of dict — each dict defines one leg:
                {
                  'K':           float — strike
                  'type':        str   — 'call' or 'put'
                  'position':    str   — 'long' or 'short'
                  'premium':     float — premium paid/received (positive = paid)
                  'quantity':    float — number of contracts (default: 1)
                  'label':       str   — optional label
                }
    S_range  : tuple  — (S_min, S_max) for x-axis (default: auto)
    n_points : int    — resolution (default: 300)

    Returns:
    --------
    matplotlib.Figure

    Examples:
    ---------
    # Long call
    fig = plot_payoff([{'K': 100, 'type': 'call', 'position': 'long', 'premium': 5.0}])

    # Long straddle
    fig = plot_payoff([
        {'K': 100, 'type': 'call', 'position': 'long', 'premium': 5.0, 'label': 'Long Call'},
        {'K': 100, 'type': 'put',  'position': 'long', 'premium': 4.0, 'label': 'Long Put'},
    ])

    # Bull spread
    fig = plot_payoff([
        {'K': 95,  'type': 'call', 'position': 'long',  'premium': 7.0},
        {'K': 105, 'type': 'call', 'position': 'short', 'premium': 3.0},
    ])
    """
    # Auto range
    all_K = [leg['K'] for leg in legs]
    K_mid = np.mean(all_K)
    if S_range is None:
        span = max(abs(K - K_mid) for K in all_K) + K_mid * 0.3
        S_range = (K_mid - span, K_mid + span)

    S = np.linspace(S_range[0], S_range[1], n_points)
    total_pnl = np.zeros(n_points)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

    leg_colors = [_BLUE, _RED, _GREEN, _AMBER, _GRAY]

    for i, leg in enumerate(legs):
        K       = leg['K']
        ltype   = leg['type'].lower()
        pos     = leg.get('position', 'long').lower()
        premium = leg.get('premium', 0.0)
        qty     = leg.get('quantity', 1.0)
        lbl     = leg.get('label', f"{pos.capitalize()} {ltype.capitalize()} K={K}")

        if ltype == 'call':
            payoff = np.maximum(S - K, 0)
        else:
            payoff = np.maximum(K - S, 0)

        sign = 1 if pos == 'long' else -1
        pnl  = sign * qty * (payoff - premium)
        total_pnl += pnl

        color = leg_colors[i % len(leg_colors)]
        ax.plot(S, pnl, linewidth=1.2, color=color, alpha=0.5,
                linestyle='--', label=lbl)

    # Total P&L
    ax.plot(S, total_pnl, linewidth=2.2, color=_DARK, label='Total P&L', zorder=5)
    ax.fill_between(S, total_pnl, 0,
                    where=(total_pnl >= 0), alpha=0.10, color=_GREEN)
    ax.fill_between(S, total_pnl, 0,
                    where=(total_pnl < 0),  alpha=0.10, color=_RED)

    # Break-even lines
    sign_changes = np.where(np.diff(np.sign(total_pnl)))[0]
    for idx in sign_changes:
        be = np.interp(0, [total_pnl[idx], total_pnl[idx+1]], [S[idx], S[idx+1]])
        ax.axvline(be, color=_GRAY, linewidth=0.8, linestyle=':', alpha=0.7)
        ax.text(be, ax.get_ylim()[0] * 0.9, f'BE\n{be:.1f}',
                ha='center', fontsize=7, color=_GRAY)

    ax.axhline(0, color=_DARK, linewidth=0.8, alpha=0.4)
    for K in all_K:
        ax.axvline(K, color=_GRAY, linewidth=0.6, linestyle=':', alpha=0.4)

    ax.legend(fontsize=8, loc='upper left')
    _apply_style(ax, title='Payoff Diagram — At Expiry',
                 xlabel='Spot Price at Expiry (S_T)',
                 ylabel='P&L')
    fig.tight_layout()
    return fig


# ── 7. BS price surface ───────────────────────────────────────────────────────────

def plot_bs_surface(
    S:     float,
    K_grid: np.ndarray,
    T_grid: np.ndarray,
    r:     float,
    sigma: float,
    option_type: str = 'call',
    cmap:  str = 'viridis',
) -> plt.Figure:
    """
    3D Black-Scholes price surface over a (K × T) grid.

    Parameters:
    -----------
    S           : float      — spot price
    K_grid      : array-like — 1D array of strikes
    T_grid      : array-like — 1D array of maturities
    r           : float      — risk-free rate
    sigma       : float      — implied volatility
    option_type : str        — 'call' or 'put'
    cmap        : str        — colormap (default: 'viridis')

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    K_grid = np.linspace(80, 120, 30)
    T_grid = np.linspace(0.1, 2.0, 20)
    fig = plot_bs_surface(100, K_grid, T_grid, 0.05, 0.2, 'call')
    plt.show()
    """
    from .pricing import bs_price_surface

    df = bs_price_surface(S, K_grid, T_grid, r, sigma, option_type)
    K_u = df.index.values
    T_u = df.columns.values
    Z   = df.values  # (nK, nT)

    K_mesh, T_mesh = np.meshgrid(K_u, T_u, indexing='ij')

    fig = plt.figure(figsize=(12, 7), facecolor='white')
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')

    surf = ax.plot_surface(K_mesh, T_mesh, Z, cmap=cmap, alpha=0.88,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.08, label='Option Price')

    # ATM line
    atm_idx = np.argmin(np.abs(K_u - S))
    ax.plot(np.full(len(T_u), K_u[atm_idx]), T_u, Z[atm_idx, :],
            color=_RED, linewidth=1.5, linestyle='--', alpha=0.8, label='ATM')

    ax.set_xlabel('Strike K', fontsize=9, color=_GRAY)
    ax.set_ylabel('Maturity T (y)', fontsize=9, color=_GRAY)
    ax.set_zlabel('Price', fontsize=9, color=_GRAY)
    ax.set_title(
        f'BS Price Surface — {option_type.capitalize()}  |  S={S}, σ={sigma:.0%}, r={r:.1%}',
        fontsize=11, fontweight='500', pad=12, color=_DARK
    )
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── 8. Portfolio dashboard ────────────────────────────────────────────────────────

def plot_portfolio(
    returns:       pd.DataFrame,
    weights:       Optional[np.ndarray] = None,
    risk_free_rate: float = 0.05,
    freq:          str = 'daily',
    benchmark:     Optional[pd.Series] = None,
    label:         str = 'Portfolio',
) -> plt.Figure:
    """
    Portfolio performance dashboard: cumulative returns + drawdown + rolling Sharpe.

    Parameters:
    -----------
    returns        : pd.DataFrame  — asset return time series (output of fetch_returns)
    weights        : array-like    — portfolio weights (default: equal weight)
    risk_free_rate : float         — annualized risk-free rate (default: 0.05)
    freq           : str           — return frequency 'daily','weekly','monthly'
    benchmark      : pd.Series     — optional benchmark return series (same index)
    label          : str           — portfolio name for legend

    Returns:
    --------
    matplotlib.Figure

    Example:
    --------
    returns = fetch_returns(['AAPL','MSFT','GOOG'], start='2022-01-01')
    fig = plot_portfolio(returns, label='Tech Portfolio')
    plt.show()
    """
    from .portfolio import portfolio_analysis

    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12, 'annual': 1}
    periods  = freq_map.get(freq, 252)

    result = portfolio_analysis(returns, weights=weights,
                                risk_free_rate=risk_free_rate, freq=freq)
    port_ret = result['returns']
    cum_ret  = result['cumulative_index']
    dd       = result['drawdown']

    fig = _fig(14, 9)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.12, height_ratios=[3, 1.5, 1.5])

    # ── 1. Cumulative returns ──
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(cum_ret.index, cum_ret.values, color=_BLUE, linewidth=1.8,
             label=label, zorder=3)

    if benchmark is not None:
        bench_cum = (1 + benchmark.reindex(port_ret.index).fillna(0)).cumprod()
        ax1.plot(bench_cum.index, bench_cum.values, color=_GRAY, linewidth=1.2,
                 linestyle='--', label='Benchmark', zorder=2, alpha=0.8)

    ax1.axhline(1.0, color=_DARK, linewidth=0.5, alpha=0.3)
    ax1.fill_between(cum_ret.index, 1.0, cum_ret.values,
                     where=(cum_ret.values >= 1.0), alpha=0.08, color=_BLUE)
    ax1.fill_between(cum_ret.index, 1.0, cum_ret.values,
                     where=(cum_ret.values < 1.0),  alpha=0.08, color=_RED)

    # Stats box
    sr   = result['sharpe_ratio']
    vol  = result['volatility']
    mdd  = result['max_drawdown']
    tot  = cum_ret.iloc[-1] - 1
    stats_txt = (f'Total return : {tot:+.1%}\n'
                 f'Ann. vol     : {vol:.1%}\n'
                 f'Sharpe       : {sr:.2f}\n'
                 f'Max drawdown : {mdd:.1%}')
    ax1.text(0.02, 0.97, stats_txt, transform=ax1.transAxes,
             fontsize=8, va='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=_LIGHT, alpha=0.9))
    ax1.legend(fontsize=9)
    _apply_style(ax1, title=f'{label} — Performance Dashboard',
                 ylabel='Wealth Index')

    # ── 2. Drawdown ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(dd.index, dd.values, 0, color=_RED, alpha=0.4)
    ax2.plot(dd.index, dd.values, color=_RED, linewidth=0.8)
    ax2.axhline(mdd, color=_DARK, linewidth=0.8, linestyle='--', alpha=0.5,
                label=f'Max DD = {mdd:.1%}')
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
    _apply_style(ax2, ylabel='Drawdown')

    # ── 3. Rolling Sharpe (1 year) ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    window = min(periods, len(port_ret) // 2)
    rf_per = risk_free_rate / periods
    roll_mean = port_ret.rolling(window).mean() - rf_per
    roll_std  = port_ret.rolling(window).std()
    roll_sr   = (roll_mean / roll_std * np.sqrt(periods)).dropna()

    ax3.plot(roll_sr.index, roll_sr.values, color=_GREEN, linewidth=1.2)
    ax3.axhline(0, color=_DARK, linewidth=0.5, alpha=0.3)
    ax3.axhline(1, color=_GREEN, linewidth=0.5, linestyle=':', alpha=0.5)
    ax3.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=(roll_sr.values >= 0), alpha=0.12, color=_GREEN)
    ax3.fill_between(roll_sr.index, roll_sr.values, 0,
                     where=(roll_sr.values < 0),  alpha=0.12, color=_RED)
    _apply_style(ax3, ylabel=f'Rolling Sharpe ({window}d)')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    fig.tight_layout()
    return fig
