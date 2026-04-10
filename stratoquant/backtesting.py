"""
stratoquant.backtesting
=======================
Structured backtesting framework for StratoQuant.

Architecture
------------
A Strategy is a function (or callable) with the signature:

    def my_strategy(data: pd.DataFrame, params: dict) -> pd.Series:
        ...
        return signals  # pd.Series of {-1, 0, 1} indexed like data

where `data` is the OHLCV DataFrame and `signals` is a Series of position
signals: 1 = long, -1 = short, 0 = flat.

The Backtest class wires the strategy to execution:
  - applies transaction costs
  - computes returns, metrics, and drawdown
  - exposes a full report and plot

Built-in strategies (ready to use or as templates):
  - MACrossStrategy    — dual SMA crossover
  - RSIMeanReversion   — RSI oversold/overbought mean reversion
  - BollingerBreakout  — Bollinger Band breakout/reversion
  - MomentumStrategy   — trailing momentum (buy recent winners)

Usage
-----
    from stratoquant.backtesting import Backtest, MACrossStrategy
    from stratoquant.data import fetch_prices

    ohlcv = fetch_prices('SPY', start='2020-01-01')
    bt = Backtest(ohlcv, MACrossStrategy, params={'fast': 20, 'slow': 50})
    results = bt.run()
    print(bt.summary())
    fig = bt.plot()
"""

import warnings
import numpy as np
import pandas as pd
from typing import Callable, Optional, Dict, Any


# ── Internal helpers ──────────────────────────────────────────────────────────────

def _compute_metrics(
    returns:        pd.Series,
    risk_free_rate: float = 0.05,
    freq:           str   = 'daily',
) -> dict:
    """Compute full performance metrics from a return series."""
    freq_map = {'daily': 252, 'weekly': 52, 'monthly': 12, 'annual': 1}
    periods  = freq_map.get(freq, 252)
    rf_per   = risk_free_rate / periods

    r = returns.dropna()
    n = len(r)

    if n == 0:
        return {}

    # Core
    total_return   = (1 + r).prod() - 1
    ann_return     = (1 + total_return) ** (periods / n) - 1
    ann_vol        = r.std() * np.sqrt(periods)
    sharpe         = (r.mean() - rf_per) / r.std() * np.sqrt(periods) if r.std() > 0 else np.nan

    # Sortino (downside deviation)
    downside = r[r < rf_per]
    sortino  = (r.mean() - rf_per) / (downside.std() * np.sqrt(periods)) \
               if len(downside) > 1 and downside.std() > 0 else np.nan

    # Drawdown
    wealth   = (1 + r).cumprod()
    peak     = wealth.cummax()
    dd       = (wealth - peak) / peak
    max_dd   = dd.min()
    calmar   = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    # Win rate & trade stats
    wins     = (r > 0).sum()
    losses   = (r < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
    avg_win  = r[r > 0].mean() if wins > 0 else 0.0
    avg_loss = r[r < 0].mean() if losses > 0 else 0.0
    profit_factor = (wins * avg_win) / abs(losses * avg_loss) \
                    if losses > 0 and avg_loss != 0 else np.nan

    # Value at Risk / CVaR (95%)
    var_95  = np.percentile(r, 5)
    cvar_95 = r[r <= var_95].mean()

    return {
        'total_return':   total_return,
        'ann_return':     ann_return,
        'ann_volatility': ann_vol,
        'sharpe_ratio':   sharpe,
        'sortino_ratio':  sortino,
        'calmar_ratio':   calmar,
        'max_drawdown':   max_dd,
        'win_rate':       win_rate,
        'avg_win':        avg_win,
        'avg_loss':       avg_loss,
        'profit_factor':  profit_factor,
        'var_95':         var_95,
        'cvar_95':        cvar_95,
        'n_periods':      n,
    }


# ── Backtest engine ───────────────────────────────────────────────────────────────

class Backtest:
    """
    Event-driven backtesting engine.

    Parameters:
    -----------
    data            : pd.DataFrame  — OHLCV price data (output of fetch_prices/simulate_prices)
    strategy        : callable      — strategy function or class with __call__
                                      signature: (data, params) -> pd.Series of signals {-1, 0, 1}
    params          : dict          — strategy parameters passed to the strategy callable
    initial_capital : float         — starting capital in currency units (default: 10_000)
    cost_bps        : float         — transaction cost in basis points per trade (default: 10 = 0.10%)
    slippage_bps    : float         — slippage in basis points per trade (default: 5 = 0.05%)
    risk_free_rate  : float         — annualized risk-free rate for Sharpe (default: 0.05)
    freq            : str           — data frequency: 'daily','weekly','monthly'
    allow_short     : bool          — allow short positions (default: True)

    Example:
    --------
    bt = Backtest(ohlcv, MACrossStrategy, params={'fast': 20, 'slow': 50})
    results = bt.run()
    print(bt.summary())
    fig = bt.plot()
    """

    def __init__(
        self,
        data:            pd.DataFrame,
        strategy:        Callable,
        params:          Optional[Dict[str, Any]] = None,
        initial_capital: float = 10_000.0,
        cost_bps:        float = 10.0,
        slippage_bps:    float = 5.0,
        risk_free_rate:  float = 0.05,
        freq:            str   = 'daily',
        allow_short:     bool  = True,
    ):
        self.data            = data.copy()
        self.strategy        = strategy
        self.params          = params or {}
        self.initial_capital = initial_capital
        self.cost_bps        = cost_bps
        self.slippage_bps    = slippage_bps
        self.risk_free_rate  = risk_free_rate
        self.freq            = freq
        self.allow_short     = allow_short

        self._results:  Optional[pd.DataFrame] = None
        self._metrics:  Optional[dict]         = None

    # ── Run ──────────────────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """
        Execute the backtest.

        Returns:
        --------
        pd.DataFrame with columns:
            close, signal, position, trade, gross_return, cost,
            net_return, equity, drawdown
        """
        data   = self.data
        params = self.params

        # 1. Generate signals
        signals = self.strategy(data, params)
        if not isinstance(signals, pd.Series):
            raise TypeError("Strategy must return a pd.Series of signals {-1, 0, 1}.")
        signals = signals.reindex(data.index).fillna(0)

        if not self.allow_short:
            signals = signals.clip(lower=0)

        # 2. Build results DataFrame
        df = pd.DataFrame(index=data.index)
        df['close']    = data['close']
        df['signal']   = signals

        # Position = signal shifted by 1 (trade executes on next bar open)
        df['position'] = df['signal'].shift(1).fillna(0)

        # Trade = position change (1 = enter/flip, 0 = hold, -1 = exit)
        df['trade'] = df['position'].diff().fillna(0).abs()

        # 3. Returns
        df['price_return'] = df['close'].pct_change().fillna(0)
        df['gross_return'] = df['position'] * df['price_return']

        # Transaction costs + slippage (applied on trade bars)
        total_cost_bps     = (self.cost_bps + self.slippage_bps) / 10_000
        df['cost']         = df['trade'] * total_cost_bps
        df['net_return']   = df['gross_return'] - df['cost']

        # 4. Equity curve
        df['equity']   = self.initial_capital * (1 + df['net_return']).cumprod()
        peak           = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - peak) / peak

        self._results = df
        self._metrics = _compute_metrics(
            df['net_return'], self.risk_free_rate, self.freq
        )
        return df

    # ── Summary ──────────────────────────────────────────────────────────────────

    def summary(self, verbose: bool = True) -> pd.DataFrame:
        """
        Print and return a formatted performance summary table.

        Returns:
        --------
        pd.DataFrame — metrics table

        Example:
        --------
        bt.run()
        summary = bt.summary()
        """
        if self._metrics is None:
            raise RuntimeError("Call bt.run() before bt.summary().")

        fmt_map = {
            'total_return':   '{:.2%}',
            'ann_return':     '{:.2%}',
            'ann_volatility': '{:.2%}',
            'sharpe_ratio':   '{:.3f}',
            'sortino_ratio':  '{:.3f}',
            'calmar_ratio':   '{:.3f}',
            'max_drawdown':   '{:.2%}',
            'win_rate':       '{:.2%}',
            'avg_win':        '{:.4f}',
            'avg_loss':       '{:.4f}',
            'profit_factor':  '{:.3f}',
            'var_95':         '{:.4f}',
            'cvar_95':        '{:.4f}',
            'n_periods':      '{:.0f}',
        }

        labels = {
            'total_return':   'Total Return',
            'ann_return':     'Ann. Return',
            'ann_volatility': 'Ann. Volatility',
            'sharpe_ratio':   'Sharpe Ratio',
            'sortino_ratio':  'Sortino Ratio',
            'calmar_ratio':   'Calmar Ratio',
            'max_drawdown':   'Max Drawdown',
            'win_rate':       'Win Rate',
            'avg_win':        'Avg Win (daily)',
            'avg_loss':       'Avg Loss (daily)',
            'profit_factor':  'Profit Factor',
            'var_95':         'VaR 95%',
            'cvar_95':        'CVaR 95%',
            'n_periods':      'N Periods',
        }

        rows = []
        for key, val in self._metrics.items():
            if val is None or (isinstance(val, float) and np.isnan(val)):
                formatted = 'N/A'
            else:
                formatted = fmt_map.get(key, '{:.4f}').format(val)
            rows.append({'Metric': labels.get(key, key), 'Value': formatted})

        table = pd.DataFrame(rows).set_index('Metric')

        if verbose:
            print(table.to_string())

        return table

    # ── Plot ─────────────────────────────────────────────────────────────────────

    def plot(self, benchmark: Optional[pd.Series] = None) -> 'plt.Figure':
        """
        Plot the backtest results: equity curve, drawdown, positions, and returns.

        Parameters:
        -----------
        benchmark : pd.Series — optional benchmark return series for comparison

        Returns:
        --------
        matplotlib.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.gridspec import GridSpec

        if self._results is None:
            raise RuntimeError("Call bt.run() before bt.plot().")

        df = self._results
        strat_name = getattr(self.strategy, '__name__', str(self.strategy))

        _BLUE  = '#2563EB'
        _RED   = '#DC2626'
        _GREEN = '#16A34A'
        _AMBER = '#D97706'
        _GRAY  = '#6B7280'
        _DARK  = '#111827'
        _LIGHT = '#F3F4F6'

        fig = plt.figure(figsize=(14, 11), facecolor='white')
        gs  = GridSpec(4, 1, figure=fig, hspace=0.10,
                       height_ratios=[3, 1.2, 1.2, 1.2])

        def _style(ax, title='', ylabel=''):
            ax.set_title(title, fontsize=10, fontweight='500', pad=6, color=_DARK)
            ax.set_ylabel(ylabel, fontsize=8, color=_GRAY)
            ax.tick_params(labelsize=7, colors=_GRAY)
            for sp in ax.spines.values():
                sp.set_color('#E5E7EB'); sp.set_linewidth(0.7)
            ax.grid(True, color='#E5E7EB', linewidth=0.4, linestyle='--', alpha=0.6)
            ax.set_axisbelow(True)
            ax.set_facecolor('white')

        # ── 1. Equity curve ──
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df.index, df['equity'], color=_BLUE, linewidth=1.8,
                 label='Strategy', zorder=3)

        # Buy & hold
        bh = self.initial_capital * (1 + df['price_return']).cumprod()
        ax1.plot(df.index, bh, color=_GRAY, linewidth=1.2, linestyle='--',
                 label='Buy & Hold', zorder=2, alpha=0.7)

        if benchmark is not None:
            bench_eq = self.initial_capital * (1 + benchmark.reindex(df.index).fillna(0)).cumprod()
            ax1.plot(df.index, bench_eq, color=_AMBER, linewidth=1.2,
                     linestyle=':', label='Benchmark', zorder=2, alpha=0.8)

        ax1.axhline(self.initial_capital, color=_DARK, linewidth=0.5, alpha=0.3)
        ax1.fill_between(df.index, self.initial_capital, df['equity'],
                         where=(df['equity'] >= self.initial_capital),
                         alpha=0.07, color=_BLUE)
        ax1.fill_between(df.index, self.initial_capital, df['equity'],
                         where=(df['equity'] < self.initial_capital),
                         alpha=0.07, color=_RED)

        # Stats box
        m = self._metrics
        txt = (f"Total return : {m['total_return']:+.1%}\n"
               f"Ann. return  : {m['ann_return']:+.1%}\n"
               f"Sharpe       : {m['sharpe_ratio']:.2f}\n"
               f"Sortino      : {m['sortino_ratio']:.2f}\n"
               f"Max DD       : {m['max_drawdown']:.1%}\n"
               f"Win rate     : {m['win_rate']:.1%}")
        ax1.text(0.02, 0.97, txt, transform=ax1.transAxes,
                 fontsize=8, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.4', facecolor=_LIGHT, alpha=0.9))
        ax1.legend(fontsize=8, loc='upper right')
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda v, _: f'{v:,.0f}'))
        _style(ax1, title=f'Backtest — {strat_name}  |  '
               f'cost={self.cost_bps}bps  slip={self.slippage_bps}bps',
               ylabel='Equity')

        # ── 2. Drawdown ──
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(df.index, df['drawdown'], 0, color=_RED, alpha=0.40)
        ax2.plot(df.index, df['drawdown'], color=_RED, linewidth=0.8)
        ax2.axhline(m['max_drawdown'], color=_DARK, linewidth=0.8,
                    linestyle='--', alpha=0.5,
                    label=f"Max DD = {m['max_drawdown']:.1%}")
        ax2.legend(fontsize=8)
        ax2.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.0%}'))
        _style(ax2, ylabel='Drawdown')

        # ── 3. Position ──
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        pos = df['position']
        ax3.fill_between(df.index, pos, 0,
                         where=(pos > 0), color=_BLUE, alpha=0.5, label='Long')
        ax3.fill_between(df.index, pos, 0,
                         where=(pos < 0), color=_RED, alpha=0.5, label='Short')
        ax3.axhline(0, color=_DARK, linewidth=0.5, alpha=0.4)
        ax3.set_ylim(-1.5, 1.5)
        ax3.legend(fontsize=8, loc='upper right')
        _style(ax3, ylabel='Position')

        # ── 4. Daily net returns ──
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        colors = np.where(df['net_return'] >= 0, _GREEN, _RED)
        ax4.bar(df.index, df['net_return'], color=colors, alpha=0.6, width=1.0)
        ax4.axhline(0, color=_DARK, linewidth=0.5, alpha=0.4)
        ax4.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:.1%}'))
        _style(ax4, ylabel='Daily Return')

        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.setp(ax3.get_xticklabels(), visible=False)
        fig.tight_layout()
        return fig

    # ── Walk-forward ─────────────────────────────────────────────────────────────

    def walk_forward(
        self,
        train_periods: int,
        test_periods:  int,
    ) -> pd.DataFrame:
        """
        Walk-forward validation: train on rolling window, test on out-of-sample.

        Parameters:
        -----------
        train_periods : int — number of periods in each training window
        test_periods  : int — number of periods in each test window

        Returns:
        --------
        pd.DataFrame — out-of-sample returns concatenated across all folds,
                       with columns: net_return, fold

        Example:
        --------
        oos = bt.walk_forward(train_periods=252, test_periods=63)
        print(_compute_metrics(oos['net_return']))
        """
        data     = self.data
        n        = len(data)
        results  = []
        fold     = 0

        start = 0
        while start + train_periods + test_periods <= n:
            train_data = data.iloc[start : start + train_periods]
            test_data  = data.iloc[start + train_periods : start + train_periods + test_periods]

            # Run strategy on train data (allows params to adapt if strategy supports it)
            # Then generate signals on test data with same params
            signals = self.strategy(test_data, self.params)
            if not isinstance(signals, pd.Series):
                break

            signals = signals.reindex(test_data.index).fillna(0)
            if not self.allow_short:
                signals = signals.clip(lower=0)

            pos   = signals.shift(1).fillna(0)
            trade = pos.diff().fillna(0).abs()
            pr    = test_data['close'].pct_change().fillna(0)
            gross = pos * pr
            cost  = trade * (self.cost_bps + self.slippage_bps) / 10_000
            net   = gross - cost

            fold_df = pd.DataFrame({
                'net_return': net,
                'fold':       fold,
            })
            results.append(fold_df)

            start += test_periods
            fold  += 1

        if not results:
            warnings.warn("Walk-forward produced no folds. Check train/test periods vs data length.")
            return pd.DataFrame(columns=['net_return', 'fold'])

        oos = pd.concat(results)
        return oos

    # ── Properties ───────────────────────────────────────────────────────────────

    @property
    def metrics(self) -> dict:
        if self._metrics is None:
            raise RuntimeError("Call bt.run() first.")
        return self._metrics

    @property
    def results(self) -> pd.DataFrame:
        if self._results is None:
            raise RuntimeError("Call bt.run() first.")
        return self._results


# ── Built-in strategies ───────────────────────────────────────────────────────────
#
# Each strategy follows the signature:
#   strategy(data: pd.DataFrame, params: dict) -> pd.Series
#
# Signals: 1 = long, -1 = short, 0 = flat
#
# These are templates — modify or combine as needed.
# ─────────────────────────────────────────────────────────────────────────────────

def MACrossStrategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Dual SMA crossover strategy.

    Long  when fast SMA > slow SMA.
    Short when fast SMA < slow SMA.

    Params:
        fast : int — fast SMA period (default: 20)
        slow : int — slow SMA period (default: 50)

    Example:
        bt = Backtest(ohlcv, MACrossStrategy, params={'fast': 20, 'slow': 50})
    """
    fast = params.get('fast', 20)
    slow = params.get('slow', 50)

    close = data['close']
    sma_f = close.rolling(fast).mean()
    sma_s = close.rolling(slow).mean()

    signal = pd.Series(0.0, index=data.index)
    signal[sma_f > sma_s] =  1.0
    signal[sma_f < sma_s] = -1.0
    return signal.fillna(0)


def RSIMeanReversion(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    RSI mean-reversion strategy.

    Long  when RSI < oversold threshold.
    Short when RSI > overbought threshold.
    Flat  otherwise.

    Params:
        period     : int   — RSI period (default: 14)
        oversold   : float — RSI level to go long (default: 30)
        overbought : float — RSI level to go short (default: 70)

    Example:
        bt = Backtest(ohlcv, RSIMeanReversion, params={'oversold': 25, 'overbought': 75})
    """
    period     = params.get('period', 14)
    oversold   = params.get('oversold', 30)
    overbought = params.get('overbought', 70)

    close  = data['close']
    delta  = close.diff()
    gain   = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss   = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs     = gain / loss.replace(0, np.nan)
    rsi    = 100 - 100 / (1 + rs)

    signal = pd.Series(0.0, index=data.index)
    signal[rsi < oversold]   =  1.0
    signal[rsi > overbought] = -1.0
    return signal.fillna(0)


def BollingerBreakout(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Bollinger Band strategy — two modes selectable via `mode` param.

    mode='breakout'  : Long  when price breaks above upper band.
                       Short when price breaks below lower band.
    mode='reversion' : Long  when price is below lower band (buy dip).
                       Short when price is above upper band (sell rally).

    Params:
        period : int   — rolling window (default: 20)
        k      : float — number of standard deviations (default: 2)
        mode   : str   — 'breakout' or 'reversion' (default: 'reversion')

    Example:
        bt = Backtest(ohlcv, BollingerBreakout, params={'mode': 'breakout', 'k': 2.5})
    """
    period = params.get('period', 20)
    k      = params.get('k', 2.0)
    mode   = params.get('mode', 'reversion')

    close = data['close']
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + k * std
    lower = mid - k * std

    signal = pd.Series(0.0, index=data.index)

    if mode == 'breakout':
        signal[close > upper] =  1.0
        signal[close < lower] = -1.0
    elif mode == 'reversion':
        signal[close < lower] =  1.0
        signal[close > upper] = -1.0
    else:
        raise ValueError(f"mode must be 'breakout' or 'reversion', got '{mode}'.")

    return signal.fillna(0)


def MomentumStrategy(data: pd.DataFrame, params: dict) -> pd.Series:
    """
    Trailing momentum strategy.

    Long  when trailing return over `lookback` periods is positive.
    Short when trailing return is negative.
    Optional skip of the most recent `skip` periods (momentum crash protection).

    Params:
        lookback : int — momentum lookback in periods (default: 252)
        skip     : int — skip most recent N periods (default: 21)

    Example:
        bt = Backtest(ohlcv, MomentumStrategy, params={'lookback': 126, 'skip': 5})
    """
    lookback = params.get('lookback', 252)
    skip     = params.get('skip', 21)

    close = data['close']
    # Return from `lookback` periods ago to `skip` periods ago
    mom = close.shift(skip) / close.shift(lookback) - 1

    signal = pd.Series(0.0, index=data.index)
    signal[mom > 0] =  1.0
    signal[mom < 0] = -1.0
    return signal.fillna(0)
