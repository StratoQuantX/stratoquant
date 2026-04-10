import pandas as pd
import numpy as np


# ── Input validation helper ──────────────────────────────────────────────────────

def _require_ohlcv(series, *cols):
    """Check that required OHLCV columns are present in the DataFrame."""
    missing = [c for c in cols if c not in series.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(series.columns)}"
        )


# ── Trend indicators ─────────────────────────────────────────────────────────────

def SMA(series, period=14, mode='mean'):
    """
    Rolling window statistic on a price series.

    Parameters:
    -----------
    series : pd.Series — price series
    period : int       — lookback window
    mode   : str       — 'mean' (SMA), 'upper' (rolling max), 'down' (rolling min),
                         'std' (rolling standard deviation)

    Returns:
    --------
    pd.Series
    """
    valid_modes = {'mean', 'upper', 'down', 'std'}
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: '{mode}'. Choose from {valid_modes}.")

    if mode == 'mean':
        return series.rolling(window=period).mean()
    elif mode == 'upper':
        return series.rolling(window=period).max()
    elif mode == 'down':
        return series.rolling(window=period).min()
    elif mode == 'std':
        return series.rolling(window=period).std()


def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Moving Average Convergence Divergence.

    Parameters:
    -----------
    series        : pd.Series — close price series
    fast_period   : int       — fast EMA period (default: 12)
    slow_period   : int       — slow EMA period (default: 26)
    signal_period : int       — signal line EMA period (default: 9)

    Returns:
    --------
    (macd, signal, histogram) : tuple of pd.Series
        macd      — MACD line (fast EMA - slow EMA)
        signal    — signal line (EMA of MACD)
        histogram — MACD - signal
    """
    macd      = series.ewm(span=fast_period, adjust=False).mean() - series.ewm(span=slow_period, adjust=False).mean()
    signal    = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram


def KAMA(series, n=14, fastest=2, slowest=30):
    """
    Kaufman's Adaptive Moving Average.

    Adapts its speed to market noise: fast in trending markets, slow in choppy ones.
    Uses the standard KAMA smoothing constant formula:
        SC = (ER × (fast_sc - slow_sc) + slow_sc)²
    where fast_sc = 2/(fastest+1), slow_sc = 2/(slowest+1).

    Parameters:
    -----------
    series  : pd.Series — price series
    n       : int       — efficiency ratio lookback period (default: 14)
    fastest : int       — fast EMA period for SC upper bound (default: 2)
    slowest : int       — slow EMA period for SC lower bound (default: 30)

    Returns:
    --------
    pd.Series : KAMA values
    """
    fast_sc = 2.0 / (fastest + 1)
    slow_sc = 2.0 / (slowest + 1)

    # Efficiency Ratio: directional change / total path length
    direction = (series - series.shift(n)).abs()
    volatility_path = series.diff().abs().rolling(window=n).sum()
    ER = direction / volatility_path.replace(0, np.nan)

    # Smoothing constant (squared for convexity)
    SC = (ER * (fast_sc - slow_sc) + slow_sc) ** 2

    # Initialize with float dtype to avoid object Series issues
    kama = pd.Series(np.nan, index=series.index, dtype=float)
    kama.iloc[0] = series.iloc[0]

    kama_vals  = kama.values
    sc_vals    = SC.values
    price_vals = series.values

    for t in range(1, len(series)):
        if np.isnan(sc_vals[t]):
            kama_vals[t] = kama_vals[t - 1]
        else:
            kama_vals[t] = kama_vals[t - 1] + sc_vals[t] * (price_vals[t] - kama_vals[t - 1])

    return pd.Series(kama_vals, index=series.index, dtype=float)


def Ichimoku(series, n1=9, n2=26, n3=52):
    """
    Ichimoku Kinko Hyo cloud indicator.

    Reading guide:
    - Price > Cloud  → bullish trend
    - Price < Cloud  → bearish trend
    - Price in Cloud → neutral / consolidation
    - Span A > Span B → bullish cloud
    - Span A < Span B → bearish cloud
    - Chikou Span confirms trend strength

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    n1     : int          — Tenkan-sen period (default: 9)
    n2     : int          — Kijun-sen period (default: 26)
    n3     : int          — Senkou Span B period (default: 52)

    Returns:
    --------
    (tenkan, kijun, span_a, span_b, chikou) : tuple of pd.Series
    """
    _require_ohlcv(series, 'high', 'low', 'close')
    high  = series['high']
    low   = series['low']
    close = series['close']

    tenkan = (high.rolling(window=n1).max() + low.rolling(window=n1).min()) / 2
    kijun  = (high.rolling(window=n2).max() + low.rolling(window=n2).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(n2)
    span_b = ((high.rolling(window=n3).max() + low.rolling(window=n3).min()) / 2).shift(n2)
    chikou = close.shift(-n2)

    return tenkan, kijun, span_a, span_b, chikou


# ── Momentum indicators ──────────────────────────────────────────────────────────

def RSI(series, period=14):
    """
    Relative Strength Index.

    Measures momentum: overbought above 70, oversold below 30.
    Uses Wilder's EMA smoothing (equivalent to ewm with span=period).

    Parameters:
    -----------
    series : pd.Series — close price series
    period : int       — lookback period (default: 14)

    Returns:
    --------
    pd.Series : RSI values in [0, 100]
    """
    change = series.diff()
    gain = pd.Series(np.where(change > 0, change, 0), index=series.index)
    loss = pd.Series(np.where(change < 0, -change, 0), index=series.index)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    RS  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + RS))
    return rsi


def stoch_oscillator(series, period=14, d=3):
    """
    Stochastic Oscillator (%K and %D).

    %K measures where close is relative to the high-low range over `period`.
    %D is a smoothed signal line.
    Overbought above 80, oversold below 20.

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    period : int          — %K lookback window (default: 14)
    d      : int          — %D smoothing period (default: 3)

    Returns:
    --------
    (K, D) : tuple of pd.Series
    """
    _require_ohlcv(series, 'high', 'low', 'close')
    high  = series['high']
    low   = series['low']
    close = series['close']

    rolling_low  = low.rolling(window=period).min()
    rolling_high = high.rolling(window=period).max()

    K = 100 * (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    D = K.ewm(span=d, adjust=False).mean()
    return K, D


def CCI(series, n=20):
    """
    Commodity Channel Index.

    Measures deviation of price from its average. Values above +100 suggest
    overbought; below -100 suggest oversold.

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    n      : int          — lookback period (default: 20)

    Returns:
    --------
    pd.Series : CCI values
    """
    _require_ohlcv(series, 'high', 'low', 'close')
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    ma = typical_price.rolling(window=n).mean()
    md = (typical_price - ma).abs().rolling(window=n).mean()
    return (typical_price - ma) / (0.015 * md.replace(0, np.nan))


# ── Volatility indicators ────────────────────────────────────────────────────────

def BBands(series, period=20, k=2):
    """
    Bollinger Bands.

    Parameters:
    -----------
    series : pd.Series — close price series
    period : int       — rolling window for mean and std (default: 20)
    k      : float     — number of standard deviations for band width (default: 2)

    Returns:
    --------
    (upper, middle, lower) : tuple of pd.Series
    """
    middle = series.rolling(window=period).mean()
    std    = series.rolling(window=period).std()
    return middle + k * std, middle, middle - k * std


def ATR(series, period=14):
    """
    Average True Range — measures market volatility.

    True Range = max(high-low, |high-prev_close|, |low-prev_close|).

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    period : int          — EMA smoothing period (default: 14)

    Returns:
    --------
    pd.Series : ATR values
    """
    _require_ohlcv(series, 'high', 'low', 'close')
    high_low   = series['high'] - series['low']
    high_close = (series['high'] - series['close'].shift(1)).abs()
    low_close  = (series['low']  - series['close'].shift(1)).abs()

    TR = pd.Series(
        np.maximum.reduce([high_low.values, high_close.values, low_close.values]),
        index=series.index
    )
    return TR.ewm(span=period, adjust=False).mean()


# ── Trend strength ───────────────────────────────────────────────────────────────

def ADX(series, period=14):
    """
    Average Directional Index — measures trend strength (not direction).

    ADX > 25 → strong trend; ADX < 20 → weak or no trend.
    DI_up > DI_down → bullish; DI_down > DI_up → bearish.

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    period : int          — EMA smoothing period (default: 14)

    Returns:
    --------
    (ADX, DI_up, DI_down) : tuple of pd.Series
    """
    _require_ohlcv(series, 'high', 'low', 'close')

    high_low   = series['high'] - series['low']
    high_close = (series['high'] - series['close'].shift(1)).abs()
    low_close  = (series['low']  - series['close'].shift(1)).abs()

    TR = pd.Series(
        np.maximum.reduce([high_low.values, high_close.values, low_close.values]),
        index=series.index
    )

    # Directional movement — convert to pd.Series before ewm
    DM_up = pd.Series(
        np.where(
            (series['high'] - series['high'].shift(1)) > (series['low'].shift(1) - series['low']),
            np.maximum(series['high'] - series['high'].shift(1), 0),
            0.0
        ),
        index=series.index
    )
    DM_down = pd.Series(
        np.where(
            (series['low'].shift(1) - series['low']) > (series['high'] - series['high'].shift(1)),
            np.maximum(series['low'].shift(1) - series['low'], 0),
            0.0
        ),
        index=series.index
    )

    TR_smooth   = TR.ewm(span=period, adjust=False).mean()
    DI_up       = 100 * DM_up.ewm(span=period, adjust=False).mean()   / TR_smooth
    DI_down     = 100 * DM_down.ewm(span=period, adjust=False).mean() / TR_smooth

    DX  = (100 * (DI_up - DI_down).abs() / (DI_up + DI_down).replace(0, np.nan)).fillna(0)
    adx = DX.ewm(span=period, adjust=False).mean()

    return adx, DI_up, DI_down


def Parabolic_SAR(series, acceleration=0.02, maximum=0.2):
    """
    Parabolic SAR — trend-following stop-and-reverse indicator.

    When price crosses SAR, the trend reverses and SAR flips to the other side.

    Parameters:
    -----------
    series       : pd.DataFrame — OHLCV with columns ['high', 'low', 'close']
    acceleration : float        — acceleration factor step (default: 0.02)
    maximum      : float        — maximum acceleration factor (default: 0.2)

    Returns:
    --------
    pd.Series : SAR values
    """
    _require_ohlcv(series, 'high', 'low', 'close')
    high  = series['high'].to_numpy()
    low   = series['low'].to_numpy()
    close = series['close'].to_numpy()
    n     = len(series)

    psar     = np.zeros(n)
    trend_up = True
    af       = acceleration
    ep       = high[0]
    psar[0]  = low[0]

    for i in range(1, n):
        prev_psar = psar[i - 1]
        if trend_up:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i - 1], low[i])
            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)
            if close[i] < psar[i]:
                trend_up = False
                psar[i]  = ep
                ep       = low[i]
                af       = acceleration
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i - 1], high[i])
            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)
            if close[i] > psar[i]:
                trend_up = True
                psar[i]  = ep
                ep       = high[i]
                af       = acceleration

    return pd.Series(psar, index=series.index)


# ── Volume indicators ────────────────────────────────────────────────────────────

def VWAP(series):
    """
    Volume Weighted Average Price (session cumulative).

    Benchmark for execution quality. Use VWAP_intraday for intraday reset.

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with columns ['close', 'volume']

    Returns:
    --------
    pd.Series : VWAP values
    """
    _require_ohlcv(series, 'close', 'volume')
    price  = series['close']
    volume = series['volume']
    return (price * volume).cumsum() / volume.cumsum()


def VWAP_intraday(series):
    """
    Intraday VWAP — resets at the start of each trading day.

    Uses typical price (H+L+C)/3 instead of close for a more robust anchor.

    Parameters:
    -----------
    series : pd.DataFrame — OHLCV with DatetimeIndex and columns ['high', 'low', 'close', 'volume']

    Returns:
    --------
    pd.Series : intraday VWAP values
    """
    _require_ohlcv(series, 'high', 'low', 'close', 'volume')
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    volume        = series['volume']

    df          = series.copy()
    date_groups = df.index.date

    cum_vol = volume.groupby(date_groups).cumsum()
    cum_pv  = (typical_price * volume).groupby(date_groups).cumsum()

    return cum_pv / cum_vol
