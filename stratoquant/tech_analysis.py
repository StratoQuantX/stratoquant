import pandas as pd
import numpy as np

def SMA(series, period=14, mode = 'mean'):

    valid_modes = {'mean', 'upper', 'down', 'std'}
    
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Choose from {valid_modes}")


    if mode == 'mean':
        return series.rolling(window=period).mean()
    elif mode == 'upper':
        return series.rolling(window=period).max()
    elif mode == 'down':
        return series.rolling(window=period).min()
    elif mode == 'std':
        return series.rolling(window=period).std()
    

def RSI(series, period=14):
    change = series.diff()
    gain = np.where(change > 0, change, 0)
    loss = np.where(change < 0, -change, 0)

    gain = pd.Series(gain, index=series.index)
    loss = pd.Series(loss, index=series.index)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    RS = avg_gain / avg_loss

    RSI = 100 - (100 / (1 + RS))

    return RSI


def MACD(series, fast_period=12, slow_period=26, signal_period=9):
    macd = series.ewm(span=fast_period, adjust=False).mean() - series.ewm(span=slow_period, adjust=False).mean()
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    
    return macd, signal, histogram


def BBands(series, period=20, k=2):
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = middle + (k * std)
    lower = middle - (k * std)
    
    return upper, middle, lower

def ATR(series, period=14):
    high_low = series['high'] - series['low']

    high_close = abs(series['high'] - series['close'].shift(1))
    low_close = abs(series['low'] - series['close'].shift(1))

    TR = np.maximum.reduce([high_low, high_close, low_close])

    ATR = pd.Series(TR, index=series.index).ewm(span=period, adjust=False).mean()
    
    return ATR

def KAMA(series, n=14, fastest=2, slowest=30):
    
    ER = abs(series - series.shift(n)) / series.diff().abs().rolling(window=n).sum()

    smoothing_constant = (ER * (fastest - slowest) + slowest) ** 2

    kama = pd.Series(index=series.index)
    kama.iloc[0] = series.iloc[0]

    for t in range(1, len(series)):
        kama.iloc[t] = kama.iloc[t-1] + smoothing_constant.iloc[t] * (series.iloc[t] - kama.iloc[t-1])

    return kama

def ADX (series, period=14):
    
    high_low = series['high'] - series['low']
    high_close = abs(series['high'] - series['close'].shift(1))
    low_close = abs(series['low'] - series['close'].shift(1))

    TR = np.maximum.reduce([high_low, high_close, low_close])

    DM_up = np.where((series['high'] - series['high'].shift(1)) > (series['low'].shift(1) - series['low']),
                     series['high'] - series['high'].shift(1), 0.0)
    DM_down = np.where((series['low'].shift(1) - series['low']) > (series['high'] - series['high'].shift(1)),
                       series['low'].shift(1) - series['low'], 0.0)
    
    DI_up = 100 * (DM_up / TR).ewm(span=period, adjust=False).mean()
    DI_down = 100 * (DM_down / TR).ewm(span=period, adjust=False).mean()

    DX = (100 * abs(DI_up - DI_down) / (DI_up + DI_down)).fillna(0)

    ADX = DX.ewm(span=period, adjust=False).mean()

    return ADX, DI_up, DI_down

def Parabolic_SAR(series, acceleration=0.02, maximum=0.2):
    high = series['high'].to_numpy()
    low = series['low'].to_numpy()
    close = series['close'].to_numpy()
    n = len(series)

    psar = np.zeros(n)
    trend_up = True
    af = acceleration
    ep = high[0]
    psar[0] = low[0]


    for i in range(1, n):
        prev_psar = psar[i-1]
        if trend_up:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = min(psar[i], low[i-1], low[i])

            if high[i] > ep:
                ep = high[i]
                af = min(af + acceleration, maximum)

            if close[i] < psar[i]:
                trend_up = False
                psar[i] = ep
                ep = low[i]
                af = acceleration
        else:
            psar[i] = prev_psar + af * (ep - prev_psar)
            psar[i] = max(psar[i], high[i-1], high[i])

            if low[i] < ep:
                ep = low[i]
                af = min(af + acceleration, maximum)

            if close[i] > psar[i]:
                trend_up = True
                psar[i] = ep
                ep = high[i]
                af = acceleration

    return pd.Series(psar, index=series.index)

def stoch_oscillator(series, period=14, d=3):
    high = series['high']
    low = series['low']
    close = series['close']

    rolling_low = low.rolling(window=period).min()
    rolling_high = high.rolling(window=period).max()

    K = 100 * (close - rolling_low) / (rolling_high - rolling_low)

    D = K.ewm(span=d, adjust=False).mean()

    return K, D

def CCI(series, n=20):
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    ma = typical_price.rolling(window=n).mean()

    md = (typical_price - ma).abs().rolling(window=n).mean()

    cci = (typical_price - ma) / (0.015 * md)

    return cci

def VWAP(series):
    price = series['close']
    volume = series['volume']

    vwap = (price * volume).cumsum() / volume.cumsum()

    return pd.Series(vwap, index=series.index)

def VWAP_intraday(series):
    typical_price = (series['high'] + series['low'] + series['close']) / 3
    volume = series['volume']

    df = series.copy()
    df['typical_price'] = typical_price
    df['cum_vol'] = volume.groupby(df.index.date).cumsum()
    df['cum_pv'] = (typical_price * volume).groupby(df.index.date).cumsum()

    vwap = df['cum_pv'] / df['cum_vol']
    return pd.Series(vwap, index=series.index)

def Ichimoku(series, n1=9, n2=26, n3=52):
    """Lecture rapide
    Prix > Cloud → tendance haussière

    Prix < Cloud → tendance baissière

    Prix dans Cloud → marché neutre ou consolidation

    SpanA > SpanB → nuage haussier

    SpanA < SpanB → nuage baissier

    Chikou Span → confirme la force de la tendance """

    high = series['high']
    low = series['low']
    close = series['close']
    
    tenkan = (high.rolling(window=n1).max() + low.rolling(window=n1).min()) / 2
    kijun = (high.rolling(window=n2).max() + low.rolling(window=n2).min()) / 2

    span_a = ((tenkan + kijun) / 2).shift(-n2)
    span_b = ((high.rolling(window=n3).max() + low.rolling(window=n3).min()) / 2).shift(-n2)

    chikou = close.shift(n2)

    return tenkan, kijun, span_a, span_b, chikou
