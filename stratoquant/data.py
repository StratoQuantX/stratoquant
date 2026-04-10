"""
stratoquant.data
================
Unified data ingestion layer for StratoQuant.

Provides a consistent interface to fetch and standardize:
  - OHLCV price histories     → fetch_prices()
  - Multi-asset return series → fetch_returns()
  - Option chains (calls+puts + implied vol) → fetch_option_chain()
  - Risk-free rate proxy      → fetch_risk_free_rate()

All outputs follow a standard format so they plug directly into
pricing.py, bs_greeks.py, portfolio.py, and backtesting.py without
any manual cleaning.

Dependencies: yfinance, pandas, numpy
Install:      pip install yfinance
"""

import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union, Optional


# ── Internal helpers ─────────────────────────────────────────────────────────────

def _import_yfinance():
    """Lazy import with a clear error message if yfinance is missing."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        raise ImportError(
            "yfinance is required for data fetching. "
            "Install it with:  pip install yfinance"
        )


def _standardize_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Standardize a yfinance OHLCV DataFrame to StratoQuant format:
    - lowercase column names: open, high, low, close, volume
    - DatetimeIndex (timezone-naive, UTC)
    - sorted ascending
    - NaN rows dropped
    """
    df = df.copy()
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    # Keep only OHLCV columns, drop 'dividends', 'stock_splits', etc.
    ohlcv_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df = df[ohlcv_cols]

    # Timezone-naive index
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df = df.sort_index().dropna()
    df.index.name = 'date'
    df.attrs['ticker'] = ticker
    return df


def _validate_dates(start: Optional[str], end: Optional[str]):
    """Parse and validate date strings. Returns (start_str, end_str)."""
    fmt = '%Y-%m-%d'
    end_dt   = datetime.strptime(end, fmt)   if end   else datetime.today()
    start_dt = datetime.strptime(start, fmt) if start else end_dt - timedelta(days=365 * 5)

    if start_dt >= end_dt:
        raise ValueError(f"start ({start_dt.date()}) must be before end ({end_dt.date()}).")

    return start_dt.strftime(fmt), end_dt.strftime(fmt)


# ── Price history ────────────────────────────────────────────────────────────────

def fetch_prices(
    tickers: Union[str, list],
    start: Optional[str] = None,
    end:   Optional[str] = None,
    period: Optional[str] = None,
    interval: str = '1d',
    price_col: str = 'close',
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV price history for one or multiple tickers.

    Parameters:
    -----------
    tickers     : str or list  — ticker symbol(s), e.g. 'AAPL' or ['AAPL', 'MSFT', 'SPY']
    start       : str          — start date 'YYYY-MM-DD' (default: 5 years ago)
    end         : str          — end date 'YYYY-MM-DD' (default: today)
    period      : str          — yfinance shorthand: '1d','5d','1mo','3mo','6mo',
                                 '1y','2y','5y','10y','ytd','max'
                                 (overrides start/end if provided)
    interval    : str          — data frequency: '1d','1wk','1mo' (default: '1d')
    price_col   : str          — which price column to return for multi-ticker mode:
                                 'close', 'open', 'high', 'low' (default: 'close')
    auto_adjust : bool         — adjust OHLCV for splits/dividends (default: True)

    Returns:
    --------
    Single ticker  : pd.DataFrame  — full OHLCV (columns: open, high, low, close, volume)
    Multiple tickers: pd.DataFrame — price_col for each ticker, columns = tickers

    Examples:
    ---------
    # Single ticker — full OHLCV
    spy = fetch_prices('SPY', start='2020-01-01')
    spy['close'].plot()

    # Multiple tickers — close prices
    prices = fetch_prices(['AAPL', 'MSFT', 'GOOG'], start='2022-01-01')
    prices.head()

    # Short syntax with period
    prices = fetch_prices('BNP.PA', period='2y')
    """
    yf = _import_yfinance()

    single = isinstance(tickers, str)
    if single:
        tickers = [tickers]

    kwargs = dict(interval=interval, auto_adjust=auto_adjust, progress=False)
    if period:
        kwargs['period'] = period
    else:
        start, end = _validate_dates(start, end)
        kwargs['start'] = start
        kwargs['end']   = end

    if single:
        raw = yf.Ticker(tickers[0]).history(**kwargs)
        if raw.empty:
            raise ValueError(f"No data returned for '{tickers[0]}'. Check ticker and date range.")
        return _standardize_ohlcv(raw, tickers[0])

    else:
        raw = yf.download(tickers, progress=False, **kwargs)
        if raw.empty:
            raise ValueError(f"No data returned for {tickers}. Check tickers and date range.")

        # yfinance multi-ticker returns MultiIndex columns: (metric, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            col = price_col.capitalize()
            if col not in raw.columns.get_level_values(0):
                col = 'Close'  # fallback
            df = raw[col].copy()
        else:
            df = raw[['Close']].copy()

        df.columns = [c.upper() if isinstance(c, str) else c for c in df.columns]
        df.columns = tickers  # normalize to input tickers
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index().dropna(how='all')
        df.index.name = 'date'
        return df


# ── Returns ──────────────────────────────────────────────────────────────────────

def fetch_returns(
    tickers: Union[str, list],
    start:   Optional[str] = None,
    end:     Optional[str] = None,
    period:  Optional[str] = None,
    method:  str = 'log',
    interval: str = '1d',
) -> pd.DataFrame:
    """
    Fetch price history and compute return series.

    Parameters:
    -----------
    tickers  : str or list — ticker(s)
    start    : str         — start date 'YYYY-MM-DD'
    end      : str         — end date 'YYYY-MM-DD'
    period   : str         — yfinance period shorthand (overrides start/end)
    method   : str         — 'log' (log returns) or 'simple' (arithmetic returns)
    interval : str         — '1d', '1wk', '1mo'

    Returns:
    --------
    pd.DataFrame — return series, one column per ticker (NaN first row dropped)

    Examples:
    ---------
    returns = fetch_returns(['SAN.PA', 'BNP.PA', 'ACA.PA'], start='2021-01-01')
    # Plug directly into portfolio_analysis()
    from stratoquant.portfolio import portfolio_analysis
    result = portfolio_analysis(returns, freq='daily')
    """
    prices = fetch_prices(tickers, start=start, end=end, period=period,
                          interval=interval, price_col='close')

    if isinstance(prices, pd.DataFrame) and 'close' in prices.columns:
        # Single ticker with full OHLCV — use close
        prices = prices[['close']]

    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    elif method == 'simple':
        returns = prices.pct_change()
    else:
        raise ValueError(f"method must be 'log' or 'simple', got '{method}'.")

    return returns.dropna(how='all').iloc[1:]  # drop first NaN row


# ── Option chain ─────────────────────────────────────────────────────────────────

def fetch_option_chain(
    ticker:      str,
    expiry:      Optional[str] = None,
    option_type: Optional[str] = None,
    min_volume:  int   = 0,
    min_oi:      int   = 0,
    moneyness_range: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Fetch option chain for a ticker and return a clean DataFrame.

    Parameters:
    -----------
    ticker          : str            — underlying ticker (e.g. 'SPY', 'AAPL')
    expiry          : str            — expiry date 'YYYY-MM-DD'.
                                       If None, uses the nearest available expiry.
    option_type     : str or None    — 'call', 'put', or None (both)
    min_volume      : int            — filter out options with volume < min_volume
    min_oi          : int            — filter out options with open interest < min_oi
    moneyness_range : tuple or None  — (low, high) moneyness bounds, e.g. (0.8, 1.2)
                                       keeps strikes in [spot*low, spot*high]

    Returns:
    --------
    pd.DataFrame with columns:
        strike, expiry, type (call/put), bid, ask, mid, last,
        volume, open_interest, implied_volatility,
        in_the_money, underlying_price, moneyness (strike/spot)

    Examples:
    ---------
    # Nearest expiry, all strikes
    chain = fetch_option_chain('SPY')

    # Specific expiry, calls only, near-the-money
    calls = fetch_option_chain('SPY', expiry='2025-06-20',
                               option_type='call',
                               moneyness_range=(0.9, 1.1))

    # Feed IVs to your vol smile plotter
    chain['implied_volatility'].plot(x=chain['moneyness'])
    """
    yf = _import_yfinance()

    tkr = yf.Ticker(ticker)

    # Get spot price
    try:
        spot = tkr.fast_info['lastPrice']
    except Exception:
        hist = tkr.history(period='1d')
        if hist.empty:
            raise ValueError(f"Cannot fetch spot price for '{ticker}'.")
        spot = float(hist['Close'].iloc[-1])

    # Select expiry
    available = tkr.options
    if not available:
        raise ValueError(f"No option data available for '{ticker}'.")

    if expiry is None:
        expiry = available[0]
    elif expiry not in available:
        # Find nearest available
        target = pd.Timestamp(expiry)
        nearest = min(available, key=lambda d: abs(pd.Timestamp(d) - target))
        warnings.warn(
            f"Expiry '{expiry}' not available for {ticker}. "
            f"Using nearest: '{nearest}'.",
            UserWarning
        )
        expiry = nearest

    chain_raw = tkr.option_chain(expiry)

    def _clean(df: pd.DataFrame, otype: str) -> pd.DataFrame:
        df = df.copy()
        df['type']             = otype
        df['expiry']           = pd.Timestamp(expiry)
        df['underlying_price'] = spot
        df['moneyness']        = df['strike'] / spot
        df['mid']              = (df['bid'] + df['ask']) / 2

        # Standardize column names
        df = df.rename(columns={
            'impliedVolatility': 'implied_volatility',
            'openInterest':      'open_interest',
            'inTheMoney':        'in_the_money',
            'lastPrice':         'last',
        })

        keep = ['strike', 'expiry', 'type', 'bid', 'ask', 'mid', 'last',
                'volume', 'open_interest', 'implied_volatility',
                'in_the_money', 'underlying_price', 'moneyness']
        return df[[c for c in keep if c in df.columns]]

    parts = []
    if option_type in (None, 'call'):
        parts.append(_clean(chain_raw.calls, 'call'))
    if option_type in (None, 'put'):
        parts.append(_clean(chain_raw.puts, 'put'))

    result = pd.concat(parts, ignore_index=True)

    # Filters
    if min_volume > 0:
        result = result[result['volume'].fillna(0) >= min_volume]
    if min_oi > 0:
        result = result[result['open_interest'].fillna(0) >= min_oi]
    if moneyness_range is not None:
        lo, hi = moneyness_range
        result = result[(result['moneyness'] >= lo) & (result['moneyness'] <= hi)]

    result = result.sort_values(['type', 'strike']).reset_index(drop=True)
    result.attrs['ticker'] = ticker
    result.attrs['spot']   = spot
    result.attrs['expiry'] = expiry
    return result


def fetch_option_expiries(ticker: str) -> list:
    """
    Return all available option expiry dates for a ticker.

    Parameters:
    -----------
    ticker : str — underlying ticker

    Returns:
    --------
    list of str — expiry date strings 'YYYY-MM-DD'

    Example:
    --------
    expiries = fetch_option_expiries('SPY')
    print(expiries[:5])
    """
    yf = _import_yfinance()
    return list(yf.Ticker(ticker).options)


# ── Risk-free rate ────────────────────────────────────────────────────────────────

def fetch_risk_free_rate(
    tenor: str = '3m',
    as_decimal: bool = True,
) -> float:
    """
    Fetch current US risk-free rate proxy from Yahoo Finance.

    Uses US Treasury yields as risk-free rate proxies:
        '3m'  → ^IRX  (13-week T-bill)
        '2y'  → ^IRX  (proxy; actual 2Y not on YF)
        '5y'  → ^FVX  (5-year T-note)
        '10y' → ^TNX  (10-year T-note)
        '30y' → ^TYX  (30-year T-bond)

    Parameters:
    -----------
    tenor      : str  — maturity: '3m', '5y', '10y', '30y' (default: '3m')
    as_decimal : bool — return as decimal (0.045) vs percent (4.5) (default: True)

    Returns:
    --------
    float : current rate

    Example:
    --------
    r = fetch_risk_free_rate('3m')   # e.g. 0.0523
    price = black_scholes_price(100, 100, 1.0, r, 0.2, 'call')
    """
    yf = _import_yfinance()

    ticker_map = {
        '3m':  '^IRX',
        '5y':  '^FVX',
        '10y': '^TNX',
        '30y': '^TYX',
    }
    if tenor not in ticker_map:
        raise ValueError(f"tenor must be one of {list(ticker_map.keys())}, got '{tenor}'.")

    tkr = yf.Ticker(ticker_map[tenor])
    hist = tkr.history(period='5d')
    if hist.empty:
        raise ValueError(f"Could not fetch risk-free rate for tenor '{tenor}'.")

    rate_pct = float(hist['Close'].iloc[-1])
    return rate_pct / 100 if as_decimal else rate_pct


# ── Market info ───────────────────────────────────────────────────────────────────

def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch key fundamental info for a ticker.

    Returns a clean dict with: name, sector, currency, market_cap,
    pe_ratio, beta, 52w_high, 52w_low, avg_volume.

    Parameters:
    -----------
    ticker : str — ticker symbol

    Returns:
    --------
    dict

    Example:
    --------
    info = fetch_ticker_info('AAPL')
    print(info['sector'], info['market_cap'])
    """
    yf = _import_yfinance()
    raw = yf.Ticker(ticker).info

    fields = {
        'name':        raw.get('longName')       or raw.get('shortName'),
        'sector':      raw.get('sector'),
        'industry':    raw.get('industry'),
        'currency':    raw.get('currency'),
        'exchange':    raw.get('exchange'),
        'market_cap':  raw.get('marketCap'),
        'pe_ratio':    raw.get('trailingPE'),
        'beta':        raw.get('beta'),
        '52w_high':    raw.get('fiftyTwoWeekHigh'),
        '52w_low':     raw.get('fiftyTwoWeekLow'),
        'avg_volume':  raw.get('averageVolume'),
        'description': raw.get('longBusinessSummary'),
    }
    return {k: v for k, v in fields.items() if v is not None}


# ── Synthetic data (for testing without internet) ─────────────────────────────────

def simulate_prices(
    S0:     float = 100.0,
    mu:     float = 0.05,
    sigma:  float = 0.20,
    n_days: int   = 252,
    n_assets: int = 1,
    ticker_names: Optional[list] = None,
    seed:   Optional[int] = None,
    start:  str = '2023-01-01',
) -> pd.DataFrame:
    """
    Generate synthetic GBM price paths — useful for testing without internet.

    Parameters:
    -----------
    S0           : float      — initial price (default: 100)
    mu           : float      — annualized drift (default: 0.05)
    sigma        : float      — annualized volatility (default: 0.20)
    n_days       : int        — number of trading days (default: 252)
    n_assets     : int        — number of independent price paths (default: 1)
    ticker_names : list       — column names (default: ['ASSET_1', ...])
    seed         : int        — random seed
    start        : str        — start date for index (default: '2023-01-01')

    Returns:
    --------
    pd.DataFrame — OHLCV if n_assets=1, else close prices only (columns = tickers)

    Examples:
    ---------
    # Single asset with full OHLCV — plug into ATR, BBands, delta hedge
    ohlcv = simulate_prices(S0=100, sigma=0.25, n_days=252, seed=42)
    ohlcv.head()

    # Multi-asset close prices — plug into portfolio_analysis
    prices = simulate_prices(n_assets=4, ticker_names=['A','B','C','D'], seed=0)
    """
    rng = np.random.default_rng(seed)
    dt  = 1 / 252
    dates = pd.bdate_range(start=start, periods=n_days)

    if ticker_names is None:
        ticker_names = [f'ASSET_{i+1}' for i in range(n_assets)]
    elif len(ticker_names) != n_assets:
        raise ValueError("len(ticker_names) must equal n_assets.")

    # GBM paths: shape (n_days, n_assets)
    Z = rng.standard_normal((n_days, n_assets))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    prices = S0 * np.exp(np.cumsum(log_returns, axis=0))

    if n_assets == 1:
        # Build fake OHLCV for a single asset
        close = prices[:, 0]
        noise = sigma * np.sqrt(dt) * S0 * 0.5
        high  = close + np.abs(rng.normal(0, noise, n_days))
        low   = close - np.abs(rng.normal(0, noise, n_days))
        open_ = np.roll(close, 1); open_[0] = S0
        vol   = rng.integers(500_000, 5_000_000, n_days).astype(float)

        df = pd.DataFrame({
            'open':   open_,
            'high':   np.maximum(high, close),
            'low':    np.minimum(low,  close),
            'close':  close,
            'volume': vol,
        }, index=dates)
        df.index.name = 'date'
        df.attrs['ticker'] = ticker_names[0]
        return df

    else:
        df = pd.DataFrame(prices, index=dates, columns=ticker_names)
        df.index.name = 'date'
        return df