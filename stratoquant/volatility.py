import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from .pricing import black_scholes_price
from .bs_greeks import vega


# ── Scalar implied volatility ─────────────────────────────────────────────────────

def implied_volatility(market_price, S, K, T, r, option_type='call', verbose=False):
    """
    Calculate the implied volatility using the Black-Scholes model (Brent's method).

    Parameters:
    -----------
    market_price : float — observed market price of the option
    S            : float — spot price
    K            : float — strike price
    T            : float — time to maturity (years)
    r            : float — risk-free rate
    option_type  : str   — 'call' or 'put'
    verbose      : bool  — print result (default: False)

    Returns:
    --------
    float : implied volatility (annualized decimal, e.g. 0.20 = 20%)

    Raises:
    -------
    ValueError  : if market_price violates no-arbitrage bounds
    RuntimeError: if Brent's method fails to converge
    """
    intrinsic  = max(S - K * np.exp(-r * T), 0) if option_type == 'call' else max(K * np.exp(-r * T) - S, 0)
    upper_bound = S if option_type == 'call' else K * np.exp(-r * T)

    if market_price < intrinsic:
        raise ValueError(
            f"market_price={market_price:.4f} is below intrinsic={intrinsic:.4f}. "
            "No implied vol exists (arbitrage violation)."
        )
    if market_price >= upper_bound:
        raise ValueError(
            f"market_price={market_price:.4f} is above the upper BS bound={upper_bound:.4f}."
        )

    def error_function(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    try:
        implied_vol = brentq(error_function, 1e-6, 5.0, xtol=1e-8, maxiter=500)
    except ValueError as e:
        raise RuntimeError(
            f"Brent's method failed for {option_type} "
            f"(S={S}, K={K}, T={T}, price={market_price}). Error: {e}"
        )

    if verbose:
        print(f"Implied Volatility ({option_type}): {implied_vol:.6f} ({implied_vol*100:.2f}%)")

    return implied_vol


# ── B. IV DataFrame from a grid of option prices ──────────────────────────────────

def compute_iv_dataframe(
    prices_df:   pd.DataFrame,
    S:           float,
    r:           float,
    price_col:   str   = 'price',
    strike_col:  str   = 'strike',
    maturity_col: str  = 'maturity',
    type_col:    str   = 'option_type',
    default_type: str  = 'call',
    min_price:   float = 0.01,
    verbose:     bool  = False,
) -> pd.DataFrame:
    """
    Compute implied volatility for each row of an option prices DataFrame.

    Takes a flat DataFrame of option prices (one row per option) and inverts
    Black-Scholes via Brent's method to recover the IV for each contract.
    Rows where IV cannot be computed (arbitrage violations, no convergence)
    get NaN with a reason column for diagnostics.

    Parameters:
    -----------
    prices_df    : pd.DataFrame — input DataFrame, must contain at minimum:
                                  strike, maturity (in years), price columns.
                                  Optionally an option_type column.
    S            : float        — current spot price
    r            : float        — risk-free rate
    price_col    : str          — name of the column containing option prices
                                  (default: 'price'). Also accepts 'mid', 'last',
                                  'bid', 'ask' — pass whichever is most reliable.
    strike_col   : str          — name of the strike column (default: 'strike')
    maturity_col : str          — name of the maturity column in years (default: 'maturity')
    type_col     : str          — name of the option type column (default: 'option_type')
    default_type : str          — option type to use if type_col is missing (default: 'call')
    min_price    : float        — minimum price to attempt IV computation;
                                  rows below this threshold are skipped (default: 0.01)
    verbose      : bool         — print progress every 100 rows (default: False)

    Returns:
    --------
    pd.DataFrame — input DataFrame with two new columns appended:
        implied_vol   : float — IV in decimal form (e.g. 0.20 = 20%), NaN if failed
        iv_error      : str   — error reason if IV computation failed, else None
        moneyness     : float — log(K/F) where F = S*exp(r*T)
        intrinsic     : float — intrinsic value of the option

    Example:
    --------
    # From fetch_option_chain output
    chain = sq.fetch_option_chain('SPY', expiry='2025-12-19')
    chain['maturity'] = (pd.Timestamp('2025-12-19') - pd.Timestamp.today()).days / 365
    chain['price'] = chain['mid']

    iv_df = sq.compute_iv_dataframe(chain, S=500, r=0.05)
    iv_df[['strike', 'maturity', 'implied_vol']].dropna()

    # From a custom DataFrame
    import pandas as pd
    data = pd.DataFrame({
        'strike':      [95, 100, 105],
        'maturity':    [1.0, 1.0, 1.0],
        'option_type': ['call', 'call', 'call'],
        'price':       [10.5, 7.2, 4.8],
    })
    iv_df = sq.compute_iv_dataframe(data, S=100, r=0.05)
    """
    required = [price_col, strike_col, maturity_col]
    missing  = [c for c in required if c not in prices_df.columns]
    if missing:
        raise ValueError(f"Missing columns in prices_df: {missing}")

    df = prices_df.copy()

    # Resolve option type per row
    if type_col in df.columns:
        types = df[type_col].str.lower().str.strip()
    else:
        types = pd.Series(default_type, index=df.index)

    # Pre-compute forward price and derived columns
    T_arr  = df[maturity_col].values.astype(float)
    K_arr  = df[strike_col].values.astype(float)
    F_arr  = S * np.exp(r * T_arr)

    df['moneyness'] = np.log(K_arr / F_arr)
    df['intrinsic'] = np.where(
        types == 'call',
        np.maximum(S - K_arr * np.exp(-r * T_arr), 0),
        np.maximum(K_arr * np.exp(-r * T_arr) - S, 0)
    )

    ivs    = np.full(len(df), np.nan)
    errors = np.full(len(df), None, dtype=object)

    for i, (idx, row) in enumerate(df.iterrows()):
        if verbose and i > 0 and i % 100 == 0:
            print(f"  [{i}/{len(df)}] computing IVs...")

        price = float(row[price_col])
        K     = float(row[strike_col])
        T     = float(row[maturity_col])
        otype = types.iloc[i]

        if otype not in ('call', 'put'):
            errors[i] = f"unknown option_type: '{otype}'"
            continue
        if T <= 0:
            errors[i] = 'T <= 0'
            continue
        if price < min_price:
            errors[i] = f'price {price:.4f} < min_price {min_price}'
            continue

        try:
            ivs[i] = implied_volatility(price, S, K, T, r, otype)
        except (ValueError, RuntimeError) as e:
            errors[i] = str(e)[:80]

    df['implied_vol'] = ivs
    df['iv_error']    = errors

    n_ok  = np.sum(np.isfinite(ivs))
    n_tot = len(df)
    if verbose:
        print(f"Done: {n_ok}/{n_tot} IVs computed successfully "
              f"({n_tot - n_ok} failed).")

    return df


# ── D. Rolling historical IV DataFrame ────────────────────────────────────────────

def rolling_iv_dataframe(
    option_prices: pd.DataFrame,
    S_series:      pd.Series,
    r:             float,
    price_col:     str   = 'price',
    strike_col:    str   = 'strike',
    maturity_col:  str   = 'maturity',
    type_col:      str   = 'option_type',
    default_type:  str   = 'call',
    min_price:     float = 0.01,
    annualize:     bool  = True,
    verbose:       bool  = False,
) -> pd.DataFrame:
    """
    Compute rolling implied volatility over time from a historical option price series.

    For each date in option_prices, uses the corresponding spot price from S_series
    to invert Black-Scholes and recover IV. Returns a time-indexed DataFrame of IVs —
    one column per unique (strike, maturity, type) combination.

    This lets you track how the IV of a specific contract evolved over time,
    compare IV dynamics across strikes, or compute IV term structure roll.

    Parameters:
    -----------
    option_prices : pd.DataFrame — historical option prices with a DatetimeIndex.
                                   Must contain: price_col, strike_col, maturity_col.
                                   Can contain multiple rows per date (different strikes/maturities).
    S_series      : pd.Series   — historical spot prices, DatetimeIndex aligned with option_prices.
    r             : float       — risk-free rate (assumed constant; pass a series for term-structure)
    price_col     : str         — price column name (default: 'price')
    strike_col    : str         — strike column name (default: 'strike')
    maturity_col  : str         — maturity in years (default: 'maturity')
    type_col      : str         — option type column (default: 'option_type')
    default_type  : str         — fallback option type if type_col missing (default: 'call')
    min_price     : float       — minimum price to attempt IV computation (default: 0.01)
    annualize     : bool        — annualize IV with sqrt(252) if input is daily (default: True)
    verbose       : bool        — print date-level progress (default: False)

    Returns:
    --------
    pd.DataFrame — DatetimeIndex, one column per (K, T, type) contract.
                   Column names formatted as 'IV_K{strike}_T{maturity}_{type}'.
                   Values are implied vols in decimal form. NaN where computation failed.

    Example:
    --------
    # Build historical option prices DataFrame
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
    hist_prices = pd.DataFrame({
        'date':        np.tile(dates, 3),
        'strike':      np.repeat([95, 100, 105], len(dates)),
        'maturity':    0.25,
        'option_type': 'call',
        'price':       np.random.uniform(2, 15, len(dates) * 3),
    }).set_index('date')

    spot = pd.Series(100 + np.random.randn(len(dates)).cumsum(),
                     index=dates, name='spot')

    iv_df = sq.rolling_iv_dataframe(hist_prices, spot, r=0.05)
    iv_df.plot(title='Rolling IV by Strike')

    Notes:
    ------
    - maturity_col should represent *remaining* time to expiry at each date.
      If you have a fixed expiry date, compute T = (expiry - date).days / 365 per row.
    - For a single contract over time, pass a DataFrame with one row per date.
    - For multiple contracts, pass multiple rows per date — they'll become separate columns.
    """
    required = [price_col, strike_col, maturity_col]
    missing  = [c for c in required if c not in option_prices.columns]
    if missing:
        raise ValueError(f"Missing columns in option_prices: {missing}")

    if not isinstance(option_prices.index, pd.DatetimeIndex):
        raise ValueError("option_prices must have a DatetimeIndex.")
    if not isinstance(S_series.index, pd.DatetimeIndex):
        raise ValueError("S_series must have a DatetimeIndex.")

    # Identify unique contracts → columns
    # Key on (strike, type) only — maturity varies over time as expiry approaches
    has_type_col = type_col in option_prices.columns

    if has_type_col:
        contract_keys = option_prices[[strike_col, type_col]].drop_duplicates()
        col_names = [
            f"IV_K{row[strike_col]:.0f}_{str(row[type_col]).lower()}"
            for _, row in contract_keys.iterrows()
        ]
    else:
        contract_keys = option_prices[[strike_col]].drop_duplicates()
        col_names = [
            f"IV_K{row[strike_col]:.0f}_{default_type}"
            for _, row in contract_keys.iterrows()
        ]

    dates   = option_prices.index.unique().sort_values()
    results = pd.DataFrame(index=dates, columns=col_names, dtype=float)

    for date in dates:
        if verbose:
            print(f"  {date.date()} ...")

        # Get spot for this date
        if date not in S_series.index:
            # Try nearest available date
            available = S_series.index[S_series.index <= date]
            if available.empty:
                continue
            S_t = float(S_series.loc[available[-1]])
        else:
            S_t = float(S_series.loc[date])

        rows_today = option_prices.loc[[date]] if date in option_prices.index else None
        if rows_today is None or rows_today.empty:
            continue

        for _, row in rows_today.iterrows():
            price = float(row[price_col])
            K     = float(row[strike_col])
            T     = float(row[maturity_col])
            otype = str(row[type_col]).lower() if has_type_col else default_type

            col = f"IV_K{K:.0f}_{otype}"
            if col not in results.columns:
                continue
            if T <= 0 or price < min_price:
                continue

            try:
                iv = implied_volatility(price, S_t, K, T, r, otype)
                results.loc[date, col] = iv
            except (ValueError, RuntimeError):
                pass

    return results.astype(float)


# ── Historical volatility ─────────────────────────────────────────────────────────

def historical_volatility(prices, window=252, verbose=False):
    """
    Annualized historical volatility computed on the last `window` observations.

    Parameters:
    -----------
    prices  : pd.Series — price series
    window  : int       — lookback period in trading days (default: 252)
    verbose : bool      — print result (default: False)

    Returns:
    --------
    float : annualized historical volatility
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < window:
        raise ValueError(
            f"Not enough observations: got {len(log_returns)}, need at least {window}."
        )

    hist_vol = log_returns.iloc[-window:].std(ddof=1) * np.sqrt(252)

    if verbose:
        name = prices.name if hasattr(prices, 'name') and prices.name else 'series'
        print(f"Historical Volatility ({name}, window={window}d): "
              f"{hist_vol:.6f} ({hist_vol*100:.2f}%)")

    return hist_vol


# ── Realized volatility ───────────────────────────────────────────────────────────

def realized_volatility(prices, window=252, verbose=False):
    """
    Annualized realized volatility over the last `window` observations.

    Parameters:
    -----------
    prices  : pd.Series — price series
    window  : int       — lookback period (default: 252)
    verbose : bool      — print result (default: False)

    Returns:
    --------
    float : annualized realized volatility
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < window:
        raise ValueError(
            f"Not enough observations: got {len(log_returns)}, need at least {window}."
        )

    real_vol = log_returns.iloc[-window:].std(ddof=1) * np.sqrt(252)

    if verbose:
        name = prices.name if hasattr(prices, 'name') and prices.name else 'series'
        print(f"Realized Volatility ({name}, window={window}d): "
              f"{real_vol:.6f} ({real_vol*100:.2f}%)")

    return real_vol
