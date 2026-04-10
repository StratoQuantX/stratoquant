import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from statsmodels.tsa.stattools import coint, adfuller, kpss, grangercausalitytests, acf  # type: ignore
from arch.unitroot import PhillipsPerron
from scipy.stats import jarque_bera, shapiro, probplot  # type: ignore


# ── Cointegration ────────────────────────────────────────────────────────────────

def cointegration_test(y1, y2, alpha=0.05, verbose=True):
    """
    Engle-Granger cointegration test between two time series.

    H0 : no cointegration (series are not cointegrated).
    Reject H0 (i.e. series ARE cointegrated) when p_value < alpha.

    Parameters:
    -----------
    y1      : array-like — first time series
    y2      : array-like — second time series
    alpha   : float      — significance level (default: 0.05)
    verbose : bool       — print results if True

    Returns:
    --------
    dict : test_statistic, p_value, critical_values, cointegrated (bool)
    """
    stat, pvalue, crit_vals = coint(y1, y2)
    is_cointegrated = pvalue < alpha

    result = {
        'test_statistic': stat,
        'p_value': pvalue,
        'critical_values': {
            '1%':  crit_vals[0],
            '5%':  crit_vals[1],
            '10%': crit_vals[2]
        },
        'cointegrated': is_cointegrated
    }

    if verbose:
        print(f"Test statistic  : {stat:.4f}")
        print(f"p-value         : {pvalue:.4f}")
        print("Critical values :", {k: f"{v:.4f}" for k, v in result['critical_values'].items()})
        if is_cointegrated:
            print(f"✅ Les séries sont cointégrées au niveau {int(alpha*100)}%.")
        else:
            print(f"❌ Les séries ne sont pas cointégrées au niveau {int(alpha*100)}%.")

    return result


# ── Unit root tests ──────────────────────────────────────────────────────────────

def adf_test(y, alpha=0.05, verbose=True):
    """
    Augmented Dickey-Fuller test for stationarity.

    H0 : unit root (series is non-stationary).
    Reject H0 (i.e. series IS stationary) when p_value < alpha.

    Parameters:
    -----------
    y       : array-like — time series data
    alpha   : float      — significance level (default: 0.05)
    verbose : bool       — print formatted table if True

    Returns:
    --------
    (dict, pd.DataFrame) : result dict and formatted table
        dict keys: test_statistic, p_value, critical_values, is_stationary
    """
    adf_result = adfuller(y)
    test_stat = adf_result[0]
    p_value   = adf_result[1]
    lags      = adf_result[2]
    n_obs     = adf_result[3]
    crit_vals = adf_result[4]

    is_stationary = p_value < alpha

    df_result = pd.DataFrame({
        'Valeur': [
            f"{test_stat:.4f}",
            f"{p_value:.4f}",
            lags,
            n_obs,
            f"{crit_vals['1%']:.4f}",
            f"{crit_vals['5%']:.4f}",
            f"{crit_vals['10%']:.4f}",
            "✅ Stationnaire" if is_stationary else "❌ Non stationnaire"
        ]
    }, index=[
        "ADF Stat",
        "p-value",
        "Lags Used",
        "Observations",
        "Critique 1%",
        "Critique 5%",
        "Critique 10%",
        "Conclusion"
    ])

    if verbose:
        print(df_result)

    return {
        'test_statistic': test_stat,
        'p_value':        p_value,
        'critical_values': crit_vals,
        'is_stationary':  is_stationary
    }, df_result


def kpss_test(y, alpha=0.05, verbose=True):
    """
    KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test for stationarity.

    H0 : series IS stationary (opposite of ADF).
    Reject H0 (i.e. series is NON-stationary) when p_value < alpha.

    Note: use ADF and KPSS together for robust conclusions —
    they have opposite null hypotheses, so agreement strengthens inference.

    Parameters:
    -----------
    y       : array-like — time series data
    alpha   : float      — significance level (default: 0.05)
    verbose : bool       — print formatted table if True

    Returns:
    --------
    (dict, pd.DataFrame) : result dict and formatted table
        dict keys: test_statistic, p_value, critical_values, is_stationary
    """
    kpss_result = kpss(y, nlags='auto')
    test_stat = kpss_result[0]
    p_value   = kpss_result[1]
    lags      = kpss_result[2]
    crit_vals = kpss_result[3]

    # KPSS: H0 = stationary → reject H0 when p_value < alpha → non-stationary
    is_stationary = p_value > alpha

    # Build DataFrame with all values up front — avoids shape mismatch crash
    df_result = pd.DataFrame({
        'Valeur': [
            f"{test_stat:.4f}",
            f"{p_value:.4f}",
            lags,
            f"{crit_vals['10%']:.4f}",
            f"{crit_vals['5%']:.4f}",
            f"{crit_vals['2.5%']:.4f}",
            f"{crit_vals['1%']:.4f}",
            "✅ Stationnaire" if is_stationary else "❌ Non stationnaire"
        ]
    }, index=[
        "Statistique KPSS",
        "p-valeur",
        "Retards utilisés",
        "Critique 10%",
        "Critique 5%",
        "Critique 2.5%",
        "Critique 1%",
        "Conclusion"
    ])

    if verbose:
        print(df_result)

    return {
        'test_statistic':  test_stat,
        'p_value':         p_value,
        'critical_values': crit_vals,
        'is_stationary':   is_stationary
    }, df_result


def pp_test(y, alpha=0.05, verbose=True):
    """
    Phillips-Perron test for stationarity.

    H0 : unit root (series is non-stationary), same direction as ADF.
    Reject H0 (i.e. series IS stationary) when p_value < alpha.

    More robust than ADF to heteroskedasticity and serial correlation
    (uses non-parametric correction rather than additional lags).

    Parameters:
    -----------
    y       : array-like — time series data
    alpha   : float      — significance level (default: 0.05)
    verbose : bool       — print formatted table if True

    Returns:
    --------
    (dict, pd.DataFrame) : result dict and formatted table
        dict keys: test_statistic, p_value, critical_values, lags, is_stationary
    """
    pp_result = PhillipsPerron(y)
    test_stat = pp_result.stat
    p_value   = pp_result.pvalue
    lags      = pp_result.lags
    crit_vals = pp_result.critical_values

    is_stationary = p_value < alpha

    df_result = pd.DataFrame({
        'Valeur': [
            f"{test_stat:.4f}",
            f"{p_value:.4f}",
            lags,
            f"{crit_vals['10%']:.4f}",
            f"{crit_vals['5%']:.4f}",
            f"{crit_vals['1%']:.4f}",
            "✅ Stationnaire" if is_stationary else "❌ Non stationnaire"
        ]
    }, index=[
        "Statistique PP",
        "p-valeur",
        "Retards utilisés",
        "Critique 10%",
        "Critique 5%",
        "Critique 1%",
        "Conclusion"
    ])

    if verbose:
        print(df_result)

    return {
        'test_statistic':  test_stat,
        'p_value':         p_value,
        'critical_values': crit_vals,
        'lags':            lags,
        'is_stationary':   is_stationary
    }, df_result


# ── Causality ────────────────────────────────────────────────────────────────────

def granger_causality_test(data, max_lags=10, alpha=0.05, verbose=True):
    """
    Granger causality test: does the first column help predict the second?

    Runs the F-test (ssr_ftest) for each lag up to max_lags and identifies
    the lag with the lowest p-value as the "best" lag.

    Parameters:
    -----------
    data     : pd.DataFrame — two-column DataFrame [y, x], where x is tested
                              as a Granger cause of y
    max_lags : int          — maximum number of lags to test (default: 10)
    alpha    : float        — significance level (default: 0.05)
    verbose  : bool         — print results if True

    Returns:
    --------
    dict : best_lag, p_value, is_causal, all_pvalues (dict of lag → p-value)

    Note:
    -----
    Granger causality ≠ true causality. It only tests predictive content.
    Both series must be stationary before running this test.
    """
    granger_result = grangercausalitytests(data, maxlag=max_lags, verbose=False)

    all_pvalues = {
        lag: granger_result[lag][0]['ssr_ftest'][1]
        for lag in granger_result
    }
    best_lag = min(all_pvalues, key=all_pvalues.get)
    p_value  = all_pvalues[best_lag]
    is_causal = p_value < alpha

    if verbose:
        print(f"Lag optimal : {best_lag}")
        print(f"p-valeur    : {p_value:.4f}")
        print(f"Conclusion  : {'✅ Causalité de Granger détectée' if is_causal else '❌ Pas de causalité de Granger'}")
        print("\nToutes les p-valeurs par lag :")
        for lag, pv in all_pvalues.items():
            marker = " ← best" if lag == best_lag else ""
            print(f"  lag {lag:>2} : {pv:.4f}{marker}")

    return {
        'best_lag':    best_lag,
        'p_value':     p_value,
        'is_causal':   is_causal,
        'all_pvalues': all_pvalues
    }


# ── Normality tests ──────────────────────────────────────────────────────────────

def jarque_bera_test(y, alpha=0.05, verbose=True, plot=True, title_prefix=""):
    """
    Jarque-Bera test for normality, with optional histogram and QQ plot.

    H0 : data follows a normal distribution.
    Reject H0 when p_value < alpha.

    Particularly suited for large samples (n > 2000). For small samples,
    prefer shapiro_wilk_test which has better power.

    Parameters:
    -----------
    y            : array-like — data to test
    alpha        : float      — significance level (default: 0.05)
    verbose      : bool       — print results if True
    plot         : bool       — display histogram + QQ plot if True
    title_prefix : str        — prefix for plot titles

    Returns:
    --------
    dict : jb_stat, jb_p_value, is_normal
    """
    jb_stat, jb_p_value = jarque_bera(y)
    is_normal = jb_p_value > alpha

    if verbose:
        print(f"Jarque-Bera Stat : {jb_stat:.4f}")
        print(f"p-value          : {jb_p_value:.4f}")
        print(f"Conclusion       : {'✅ Distribution normale' if is_normal else '❌ Distribution non normale'}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        sns.histplot(y, bins=30, kde=True, color='steelblue', ax=axes[0])
        axes[0].set_title(f"{title_prefix}Distribution Histogram")
        axes[0].set_xlabel("Values")
        axes[0].set_ylabel("Frequency")
        axes[0].text(
            0.95, 0.95,
            f"JB stat: {jb_stat:.2f}\np-value: {jb_p_value:.4f}\n"
            f"{'✅ Normal' if is_normal else '❌ Not normal'}",
            transform=axes[0].transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        probplot(y, dist="norm", plot=axes[1])
        axes[1].set_title(f"{title_prefix}QQ Plot")

        plt.tight_layout()
        plt.show()

    return {
        'jb_stat':    jb_stat,
        'jb_p_value': jb_p_value,
        'is_normal':  is_normal
    }


def shapiro_wilk_test(y, alpha=0.05, verbose=True):
    """
    Shapiro-Wilk test for normality.

    H0 : data follows a normal distribution.
    Reject H0 when p_value < alpha.

    More powerful than Jarque-Bera for small samples (n < 2000).
    Not recommended for n > 5000 (becomes overly sensitive to minor deviations).

    Parameters:
    -----------
    y       : array-like — data to test
    alpha   : float      — significance level (default: 0.05)
    verbose : bool       — print results if True

    Returns:
    --------
    (dict, pd.DataFrame) : result dict and formatted table
        dict keys: sw_stat, sw_p_value, is_normal
    """
    sw_stat, sw_p_value = shapiro(y)
    is_normal = sw_p_value > alpha

    df_result = pd.DataFrame({
        'Valeur': [
            f"{sw_stat:.4f}",
            f"{sw_p_value:.4f}",
            "✅ Distribution normale" if is_normal else "❌ Distribution non normale"
        ]
    }, index=[
        "Statistique SW",
        "p-valeur",
        "Conclusion"
    ])

    if verbose:
        print(df_result)

    return {
        'sw_stat':    sw_stat,
        'sw_p_value': sw_p_value,
        'is_normal':  is_normal
    }, df_result
