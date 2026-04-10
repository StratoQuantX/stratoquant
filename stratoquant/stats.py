import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from statsmodels.tsa.stattools import coint, adfuller, kpss, grangercausalitytests, acf  # type: ignore
from arch.unitroot import PhillipsPerron
from scipy.stats import jarque_bera, shapiro, probplot  # type: ignore


def cointegration_test(y1, y2, alpha=0.05, verbose=True):
    """
    Perform the Engle-Granger cointegration test between two time series.

    Parameters:
    y1 : pd.Series or np.array — first time series
    y2 : pd.Series or np.array — second time series
    alpha : float — significance level (default: 0.05)
    verbose : bool — if True, prints results and interpretation

    Returns:
    result : dict with test_statistic, p_value, critical_values, cointegrated (bool)
    """
    stat, pvalue, crit_vals = coint(y1, y2)
    is_cointegrated = pvalue < alpha

    result = {
        'test_statistic': stat,
        'p_value': pvalue,
        'critical_values': {
            '1%': crit_vals[0],
            '5%': crit_vals[1],
            '10%': crit_vals[2]
        },
        'cointegrated': is_cointegrated
    }

    if verbose:
        print(f"Test statistic     : {stat:.4f}")
        print(f"p-value            : {pvalue:.4f}")
        print("Critical values    :", {k: f"{v:.4f}" for k, v in result['critical_values'].items()})
        if is_cointegrated:
            print(f"✅ The series are cointegrated at the {int(alpha*100)}% level.")
        else:
            print(f"❌ The series are not cointegrated at the {int(alpha*100)}% level.")

    return result


def adf_test(y, alpha=0.05, verbose=True):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity of a time series.

    Parameters:
    y : array-like — time series data
    alpha : float — significance level (default is 0.05)
    verbose : bool — whether to print the results (default is True)

    Returns:
    result : dict — test_statistic, p_value, critical_values, is_stationary
    df_result : pd.DataFrame — formatted result table
    """
    adf_result = adfuller(y)
    test_stat = adf_result[0]
    p_value = adf_result[1]
    lags = adf_result[2]
    n_obs = adf_result[3]
    crit_vals = adf_result[4]

    is_stationary = p_value < alpha

    # Format in a DataFrame
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
        'p_value': p_value,
        'critical_values': crit_vals,
        'is_stationary': is_stationary
    }, df_result

def kpss_test(y, alpha=0.05, verbose=True):
    """[Documentation for kpss_test]"""
    
    # Perform KPSS test
    kpss_result = kpss(y)
    test_stat = kpss_result[0]
    p_value = kpss_result[1]
    lags = kpss_result[2]
    crit_vals = kpss_result[3]
    is_stationary = p_value > alpha

    # Format in a DataFrame
    df_result = pd.DataFrame({
        'Valeur': [
            f"{test_stat:.4f}",
            f"{p_value:.4f}",
            lags,
            f"{crit_vals['10%']:.4f}",
        ]
        }, index=[
        "Statistique KPSS",
        "p-valeur",
        "Retards utilisés",
        "Critique 10%",
        "Conclusion"
    ])
    df_result.loc['Conclusion'] = "✅ Stationnaire" if is_stationary else "❌ Non stationnaire"
    if verbose:
        print(df_result)
    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'critical_values': crit_vals,
        'is_stationary': is_stationary
    }, df_result

def pp_test(y, alpha=0.05, verbose=True):
    """[Documentation for pp_test]"""
    # Perform PP test
    pp_result = PhillipsPerron(y)
    test_stat = pp_result.stat
    p_value = pp_result.pvalue
    lags = pp_result.lags
    crit_vals = pp_result.critical_values

    is_stationary = p_value < alpha
    if verbose:
        print(f"Statistique PP: {test_stat:.4f}")
        print(f"p-valeur: {p_value:.4f}")
        print(f"Retards utilisés: {lags}")
        print(f"Critique 10%: {crit_vals['10%']:.4f}")
        print(f"Conclusion: {'✅ Stationary' if is_stationary else '❌ No stationary'}")
    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'critical_values': crit_vals,
        'is_stationary': is_stationary
    }

def granger_causality_test(data, max_lags=10, alpha=0.05, verbose=True):
    """
    Test de causalité de Granger : est-ce que x "cause" y ?
    
    Paramètres :
        y (pd.Series): série dépendante
        x (pd.Series): série explicative
        max_lags (int): nombre maximal de retards à tester
        alpha (float): seuil de significativité
        verbose (bool): afficher les résultats
        
    Retourne :
        dict: lag optimal, p-valeur, conclusion de causalité
    """

    # Effectuer le test
    granger_result = grangercausalitytests(data, maxlag=max_lags, verbose=False)

    # Trouver le meilleur lag selon la plus petite p-value
    best_lag = min(granger_result, key=lambda k: granger_result[k][0]['ssr_ftest'][1])
    p_value = granger_result[best_lag][0]['ssr_ftest'][1]
    is_causal = p_value < alpha

    if verbose:
        print(f"Lag optimal: {best_lag}")
        print(f"p-valeur: {p_value:.4f}")
        print(f"Conclusion: {'✅ Causal' if is_causal else '❌ No causal'}")

    return {
        'best_lag': best_lag,
        'p_value': p_value,
        'is_causal': is_causal
    }

def jarque_bera_test(y, alpha=0.05, verbose=True, plot=True, title_prefix=""):
    """
    Performs the Jarque-Bera test and optionally displays histogram and QQ plot.

    Parameters:
    - y : array-like, the data to test
    - alpha : significance level for the test
    - verbose : bool, whether to print test results
    - plot : bool, whether to display the plots
    - title_prefix : str, prefix for the plot titles

    Returns:
    - dict : containing jb_stat, jb_p_value, and is_normal
    """
    # Perform Jarque-Bera test
    jb_stat, jb_p_value = jarque_bera(y)
    is_normal = jb_p_value > alpha

    # Console output
    if verbose:
        print(f"Jarque-Bera Statistic: {jb_stat:.4f}")
        print(f"p-value: {jb_p_value:.4f}")
        print(f"Conclusion: {'✅ Normally distributed' if is_normal else '❌ Not normally distributed'}")

    # Plotting
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))

        # Histogram with KDE
        sns.histplot(y, bins=30, kde=True, color='steelblue', ax=axes[0])
        axes[0].set_title(f"{title_prefix}Distribution Histogram")
        axes[0].set_xlabel("Values")
        axes[0].set_ylabel("Frequency")
        axes[0].text(0.95, 0.95, 
                     f"JB stat: {jb_stat:.2f}\n"
                     f"p-value: {jb_p_value:.4f}\n"
                     f"{'✅ Normal' if is_normal else '❌ Not normal'}",
                     transform=axes[0].transAxes,
                     fontsize=12,
                     verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # QQ plot
        probplot(y, dist="norm", plot=axes[1])
        axes[1].set_title(f"{title_prefix}QQ Plot")

        plt.tight_layout()
        plt.show()

    return {
        'jb_stat': jb_stat,
        'jb_p_value': jb_p_value,
        'is_normal': is_normal
    }



def shapiro_wilk_test(y, alpha=0.05, verbose=True):
    """[Documentation for shapiro_wilk_test]"""
    # Perform Shapiro-Wilk test
    sw_stat, sw_p_value = shapiro(y)
    is_normal = sw_p_value > alpha
    if verbose:
        print(f"Statistique SW: {sw_stat:.4f}")
        print(f"p-valeur: {sw_p_value:.4f}")
        print(f"Conclusion: {'✅ Normal' if is_normal else '❌ No normal'}")
    return {
        'sw_stat': sw_stat,
        'sw_p_value': sw_p_value,
        'is_normal': is_normal
    }
# End of file stats.py