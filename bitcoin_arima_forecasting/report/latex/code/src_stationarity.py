"""ADF and KPSS tests, selection of d, and ACF/PACF plots."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config


def adf_test(series: pd.Series, regression: str = "c") -> dict:
    """H0: unit root (non-stationary). H1: stationary."""
    stat, p_value, lags, nobs, critical, _ = adfuller(
        series.dropna(), regression=regression, autolag="AIC")
    return {"statistic": stat, "p_value": p_value, "lags": lags, "nobs": nobs,
            "critical": critical, "stationary": bool(p_value < config.ALPHA)}


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """H0: stationary. H1: unit root. p-values are truncated to [0.01, 0.10]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, p_value, lags, critical = kpss(series.dropna(), regression=regression, nlags="auto")
    return {"statistic": stat, "p_value": p_value, "lags": lags,
            "critical": critical, "stationary": bool(p_value >= config.ALPHA)}


def determine_d(series: pd.Series, max_d: int = 2) -> int:
    """Smallest d whose differenced series rejects the ADF unit-root null."""
    for d in range(max_d + 1):
        candidate = series if d == 0 else series.diff(d).dropna() if d == 1 else series.diff().diff().dropna()
        if adf_test(candidate)["stationary"]:
            return d
    raise ValueError("Series is not stationary after the maximum number of differences.")


def stationarity_table(price: pd.Series, train_size: int) -> pd.DataFrame:
    log_price = np.log(price)
    cases = []
    for sample, stop in (("Training", train_size), ("Full", len(price))):
        for label, series in (("Price", price.iloc[:stop]), ("Log price", log_price.iloc[:stop]),
                              ("Differenced price", price.iloc[:stop].diff().dropna()),
                              ("Differenced log price", log_price.iloc[:stop].diff().dropna())):
            specs = [("c", "constant")]
            if not label.startswith("Diff"):
                specs.append(("ct", "constant + trend"))
            for code, spec in specs:
                for test, function in (("ADF", adf_test), ("KPSS", kpss_test)):
                    r = function(series, code)
                    cases.append({
                        "sample": sample, "series": label, "test": test, "deterministic": spec,
                        "statistic": r["statistic"], "p_value": r["p_value"], "lags": r["lags"],
                        "crit_1pct": r["critical"]["1%"], "crit_5pct": r["critical"]["5%"],
                        "crit_10pct": r["critical"]["10%"],
                        "conclusion_5pct": "Stationary" if r["stationary"] else "Non-stationary"})
    table = pd.DataFrame(cases)
    table.to_csv(config.RESULTS_DIR / "stationarity_tests.csv", index=False)
    return table


def plot_differencing(price: pd.Series) -> None:
    config.setup_plotting()
    log_price = np.log(price)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(log_price, label="Log price")
    axes[0].set(title="Log price (level)", ylabel="log(USD)")
    axes[1].plot(log_price.diff(), color="seagreen", label="First difference of log price")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(title="First-differenced log price", xlabel="Date", ylabel="Log return")
    for ax in axes:
        ax.legend()
    fig.autofmt_xdate()
    fig.savefig(config.OUTPUT_DIR / "differencing.png")
    plt.close(fig)


def acf_pacf_analysis(train_log: pd.Series, d: int, lags: int = 30) -> dict:
    """Plot ACF/PACF of the differenced TRAINING series (no test information)."""
    config.setup_plotting()
    x = train_log
    for _ in range(d):
        x = x.diff().dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    plot_acf(x, lags=lags, ax=axes[0], zero=False)
    plot_pacf(x, lags=lags, ax=axes[1], zero=False, method="ywm")
    axes[0].set(title=f"ACF, training log price (d={d})", xlabel="Lag (days)", ylabel="Autocorrelation")
    axes[1].set(title=f"PACF, training log price (d={d})", xlabel="Lag (days)",
                ylabel="Partial autocorrelation")
    for ax in axes:
        ax.relim()
        ax.set_ylim(-0.3, 0.3)
    fig.savefig(config.OUTPUT_DIR / "acf_pacf.png")
    plt.close(fig)
    a, p = acf(x, nlags=lags), pacf(x, nlags=lags, method="ywm")
    bound = 1.96 / np.sqrt(len(x))
    return {"n": len(x), "bound": bound,
            "significant_acf": [[i, a[i]] for i in range(1, lags + 1) if abs(a[i]) > bound],
            "significant_pacf": [[i, p[i]] for i in range(1, lags + 1) if abs(p[i]) > bound]}
