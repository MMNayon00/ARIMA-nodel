"""Candidate search, model selection, walk-forward evaluation and diagnostics.

All models are fitted to the natural logarithm of price. Forecasts are
exponentiated, so a USD point forecast is the conditional MEDIAN.
"""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from joblib import Parallel, delayed
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.arima.model import ARIMA

from . import config


def label(order, trend) -> str:
    return f"ARIMA{tuple(order)}" + (" + drift" if trend == "t" else "")


def fit(log_series: pd.Series, order, trend="n"):
    """Fit by exact maximum likelihood; convergence is checked, not assumed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(log_series, order=order, trend=trend).fit(method_kwargs={"maxiter": 500})


def converged(result) -> bool:
    return bool((result.mle_retvals or {}).get("converged", True))


def min_root(result) -> float:
    """Smallest AR/MA root modulus (infinity when the model has no AR or MA terms)."""
    roots = np.concatenate([np.atleast_1d(result.arroots), np.atleast_1d(result.maroots)])
    return float(np.abs(roots).min()) if roots.size else float("inf")


def ljung_box(result, order) -> dict:
    p, d, q = order
    residuals = result.resid.iloc[max(d, 1):]
    lags = [lag for lag in config.LB_LAGS if lag > p + q]
    table = acorr_ljungbox(residuals, lags=lags, model_df=p + q)
    return {int(lag): {"Q": row.lb_stat, "p_value": row.lb_pvalue} for lag, row in table.iterrows()}


def grid_search(train_log: pd.Series, d: int) -> pd.DataFrame:
    rows = []
    trends = ("n", "t") if d == 1 else ("n",)
    for p in range(config.MAX_P + 1):
        for q in range(config.MAX_Q + 1):
            for trend in trends:
                try:
                    result = fit(train_log, (p, d, q), trend)
                    tests = ljung_box(result, (p, d, q))
                    rows.append({"p": p, "d": d, "q": q, "trend": trend,
                                 "model": label((p, d, q), trend),
                                 "aic": result.aic, "bic": result.bic,
                                 "converged": converged(result),
                                 "min_root_modulus": min_root(result),
                                 "min_ljung_box_p": min(t["p_value"] for t in tests.values())})
                except (ValueError, np.linalg.LinAlgError) as error:
                    print(f"Skipped ARIMA({p},{d},{q}) trend={trend}: {error}")
    table = pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)
    table.to_csv(config.RESULTS_DIR / "arima_grid_search.csv", index=False)
    return table


def auto_arima_suggestion(train_log: pd.Series, d: int):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pm.auto_arima(train_log.values, d=d, max_p=config.MAX_P, max_q=config.MAX_Q,
                              seasonal=False, stepwise=True, information_criterion="aic",
                              suppress_warnings=True, error_action="ignore")
    trend = "t" if model.with_intercept and d == 1 else "n"
    return tuple(int(v) for v in model.order), trend


def walk_forward(price: pd.Series, order, trend, start: int, stop: int, horizon: int = 1):
    """Expanding-window rolling origin: refit at every origin using past data only.

    Origins run from start-1 to stop-1-horizon, so every target lies in [start, stop).
    """
    log_price = np.log(price)

    def one_origin(end):
        history = log_price.iloc[:end]
        result = fit(history, order, trend)
        forecast = result.get_forecast(horizon)
        mean = forecast.predicted_mean
        bounds = forecast.conf_int(alpha=config.ALPHA)
        return [{"origin": history.index[-1], "date": price.index[end + h], "horizon": h + 1,
                 "origin_price": price.iloc[end - 1], "actual": price.iloc[end + h],
                 "predicted": float(np.exp(mean.iloc[h])),
                 "lower_95": float(np.exp(bounds.iloc[h, 0])),
                 "upper_95": float(np.exp(bounds.iloc[h, 1])),
                 "converged": converged(result)} for h in range(horizon)]

    # Origins are independent, so they are fitted in parallel (order is preserved).
    batches = Parallel(n_jobs=-1)(delayed(one_origin)(end)
                                  for end in range(start, stop - horizon + 1))
    return pd.DataFrame([row for batch in batches for row in batch])


def baseline_forecasts(price: pd.Series, start: int, stop: int, horizon: int = 1):
    """Naive (last value), drift (average past log change) and 7-day mean baselines."""
    frames = {"Naive (last value)": [], "Drift": [], "7-day moving average": []}
    for end in range(start, stop - horizon + 1):
        history = price.iloc[:end]
        drift = np.log(history.iloc[-1] / history.iloc[0]) / (len(history) - 1)
        for h in range(horizon):
            common = {"origin": history.index[-1], "date": price.index[end + h], "horizon": h + 1,
                      "origin_price": history.iloc[-1], "actual": price.iloc[end + h]}
            frames["Naive (last value)"].append({**common, "predicted": history.iloc[-1]})
            frames["Drift"].append({**common, "predicted": history.iloc[-1] * np.exp(drift * (h + 1))})
            frames["7-day moving average"].append({**common, "predicted": history.iloc[-7:].mean()})
    return {name: pd.DataFrame(rows) for name, rows in frames.items()}


def metrics(frame: pd.DataFrame) -> dict:
    actual, predicted = frame["actual"].to_numpy(), frame["predicted"].to_numpy()
    reference = frame["origin_price"].to_numpy()
    error = actual - predicted
    predicted_move = predicted - reference
    # A forecast equal to the origin price states no direction, so it is not scored.
    directional = ~np.isclose(predicted_move, 0, rtol=0, atol=1e-6)
    accuracy = (np.mean(np.sign(predicted_move[directional]) ==
                        np.sign((actual - reference)[directional])) * 100
                if directional.any() else np.nan)
    return {"n": len(frame), "MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error ** 2)),
            "MAPE_%": np.mean(np.abs(error) / actual) * 100,
            "sMAPE_%": np.mean(2 * np.abs(error) / (np.abs(actual) + np.abs(predicted))) * 100,
            "Directional_accuracy_%": accuracy, "Directional_calls": int(directional.sum())}


def diebold_mariano(model_frame, baseline_frame) -> dict:
    """One-step DM test on squared errors with the Harvey et al. small-sample factor.

    Negative statistics favour the model; H0 is equal expected loss.
    """
    loss = ((model_frame["actual"] - model_frame["predicted"]) ** 2
            - (baseline_frame["actual"] - baseline_frame["predicted"]) ** 2).to_numpy()
    n = len(loss)
    if np.allclose(loss, 0):
        return {"statistic": None, "p_value": None}
    statistic = loss.mean() / np.sqrt(loss.var(ddof=0) / n) * np.sqrt((n - 1) / n)
    return {"statistic": statistic, "p_value": 2 * stats.t.sf(abs(statistic), df=n - 1)}


def select_model(price, grid, train_size, auto_choice):
    """Pre-declared rule, using training data only.

    1. Shortlist: top 5 by AIC, top 5 by BIC, auto_arima choice, random walk.
    2. Reject non-converged models, Ljung-Box p < 0.05, or any AR/MA root with
       modulus below 1.001 (Hyndman & Khandakar, 2008).  The root screen was added
       after a first run selected a model with MA roots on the unit circle; the
       result without it is retained as a sensitivity analysis.
    3. Compare 7-day rolling-origin RMSE on the last 20% of the training set.
    4. Among models within 1% of the best RMSE choose the fewest parameters, then lowest BIC.
    """
    d = int(grid["d"].iloc[0])
    wanted = {}
    for source, subset in (("AIC top 5", grid.nsmallest(5, "aic")),
                           ("BIC top 5", grid.nsmallest(5, "bic"))):
        for r in subset.itertuples():
            wanted.setdefault(((r.p, r.d, r.q), r.trend), []).append(source)
    wanted.setdefault(auto_choice, []).append("auto_arima")
    wanted.setdefault(((0, d, 0), "n"), []).append("random walk")

    validation_start = train_size - int(round(train_size * config.VALIDATION_FRACTION))
    rows = []
    for (order, trend), sources in wanted.items():
        info = grid[(grid.p == order[0]) & (grid.q == order[2]) & (grid.trend == trend)].iloc[0]
        one = metrics(walk_forward(price, order, trend, validation_start, train_size, 1))
        week = metrics(walk_forward(price, order, trend, validation_start, train_size, config.HORIZON))
        rows.append({"model": label(order, trend), "p": order[0], "d": order[1], "q": order[2],
                     "trend": trend, "shortlisted_by": "; ".join(sources),
                     "n_parameters": order[0] + order[2] + (trend == "t") + 1,
                     "aic": info.aic, "bic": info.bic, "min_ljung_box_p": info.min_ljung_box_p,
                     "converged": info.converged, "min_root_modulus": info.min_root_modulus,
                     "val_RMSE_1step": one["RMSE"],
                     "val_RMSE_7day": week["RMSE"], "val_MAPE_7day": week["MAPE_%"]})
    table = pd.DataFrame(rows)
    table["passes_diagnostics"] = table.converged & (table.min_ljung_box_p >= config.ALPHA)
    table["admissible_roots"] = table.min_root_modulus >= config.MIN_ROOT_MODULUS

    def choose(mask):
        eligible = table[mask]
        if eligible.empty:
            raise RuntimeError("No shortlisted model passes the screen; revise the search space.")
        tied = eligible[eligible.val_RMSE_7day <= eligible.val_RMSE_7day.min()
                        * (1 + config.PARSIMONY_TOLERANCE)]
        return tied.sort_values(["n_parameters", "bic"]).iloc[0]

    winner = choose(table.passes_diagnostics & table.admissible_roots)
    unscreened = choose(table.passes_diagnostics)
    table["selected"] = table.model == winner.model
    table["selected_without_root_screen"] = table.model == unscreened.model
    table = table.sort_values("val_RMSE_7day").reset_index(drop=True)
    table.to_csv(config.RESULTS_DIR / "candidate_comparison.csv", index=False)
    order = (int(winner.p), int(winner.d), int(winner.q))
    return order, winner.trend, table, validation_start


def residual_diagnostics(result, order, filename: str, title: str) -> dict:
    config.setup_plotting()
    residuals = result.resid.iloc[max(order[1], 1):]
    standardised = (residuals - residuals.mean()) / residuals.std()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes[0, 0].plot(residuals, lw=0.9, label="Residual")
    axes[0, 0].axhline(0, color="black", lw=0.7)
    axes[0, 0].set(title="Residuals over time", xlabel="Date", ylabel="Log-price residual")
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].legend()
    axes[0, 1].hist(standardised, bins=35, density=True, alpha=0.7, label="Standardised residuals")
    grid = np.linspace(-5, 5, 300)
    axes[0, 1].plot(grid, stats.norm.pdf(grid), color="firebrick", label="Standard normal")
    axes[0, 1].set(title="Residual distribution", xlabel="Standardised residual", ylabel="Density")
    axes[0, 1].legend()
    stats.probplot(standardised, dist="norm", plot=axes[1, 0])
    axes[1, 0].set(title="Normal Q-Q plot", xlabel="Theoretical quantiles", ylabel="Sample quantiles")
    plot_acf(residuals, lags=30, ax=axes[1, 1], zero=False)
    axes[1, 1].set(title="Residual ACF", xlabel="Lag (days)", ylabel="Autocorrelation", ylim=(-0.3, 0.3))
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / filename)
    plt.close(fig)

    jarque_bera = stats.jarque_bera(residuals)
    roots = {"ar_root_moduli": sorted(np.abs(result.arroots).tolist()),
             "ma_root_moduli": sorted(np.abs(result.maroots).tolist()),
             "closest_ar_ma_root_distance": (float(np.abs(
                 result.arroots[:, None] - result.maroots[None, :]).min())
                 if len(result.arroots) and len(result.maroots) else None),
             "converged": converged(result)}
    arch = het_arch(residuals, nlags=10)
    squared = acorr_ljungbox(residuals ** 2, lags=[10])
    return {**roots, "n": len(residuals), "mean": residuals.mean(), "sd": residuals.std(),
            "mean_t_p": stats.ttest_1samp(residuals, 0).pvalue,
            "skew": stats.skew(residuals), "excess_kurtosis": stats.kurtosis(residuals),
            "ljung_box": ljung_box(result, order),
            "jarque_bera": [jarque_bera.statistic, jarque_bera.pvalue],
            "arch_lm": [arch[0], arch[1]],
            "ljung_box_squared_10": [squared.lb_stat.iloc[0], squared.lb_pvalue.iloc[0]],
            "parameters": {name: [result.params[name], result.bse[name], result.pvalues[name]]
                           for name in result.params.index}}
