"""Hold-out evaluation figure and the final seven-day forecast."""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from . import config
from .arima_model import fit

USD = StrMethodFormatter("${x:,.0f}")


def static_forecast(price, order, trend, train_size) -> pd.DataFrame:
    """Single forecast from the end of training over the whole test period."""
    result = fit(np.log(price.iloc[:train_size]), order, trend)
    steps = len(price) - train_size
    forecast = result.get_forecast(steps)
    bounds = np.exp(forecast.conf_int(alpha=config.ALPHA))
    return pd.DataFrame({"date": price.index[train_size:], "horizon": range(1, steps + 1),
                         "origin_price": price.iloc[train_size - 1],
                         "actual": price.iloc[train_size:].values,
                         "predicted": np.exp(forecast.predicted_mean).values,
                         "lower_95": bounds.iloc[:, 0].values, "upper_95": bounds.iloc[:, 1].values})


def plot_test_predictions(price, train_size, one_step, static, model_name) -> None:
    config.setup_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, frame, title in ((axes[0], one_step, "One-step-ahead walk-forward"),
                             (axes[1], static, "Static multi-step forecast from end of training")):
        context = price.iloc[train_size - 45:train_size]
        ax.plot(context, color="grey", label="Training (last 45 days)")
        ax.plot(frame.date, frame.actual, color="black", label="Actual (test)")
        ax.plot(frame.date, frame.predicted, color="darkorange", ls="--", label="Predicted")
        ax.fill_between(frame.date, frame.lower_95, frame.upper_95, color="darkorange",
                        alpha=0.18, label="95% prediction interval")
        ax.axvline(price.index[train_size], color="steelblue", ls=":", label="Test start")
        ax.set(title=f"{title}: {model_name}", ylabel="Price (USD)", xlabel="Date")
        ax.yaxis.set_major_formatter(USD)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "test_predictions.png")
    plt.close(fig)


def direction(change_pct: float, threshold: float) -> str:
    if abs(change_pct) < threshold:
        return "Neutral"
    return "Up" if change_pct > 0 else "Down"


def _plot_forecast(price, dates, point, bounds, model_name) -> None:
    config.setup_plotting()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5))
    for ax, history, title in ((axes[0], price, "Full history and seven-day forecast"),
                               (axes[1], price.iloc[-60:], "Last 60 observations and forecast")):
        ax.plot(history, color="black", lw=1.2, label="Historical price")
        ax.plot(dates, point, "o--", color="darkorange", ms=4, label="Forecast (median)")
        ax.fill_between(dates, bounds[:, 0], bounds[:, 1], color="darkorange", alpha=0.22,
                        label="95% prediction interval")
        ax.axvline(price.index[-1], color="steelblue", ls=":", label="Forecast origin")
        ax.set(title=f"{title}: {model_name}", ylabel="Price (USD)", xlabel="Date")
        ax.yaxis.set_major_formatter(USD)
        ax.legend(fontsize=8)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "seven_day_forecast.png")
    plt.close(fig)


def seven_day_forecast(price, order, trend, model_name, save_as="seven_day_forecast.csv",
                       make_plot=True):
    """Refit on all observations and forecast the next seven calendar days."""
    result = fit(np.log(price), order, trend)
    forecast = result.get_forecast(config.HORIZON)
    dates = pd.date_range(price.index[-1] + pd.Timedelta(days=1), periods=config.HORIZON, freq="D")
    point = np.exp(forecast.predicted_mean.to_numpy())
    bounds = np.exp(forecast.conf_int(alpha=config.ALPHA).to_numpy())
    previous = np.concatenate([[price.iloc[-1]], point[:-1]])
    daily = (point / previous - 1) * 100
    daily[np.abs(daily) < 1e-8] = 0.0
    cumulative = (point / price.iloc[-1] - 1) * 100
    cumulative[np.abs(cumulative) < 1e-8] = 0.0
    table = pd.DataFrame({
        "forecast_date": dates.strftime("%Y-%m-%d"), "horizon_days": range(1, config.HORIZON + 1),
        "predicted_price_usd": point, "lower_95_usd": bounds[:, 0], "upper_95_usd": bounds[:, 1],
        "expected_daily_change_pct": daily, "cumulative_change_pct": cumulative,
        "direction": [direction(v, config.DAILY_NEUTRAL_PCT) for v in daily]})
    table.round(4).to_csv(config.RESULTS_DIR / save_as, index=False)
    if make_plot:
        _plot_forecast(price, dates, point, bounds, model_name)
    final_change = cumulative[-1]
    width = bounds[:, 1] - bounds[:, 0]
    summary = {
        "origin_date": str(price.index[-1].date()), "origin_price": price.iloc[-1],
        "end_price": point[-1], "change_pct": final_change,
        "classification": {"Up": "Bullish", "Down": "Bearish", "Neutral": "Neutral"}[
            direction(final_change, config.WEEKLY_NEUTRAL_PCT)],
        "width_day1": width[0], "width_day7": width[-1],
        "width_day7_pct": width[-1] / point[-1] * 100,
        "lower_day7": bounds[-1, 0], "upper_day7": bounds[-1, 1],
        "mean_adjusted_day7": float(np.exp(forecast.predicted_mean.iloc[-1]
                                           + np.asarray(forecast.var_pred_mean)[-1] / 2)),
        "log_se_day7": float(forecast.se_mean.iloc[-1]),
    }
    return result, table, summary
