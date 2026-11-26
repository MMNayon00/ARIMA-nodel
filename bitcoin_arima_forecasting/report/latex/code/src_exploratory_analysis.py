"""Exploratory figures and descriptive statistics."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

from . import config

USD = StrMethodFormatter("${x:,.0f}")


def _save(fig, name):
    fig.autofmt_xdate() if name not in ("price_distribution", "monthly_average_price") else None
    fig.savefig(config.OUTPUT_DIR / f"{name}.png")
    plt.close(fig)


def _line(series, name, title, ylabel, formatter=None, zero=False):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(series.index, series.values, lw=1.2, label=ylabel)
    if zero:
        ax.axhline(0, color="black", lw=0.7)
    if formatter:
        ax.yaxis.set_major_formatter(formatter)
    ax.set(title=title, xlabel="Date", ylabel=ylabel)
    ax.legend()
    _save(fig, name)


def descriptive_statistics(price: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame({"Price (USD)": price, "Daily change (USD)": price.diff(),
                          "Daily return (%)": price.pct_change() * 100})
    stats = table.describe().T
    stats["skewness"], stats["excess kurtosis"] = table.skew(), table.kurt()
    return stats.T


def run_eda(price: pd.Series) -> dict:
    """Create all EDA figures and return the facts quoted in the report."""
    config.setup_plotting()
    returns = price.pct_change().dropna() * 100
    changes = price.diff().dropna()
    rolling_mean, rolling_std = price.rolling(30).mean(), price.rolling(30).std()
    monthly = price.resample("MS").agg(["mean", "count"])

    _line(price, "historical_price", "Bitcoin market price: past year", "Price (USD)", USD)
    _line(returns, "daily_returns", "Daily percentage returns", "Return (%)", zero=True)
    _line(changes, "daily_changes", "Daily absolute price changes", "Change (USD)", USD, True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(price, lw=1, label="Price")
    axes[0].plot(rolling_mean, lw=2, label="30-day rolling mean")
    axes[0].set(title="Price and 30-day rolling mean", ylabel="USD")
    axes[1].plot(rolling_std, color="firebrick", label="30-day rolling standard deviation")
    axes[1].set(title="30-day rolling standard deviation of price", xlabel="Date", ylabel="USD")
    for ax in axes:
        ax.yaxis.set_major_formatter(USD)
        ax.legend()
    _save(fig, "rolling_statistics")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(price, bins=30, kde=True, ax=axes[0])
    axes[0].set(title="Distribution of daily price", xlabel="Price (USD)", ylabel="Days")
    axes[0].xaxis.set_major_formatter(USD)
    axes[0].tick_params(axis="x", rotation=30)
    sns.histplot(returns, bins=35, kde=True, ax=axes[1], color="seagreen")
    axes[1].set(title="Distribution of daily returns", xlabel="Return (%)", ylabel="Days")
    fig.tight_layout()
    _save(fig, "price_distribution")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    labels = [d.strftime("%b %Y") + ("*" if c < d.days_in_month else "")
              for d, c in zip(monthly.index, monthly["count"])]
    ax.bar(labels, monthly["mean"], label="Monthly mean (* partial month)")
    ax.yaxis.set_major_formatter(USD)
    ax.set(title="Monthly average Bitcoin price", xlabel="Month", ylabel="Mean price (USD)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _save(fig, "monthly_average_price")

    monthly.to_csv(config.RESULTS_DIR / "monthly_average_price.csv")
    descriptive_statistics(price).to_csv(config.RESULTS_DIR / "descriptive_statistics.csv")
    drawdown = price / price.cummax() - 1
    return {
        "first_price": price.iloc[0], "last_price": price.iloc[-1],
        "period_change_pct": (price.iloc[-1] / price.iloc[0] - 1) * 100,
        "min_price": price.min(), "min_date": str(price.idxmin().date()),
        "max_price": price.max(), "max_date": str(price.idxmax().date()),
        "max_drawdown_pct": drawdown.min() * 100,
        "mean_return_pct": returns.mean(), "sd_return_pct": returns.std(),
        "annualised_volatility_pct": returns.std() * np.sqrt(365),
        "skew_return": returns.skew(), "excess_kurtosis_return": returns.kurt(),
        "up_days": int((returns > 0).sum()), "down_days": int((returns < 0).sum()),
        "largest_gains": [[str(d.date()), v] for d, v in returns.nlargest(3).items()],
        "largest_falls": [[str(d.date()), v] for d, v in returns.nsmallest(3).items()],
        "rolling_sd_min": rolling_std.min(), "rolling_sd_min_date": str(rolling_std.idxmin().date()),
        "rolling_sd_max": rolling_std.max(), "rolling_sd_max_date": str(rolling_std.idxmax().date()),
        "monthly": [[d.strftime("%b %Y"), m, int(c)]
                    for d, m, c in zip(monthly.index, monthly["mean"], monthly["count"])],
    }
