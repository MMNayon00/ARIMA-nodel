"""Run the complete analysis: python run_pipeline.py [--refresh]"""
import argparse
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from src import arima_model as am
from src import config, data_collection, exploratory_analysis, forecasting, preprocessing, stationarity


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(refresh: bool = False) -> dict:
    config.ensure_directories()
    print("1/6 Data collection and cleaning")
    raw, metadata = data_collection.load_data(refresh)
    price, quality = preprocessing.clean_data(raw)
    train_size = int(len(price) * config.TRAIN_FRACTION)
    train_log = np.log(price.iloc[:train_size])

    print("2/6 Exploratory analysis")
    eda = exploratory_analysis.run_eda(price)

    print("3/6 Stationarity, ACF and PACF (d chosen from training data)")
    stationarity.stationarity_table(price, train_size)
    stationarity.plot_differencing(price)
    d = stationarity.determine_d(train_log)
    correlation = stationarity.acf_pacf_analysis(train_log, d)

    print("4/6 Model search and selection (training data only)")
    grid = am.grid_search(train_log, d)
    auto_choice = am.auto_arima_suggestion(train_log, d)
    order, trend, candidates, validation_start = am.select_model(price, grid, train_size, auto_choice)
    name = am.label(order, trend)
    print("    selected:", name)

    print("5/6 Hold-out evaluation and diagnostics")
    n = len(price)
    one_step = am.walk_forward(price, order, trend, train_size, n, 1)
    weekly = am.walk_forward(price, order, trend, train_size, n, config.HORIZON)
    static = forecasting.static_forecast(price, order, trend, train_size)
    base_one = am.baseline_forecasts(price, train_size, n, 1)
    base_week = am.baseline_forecasts(price, train_size, n, config.HORIZON)
    base_static = static[["date", "horizon", "origin_price", "actual"]].assign(
        predicted=price.iloc[train_size - 1])

    rows = []
    def add(evaluation, model, frame, role):
        rows.append({"evaluation": evaluation, "model": model, "role": role, **am.metrics(frame)})

    # Selected model, baselines, then other shortlisted models (post-hoc, never used for selection).
    for evaluation, frame, baselines in (("1-step walk-forward", one_step, base_one),
                                         ("7-day rolling origin", weekly, base_week)):
        add(evaluation, name, frame, "selected")
        for key, base in baselines.items():
            add(evaluation, key, base, "baseline")
    add("Static multi-step", name, static, "selected")
    add("Static multi-step", "Naive (last value)", base_static, "baseline")
    comparisons = {}
    for c in candidates[~candidates.selected].itertuples():
        frame = am.walk_forward(price, (c.p, c.d, c.q), c.trend, train_size, n, 1)
        add("1-step walk-forward", c.model, frame, "comparison (post-hoc)")
        add("7-day rolling origin", c.model,
            am.walk_forward(price, (c.p, c.d, c.q), c.trend, train_size, n, config.HORIZON),
            "comparison (post-hoc)")
        comparisons[c.model] = am.diebold_mariano(frame, base_one["Naive (last value)"])
    comparisons[name] = am.diebold_mariano(one_step, base_one["Naive (last value)"])
    evaluation_table = pd.DataFrame(rows)
    evaluation_table.to_csv(config.RESULTS_DIR / "model_evaluation.csv", index=False)

    detail = one_step[["date", "actual", "predicted", "lower_95", "upper_95"]].copy()
    detail["error"] = detail.actual - detail.predicted
    detail["absolute_error"] = detail.error.abs()
    detail["percentage_error"] = detail.absolute_error / detail.actual * 100
    detail.to_csv(config.RESULTS_DIR / "test_predictions_one_step.csv", index=False)
    static.assign(absolute_error=(static.actual - static.predicted).abs(),
                  percentage_error=(static.actual - static.predicted).abs() / static.actual * 100
                  ).to_csv(config.RESULTS_DIR / "test_predictions_static.csv", index=False)
    weekly.to_csv(config.RESULTS_DIR / "rolling_origin_7day.csv", index=False)
    forecasting.plot_test_predictions(price, train_size, one_step, static, name)

    def coverage(frame):
        return float(((frame.actual >= frame.lower_95) & (frame.actual <= frame.upper_95)).mean() * 100)

    train_fit = am.fit(train_log, order, trend)
    train_diagnostics = am.residual_diagnostics(
        train_fit, order, "model_diagnostics.png", f"{name}: training-sample residuals")

    print("6/6 Final refit and seven-day forecast")
    final_fit, forecast_table, forecast_summary = forecasting.seven_day_forecast(price, order, trend, name)
    full_diagnostics = am.residual_diagnostics(
        final_fit, order, "model_diagnostics_full_sample.png", f"{name}: full-sample residuals")

    # Sensitivity: what the rule would have chosen without the unit-root screen.
    other = candidates[candidates.selected_without_root_screen].iloc[0]
    other_order, other_trend = (int(other.p), int(other.d), int(other.q)), other.trend
    other_fit, other_table, other_summary = forecasting.seven_day_forecast(
        price, other_order, other_trend, other.model,
        save_as="sensitivity_forecast_without_root_screen.csv", make_plot=False)
    sensitivity = {
        "model": other.model, "forecast": other_summary,
        "table": other_table.to_dict("records"),
        "diagnostics_train": am.residual_diagnostics(
            am.fit(train_log, other_order, other_trend), other_order,
            "sensitivity_diagnostics.png", f"{other.model}: training-sample residuals"),
        "diagnostics_full": am.residual_diagnostics(
            other_fit, other_order, "sensitivity_diagnostics_full_sample.png",
            f"{other.model}: full-sample residuals")}

    summary = {
        "metadata": metadata, "quality": quality, "eda": eda,
        "split": {"n": n, "train_n": train_size, "test_n": n - train_size,
                  "train_start": str(price.index[0].date()),
                  "train_end": str(price.index[train_size - 1].date()),
                  "test_start": str(price.index[train_size].date()),
                  "test_end": str(price.index[-1].date()),
                  "validation_start": str(price.index[validation_start].date()),
                  "validation_n": train_size - validation_start},
        "d": d, "correlation": correlation,
        "grid_models": len(grid), "grid_converged": int(grid.converged.sum()),
        "auto_arima": am.label(*auto_choice),
        "selected": {"name": name, "order": order, "trend": trend},
        "coverage": {"one_step": coverage(one_step), "weekly": coverage(weekly),
                     "static": coverage(static),
                     "weekly_by_horizon": {int(h): coverage(g) for h, g in weekly.groupby("horizon")}},
        "walk_forward_nonconverged": int((~one_step.converged).sum()),
        "test_moves": {"up": int((one_step.actual > one_step.origin_price).sum()),
                       "down": int((one_step.actual < one_step.origin_price).sum())},
        "diebold_mariano": comparisons,
        "diagnostics_train": train_diagnostics, "diagnostics_full": full_diagnostics,
        "forecast": forecast_summary, "sensitivity": sensitivity,
        "environment": {"python": platform.python_version(), **{p: version(p) for p in (
            "pandas", "numpy", "matplotlib", "seaborn", "statsmodels", "scikit-learn",
            "scipy", "pmdarima", "requests")}},
    }
    summary = _jsonable(summary)
    (config.RESULTS_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(forecast_table.to_string(index=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download new data first")
    main(parser.parse_args().refresh)
