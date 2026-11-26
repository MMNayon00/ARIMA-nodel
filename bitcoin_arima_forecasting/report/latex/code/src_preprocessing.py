"""Validation and cleaning of the raw price series."""
import numpy as np
import pandas as pd


def clean_data(raw: pd.DataFrame):
    """Return (daily price Series, quality report).

    Gaps, if any, are forward-filled: this uses only past information, so it
    cannot leak future prices into the model.
    """
    frame = raw.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["market_price_usd"] = pd.to_numeric(frame["market_price_usd"], errors="coerce")
    non_numeric = int(frame["market_price_usd"].isna().sum())
    non_positive = int((frame["market_price_usd"] <= 0).sum())
    frame.loc[frame["market_price_usd"] <= 0, "market_price_usd"] = np.nan
    was_sorted = bool(frame["date"].is_monotonic_increasing)
    frame = frame.sort_values("timestamp")
    duplicates = int(frame["date"].duplicated().sum())
    frame = frame.drop_duplicates("date", keep="last").set_index("date")

    calendar = pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date")
    missing_dates = len(calendar.difference(frame.index))
    series = frame["market_price_usd"].reindex(calendar)
    values_to_fill = int(series.isna().sum())
    series = series.ffill().dropna().astype(float).asfreq("D")
    if series.empty or not (series > 0).all():
        raise ValueError("Cleaned series is empty or contains non-positive prices.")

    report = {
        "raw_rows": int(len(raw)), "clean_rows": int(len(series)),
        "start_date": str(series.index[0].date()), "end_date": str(series.index[-1].date()),
        "originally_sorted": was_sorted, "duplicate_dates": duplicates,
        "missing_calendar_dates": missing_dates, "non_numeric_values": non_numeric,
        "non_positive_values": non_positive, "values_forward_filled": values_to_fill,
        "frequency": series.index.freqstr,
    }
    return series.rename("price"), report
