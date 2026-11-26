"""Independent plausibility check against Coinbase daily closes (NOT used for modelling).

python -m src.cross_check            # reuse the saved Coinbase snapshot
python -m src.cross_check --refresh  # download it again
"""
import argparse
import json

import pandas as pd
import requests

from . import config
from .data_collection import load_data
from .preprocessing import clean_data

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
SNAPSHOT = config.DATA_DIR / "coinbase_cross_check.csv"


def run(refresh: bool = False) -> dict:
    price, _ = clean_data(load_data()[0])
    if refresh or not SNAPSHOT.exists():
        end = price.index[-1]
        response = requests.get(URL, timeout=60, params={
            "granularity": 86400, "end": str(end.date()),
            "start": str((end - pd.Timedelta(days=297)).date())})
        response.raise_for_status()
        rows = sorted((r[0], r[4]) for r in response.json())
        frame = pd.DataFrame(rows, columns=["timestamp", "close_usd"])
        frame.insert(0, "candle_date", pd.to_datetime(frame.timestamp, unit="s").dt.strftime("%Y-%m-%d"))
        frame.to_csv(SNAPSHOT, index=False)
    frame = pd.read_csv(SNAPSHOT, parse_dates=["candle_date"])
    close = frame.set_index("candle_date")["close_usd"]
    result = {"source": URL}
    for name, shift in (("same_date", 0), ("previous_day_close", 1)):
        other = close.copy()
        other.index = other.index + pd.Timedelta(days=shift)
        pair = pd.concat([price, other], axis=1, join="inner")
        gap = (pair.price / pair.close_usd - 1).abs() * 100
        result[name] = {"days": int(len(pair)), "mean_abs_diff_pct": float(gap.mean()),
                        "max_abs_diff_pct": float(gap.max())}
    (config.RESULTS_DIR / "source_cross_check.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    print(json.dumps(run(parser.parse_args().refresh), indent=2))
