"""Download (or reload) Blockchain.com daily market-price data.

By default the saved snapshot is reused so that results are reproducible.
Run ``python -m src.data_collection --refresh`` to download new data.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from . import config


def download_data() -> pd.DataFrame:
    """Fetch one year of data from the official API and preserve it unmodified."""
    config.ensure_directories()
    try:
        response = requests.get(config.API_URL, params=config.API_PARAMS, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"Could not retrieve data from {config.API_URL}: {error}") from error
    if payload.get("status") != "ok" or not payload.get("values"):
        raise RuntimeError(f"Unexpected API response: {str(payload)[:200]}")

    config.RAW_JSON.write_bytes(response.content)
    frame = pd.DataFrame(payload["values"]).rename(
        columns={"x": "timestamp", "y": "market_price_usd"})
    frame.insert(0, "date", pd.to_datetime(frame["timestamp"], unit="s", utc=True)
                 .dt.strftime("%Y-%m-%d %H:%M:%S"))
    frame.to_csv(config.DATA_CSV, index=False)

    metadata = {
        "source": "Blockchain.com Charts API",
        "chart_page": config.CHART_PAGE,
        "request_url": response.url,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "metric_name": payload.get("name"),
        "metric_description": payload.get("description"),
        "unit": payload.get("unit"),
        "period": payload.get("period"),
        "observations": len(frame),
        "csv_sha256": hashlib.sha256(config.DATA_CSV.read_bytes()).hexdigest(),
    }
    config.METADATA_JSON.write_text(json.dumps(metadata, indent=2))
    return frame


def load_data(refresh: bool = False):
    """Return (raw_frame, metadata); verify the snapshot has not been altered."""
    if refresh or not config.DATA_CSV.exists():
        download_data()
    if not config.METADATA_JSON.exists():
        raise FileNotFoundError("Metadata file missing; rerun with --refresh.")
    metadata = json.loads(config.METADATA_JSON.read_text())
    digest = hashlib.sha256(config.DATA_CSV.read_bytes()).hexdigest()
    if digest != metadata["csv_sha256"]:
        raise ValueError("Dataset checksum mismatch: the CSV was modified after download.")
    return pd.read_csv(config.DATA_CSV), metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="download a new snapshot")
    data, info = load_data(parser.parse_args().refresh)
    print(json.dumps(info, indent=2))
    print(data.head(), data.tail(), sep="\n")
