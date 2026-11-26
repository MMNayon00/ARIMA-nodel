"""Project-wide paths and analysis settings (all choices are declared here)."""
import warnings
from pathlib import Path
import matplotlib
import seaborn as sns

# statsmodels 0.15 announces future return-type changes; versions are pinned.
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, OUTPUT_DIR, RESULTS_DIR = ROOT / "data", ROOT / "outputs", ROOT / "results"
DATA_CSV = DATA_DIR / "bitcoin_historical_data.csv"
RAW_JSON = DATA_DIR / "blockchain_market_price_raw.json"
METADATA_JSON = DATA_DIR / "data_source_metadata.json"

CHART_PAGE = "https://www.blockchain.com/ko/charts/market-price"
API_URL = "https://api.blockchain.info/charts/market-price"
API_PARAMS = {"timespan": "1year", "format": "json", "sampled": "false"}

TRAIN_FRACTION = 0.80        # chronological train/test split
VALIDATION_FRACTION = 0.20   # last part of the TRAINING set, used to choose the model
MAX_P = MAX_Q = 7            # grid limit; 7 allows a weekly (lag-7) effect
HORIZON = 7
ALPHA = 0.05
LB_LAGS = (10, 20)
MIN_ROOT_MODULUS = 1.001     # reject near unit-root / non-invertible fits
PARSIMONY_TOLERANCE = 0.01   # validation RMSEs within 1% are treated as tied
DAILY_NEUTRAL_PCT = 0.10     # |daily change| below this is labelled Neutral
WEEKLY_NEUTRAL_PCT = 1.00    # |7-day change| below this is labelled Neutral


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    matplotlib.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200,
                                "savefig.bbox": "tight", "axes.titleweight": "bold"})


def ensure_directories() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
