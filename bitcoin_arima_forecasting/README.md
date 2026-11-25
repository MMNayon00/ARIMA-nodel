# Bitcoin Seven-Day Price Forecast with ARIMA

## Result (data retrieved 25 November 2026)
* Selected model: **ARIMA(0,1,0) on log price** (a random walk)
* Forecast, 26 Nov to 2 Dec 2026: **Neutral**, flat at **$59,742.21**
* 95% interval on day 7: about **$52,944 to $67,414**
* The model has **no skill beyond the naive forecast**. This is not investment advice.

---

## 1. Where to run
Run all commands in a **terminal**, from this project folder (`bitcoin_arima_forecasting`).

* **VS Code:** menu **Terminal → New Terminal**
* **macOS Terminal app:** press `Cmd + Space`, type `Terminal`, press Enter

Go to the project folder (keep the quotes, because the path contains a space):
```bash
cd "/Users/md.mostofanayon/Desktop/ARIMA nodel/bitcoin_arima_forecasting"
```
On another computer, replace this with the folder's location there.

## 2. How to run everything (one command)
```bash
./run_all.sh
```
It takes about 10 to 15 minutes (the notebook repeats the full analysis). It is finished when it prints `Done.`
Warning messages from the statistics libraries can appear; a real failure stops the script with a `Traceback`.

> **Note:** this rebuilds the notebook and the Word report. If you have typed your name or other edits into
> the report, **save a copy under another file name first**, or your edits will be overwritten.

No internet connection is needed, because the saved dataset is used.

## 3. First-time setup on another computer
`run_all.sh` creates the environment automatically if none exists. To do it by hand (tested with
Python 3.14 on macOS):
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
On this computer the environment already exists one folder up. To activate it:
```bash
source ../.venv/bin/activate
```
Windows cannot run `run_all.sh` directly; run the commands in section 4.

## 4. Run one step at a time
Activate the environment first (section 3).

| Command | What it does |
|---|---|
| `python run_pipeline.py` | Full analysis: figures and result files |
| `python -m src.cross_check` | Compares the prices with Coinbase |
| `python build_notebook.py` | Rebuilds and executes the notebook |
| `python -m src.report_builder` | Rebuilds the Word report |

## 5. Where to see everything
| What | Location | How to open |
|---|---|---|
| **Academic report** | `report/bitcoin_arima_assessment_report.docx` | Word or Pages |
| **Notebook** (code, outputs, explanations) | `notebooks/bitcoin_arima_analysis.ipynb` | VS Code |
| **Seven-day forecast** | `results/seven_day_forecast.csv` | VS Code, Excel or Numbers |
| **Model accuracy** | `results/model_evaluation.csv` | VS Code, Excel or Numbers |
| Actual vs predicted | `results/test_predictions_one_step.csv` | as above |
| Candidate models | `results/candidate_comparison.csv`, `results/arima_grid_search.csv` | as above |
| Stationarity tests | `results/stationarity_tests.csv` | as above |
| Alternative-model forecast | `results/sensitivity_forecast_without_root_screen.csv` | as above |
| **All graphs** | `outputs/` | Click an image in VS Code, or use Finder |
| **Dataset** | `data/bitcoin_historical_data.csv` | VS Code, Excel or Numbers |
| Data source and retrieval date | `data/data_source_metadata.json` | VS Code |
| Source code | `src/` | VS Code |

Open files from the terminal (macOS):
```bash
open report/bitcoin_arima_assessment_report.docx
open outputs                        # all graphs in Finder
open outputs/seven_day_forecast.png
open results/seven_day_forecast.csv
```

**Main graphs:** `historical_price.png`, `daily_returns.png`, `rolling_statistics.png`, `acf_pacf.png`,
`model_diagnostics.png`, `test_predictions.png`, `seven_day_forecast.png`.

### Viewing the notebook in VS Code
1. Open `notebooks/bitcoin_arima_analysis.ipynb`. The saved outputs are visible without running anything.
2. To run it yourself, install the **Python** and **Jupyter** extensions if VS Code asks.
3. Click **Select Kernel** (top right) → **Python Environments** → choose the `.venv` interpreter.
4. Click **Run All**.

## 6. Download newer data (optional)
```bash
cp -r data data_backup
python run_pipeline.py --refresh
```
This **overwrites** the saved dataset, and the forecast dates and results will change. The report builder and
`run_all.sh` will then stop at the report step on purpose, because the report's commentary describes the
original data. The notebook commentary would also need review.

## 7. Troubleshooting
| Problem | Solution |
|---|---|
| `permission denied: ./run_all.sh` | Run `bash run_all.sh` |
| `No such file or directory` | Run the `cd` command in section 1 |
| `ModuleNotFoundError` | Activate the environment, then `pip install -r requirements.txt` |
| `Dataset checksum mismatch` | The CSV was edited. Restore it, or download again with `--refresh` |
| Report will not save | Close the report in Word, then run again |

## Method notes
* Source: Blockchain.com chart API (`api.blockchain.info/charts/market-price`), the data behind the chart page.
* Chronological 80/20 split; the model is chosen using training data only.
* An AR/MA root screen was added after a first run selected ARIMA(6,1,4) with unit-circle MA roots.
  That model is reported as a sensitivity analysis.
* Point forecasts are medians (exponentiated log forecasts).

## Before submitting
Fill in the name, ID, module and date placeholders on the report title page, and follow your
institution's rules on declaring AI assistance.

## View in the browser (localhost)
This project has no web app. To browse the notebook, graphs and CSV files at localhost, use JupyterLab:
```bash
cd "/Users/md.mostofanayon/Desktop/ARIMA nodel/bitcoin_arima_forecasting"
source ../.venv/bin/activate
pip install jupyterlab        # first time only
jupyter lab
```
The browser opens automatically (otherwise copy the `http://localhost:8888/lab?token=...` link from the terminal).
Open `notebooks/bitcoin_arima_analysis.ipynb`. Press `Ctrl + C` twice in the terminal to stop the server.
