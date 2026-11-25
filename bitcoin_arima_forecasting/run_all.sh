#!/usr/bin/env bash
# Run every step on the saved dataset. Rebuilds the notebook and the Word report.
set -euo pipefail
cd "$(dirname "$0")"

# Use the first virtual environment found: inside the project, or one folder up.
if [ -x .venv/bin/python ]; then PY=.venv/bin/python
elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
else
  echo "No virtual environment found. Creating .venv and installing packages..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  PY=.venv/bin/python
fi
export OMP_NUM_THREADS=1

echo "== 1/4 Analysis pipeline";  "$PY" run_pipeline.py
echo "== 2/4 Source cross-check"; "$PY" -m src.cross_check
echo "== 3/4 Notebook";           "$PY" build_notebook.py
echo "== 4/4 Word report";        "$PY" -m src.report_builder
echo "Done. See results/, outputs/, report/ and notebooks/."
