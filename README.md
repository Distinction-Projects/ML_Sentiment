# SARIMA Dashboard (Python + Dash)

Interactive Plotly Dash application that guides you through SARIMA modeling for time series forecasting using Python. The current dependency set targets Python 3.11 (works on 3.10–3.12, but 3.11 is recommended locally and on Render).

- Live app: https://gabria1.pythonanywhere.com/
- Original tutorial: [Time Series Data Analysis with SARIMA and Dash](https://medium.com/towards-data-science/time-series-data-analysis-with-sarima-and-dash-f4199c3fc092) on Towards Data Science. Please retain this attribution if you reuse or deploy.

## Workflow in the app
- **Data set up:** start from the built-in AirPassengers dataset (`data/AirPassengers.csv`). The app expects two columns (`Time`, `Values`) parsed as dates and numeric data.
- **Stationarity checks:** apply log transforms and differencing, run the Augmented Dickey-Fuller test, and inspect ACF/PACF and Box-Cox plots.
- **Model selection:** run a SARIMA `(p,d,q)(P,D,Q,m)` grid search scored by AIC with train/test split control.
- **Prediction:** fit the selected model, visualize train/test forecasts with confidence intervals, and review residual ACF/PACF.

## Run locally (Python 3.11)
```bash
git clone <your fork of this repo>
cd sarima_dashboard

# macOS / Linux (recommended when available)
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell) — preferred if you have the Python launcher
py -3.11 -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
py -3.11 -m venv .venv
.\.venv\Scripts\activate.bat

# If `py` is not available but `python` points to a suitable interpreter
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell

pip install --upgrade pip
pip install -r requirements.txt
# Pre-download NLTK data to speed up first run (optional; auto-downloads on startup if missing)
python -m nltk.downloader stopwords punkt wordnet vader_lexicon punkt_tab
python app.py
```

The server runs at http://localhost:8050 by default. Edit `app.py` to set `debug=True` while developing if you want auto-reload. The `.venv` folder is gitignored; if you use a different env name, add it to `.gitignore` to keep it out of commits.

### Working with data
- The default AirPassengers sample lives in `data/AirPassengers.csv`. Replace that file (keep the `Time` and `Values` headers) to experiment with your own series.
- Restart the app after swapping data to ensure the in-memory dataset refreshes.

## Deploy to Render
- Option A: create a new Web Service, point at your repo, set build command `pip install -r requirements.txt`, start command `gunicorn app:server`, and env var `PYTHON_VERSION=3.11`. Render supplies `PORT`; gunicorn will bind to it automatically.
- Option B: keep the provided `render.yaml` and let Render auto-detect it as a blueprint.

## Deploy to DigitalOcean
- DigitalOcean App Platform: the included `app.yaml` is a ready-to-deploy spec. Point App Platform at this repo (`Distinction-Projects/ML_Sentiment`, branch `main`), confirm the detected spec, and deploy. Adjust the repo/branch in the manifest if your fork differs. The spec installs requirements, pre-downloads NLTK data, and starts `gunicorn app:server --bind 0.0.0.0:$PORT --worker-tmp-dir ${WORKER_TMP_DIR:-/tmp}` with Python 3.11 on a `basic-xxs` instance. A `.python-version` file pins Python 3.11 for the DO buildpack. `WORKER_TMP_DIR` defaults to `/dev/shm` (set in `app.yaml`) but the command falls back to `/tmp` if that path is absent (e.g., local macOS test).
- Manual commands (if you prefer to enter them in the UI): build `pip install -r requirements.txt && python -m nltk.downloader stopwords punkt wordnet vader_lexicon punkt_tab`; run `gunicorn app:server`.

## Project layout
- `app.py` bootstraps the multi-page Dash app.
- `pages/` contains the four guided steps described above.
- `assets/` holds shared components (navigation, footer, styles).
- `data/` contains the default time series sample.

![dash_app_step03](https://user-images.githubusercontent.com/57110246/236455995-a98416d9-57f3-4c6e-b41b-0583ba66c86d.gif)


This is a small change to the readme!
