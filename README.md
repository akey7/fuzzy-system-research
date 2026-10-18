# fuzzy-system-research
Backend code the fuzzy-system-finance app. Contains code to model stock prices with ARIMA and optimize portfolios. Obtains data from Massive.com. Uploads to front end via an S3 bucket. See a full listing of functioanlity in the tables of scripts and modules below.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python versions, virtual environments, and dependencies. Install `uv` first (for example, `brew install uv` on macOS). Dependencies are declared in `pyproject.toml`, and `uv.lock` pins the exact versions that were resolved. The project requires Python 3.13 and pins 3.13.5 in `.python-version`, which `uv` downloads automatically if it is not already installed.

### Development Dependencies

Create the `.venv` virtual environment and install production plus development dependencies (the `dev` group is included by default). This also installs the `fuzzy_system_research` package from `src/` in editable mode, so edits to the library code take effect immediately:

```
uv sync
```

### Production Dependencies

To install only the production dependencies:

```
uv sync --no-dev
```

### Running the backend

One command runs everything: it downloads prices (or reuses today's cached download), fits the walk-forward ARIMA forecasts (or reuses the cached forecasts for the data's as-of date), optimizes the portfolio, and uploads all results for the frontend. Run it through `uv` so it uses the project environment; there is no need to activate the virtual environment manually:

```
uv run python scripts/run_optimization_and_time_series.py
```

Options:

| Flag | Effect |
| --- | --- |
| `--no-upload` | Write the `output/` files but upload nothing. Use this to check a run without touching any bucket. |
| `--retrain-arima` | Retrain the ARIMA models even if forecasts for the as-of date are cached. |
| `--refresh-data` | Download prices again even if today's download is cached. |
| `--workers N` | Number of worker processes for ARIMA training. Defaults to half the logical CPUs (for example 8 on a 16-thread Intel CPU, 5 on a 10-core M4). `--workers 1` trains everything in one process. The forecasts do not depend on this setting. |
| `--config PATH` | Model settings file to use instead of `config/modeling_and_optimization.yml` (see below). |

Nothing is uploaded unless every step succeeds. If a later step fails, the ARIMA forecasts already written to `output/` are reused on the next run.

Always run scripts from the repository root, because the scripts read and write `input/` and `output/` (and read `config/`) relative to the current working directory.

### Model settings

Time series settings live in `config/modeling_and_optimization.yml`, which is checked into the repository:

```yaml
time_series_models:
  training_window_length: 252    # closes (trading sessions) per ARIMA training window
  evaluation_window_length: 20   # most recent trading days backtested and scored
```

- `training_window_length` counts **closing prices**, so 252 closes give 251 daily log returns, about one trading year. It must be at least 30.
- `evaluation_window_length` is the number of backtest days in `all_forecasts.csv` (plus the live forecast) and in the metrics of `all_forecasts_metadata.yml`. It must be at least 1.
- Both keys are required, and unknown keys are rejected, so a typo stops the run before any download instead of silently using a default.
- The run downloads enough history for both settings and the optimization's 252 sessions.

Changing a value changes what is uploaded. Run with `--retrain-arima` afterwards, because the cached forecasts for a date are otherwise reused, and update `INTERFACE_CONTRACT.md` wherever it documents the value.

Run development tools the same way, for example `uv run pytest` or `uv run black .`.

### Managing dependencies

```
uv add <package>           # add a production dependency
uv add --dev <package>     # add a development dependency
uv remove <package>        # remove a dependency
uv lock --upgrade          # upgrade all packages within the declared constraints
uv sync                    # bring .venv in line with uv.lock
```

Commit both `pyproject.toml` and `uv.lock` whenever dependencies change.

### `.env` file

**Never commit the `.env` file to GitHub!** It should never be made publicly accessible because it contains API keys.

Here are the keys the scripts need to function:

1. `MASSIVE_API_KEY`: Key to Massive.com API for stock and index data. Obtain from API provider.

2. `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_RWDELETE`, `FSF_FRONT_END_BUCKET_KEY_ID`, `FSF_FRONT_END_BUCKET_ENDPOINT`: Region, read/write/delete key, key id, and endpoint of the DigitalOcean S3/Spaces bucket. Set as appropriate for development or production environments. Obtain values from DigitalOcean or AWS environments.

3. `FSF_FRONT_END_BUCKET_NAME`: Name of the one DigitalOcean Space (S3 bucket) that holds all frontend data. `FSF_FRONT_END_BUCKET_ENDPOINT` must be the region-only endpoint (e.g. `https://nyc3.digitaloceanspaces.com`) without the Space name; the uploader refuses to run otherwise.

4. `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER` and `FSF_FRONT_END_TIME_SERIES_FOLDER`: names of the two top-level folders inside that Space, holding the portfolio optimization and ARIMA time series data. Dev and production share the Space and use different folder names. Files are uploaded to `<folder>/YYYY-MM-DD/<file>`; see `INTERFACE_CONTRACT.md` for the full key layout.

Items 2 through 4 connect the backend to the front end.

## Project structure

The repository separates code that you **run** from code that you **import**:

```
fuzzy-system-research/
├── pyproject.toml                # Dependencies, Python version, build config
├── uv.lock                       # Exact resolved dependency versions
├── scripts/                      # RUN these: entry points that wire the library together
│   ├── run_optimization_and_time_series.py
│   └── ticker_download_to_csv.py
├── src/fuzzy_system_research/    # IMPORT these: reusable library code, no entry points
│   ├── pipeline.py               # One full run: prices → ARIMA → optimization → uploads
│   ├── data/                     # Market data download, caching, price reshaping, and date logic
│   ├── optimization/             # Portfolio optimization math
│   ├── forecasting/              # ARIMA models and the walk-forward training workflow
│   └── storage/                  # Output/bucket key layout and S3/Spaces uploads
├── tests/                        # Tests (use mocks and temporary directories only)
├── input/                        # Cached market data (not committed)
└── output/                       # Results of runs (not committed)
```

Scripts contain argument handling and orchestration: they decide *what* to run. The package contains the computation and I/O helpers: it knows *how*. Scripts import from the package (for example, `from fuzzy_system_research.optimization.portfolio import calc_min_variance_portfolio`), and the package never imports from `scripts/`. This keeps the quantitative code importable and testable without triggering a download or an upload.

### Data folders

The following folders should be created before running the scripts in the section below. Their content is not committed to GitHub.

1. `input/`: Holds risk free rate data.

2. `input/annual/`: Holds the cached price downloads (about 277 trading days, one file per run date). Both the optimization and the ARIMA forecasts use this data: the optimization uses its last 252 trading days, and ARIMA its last 272 (one-year training windows of 252 closes plus 20 scored backtest days). A cached file that is too short, for example one written before the ARIMA window grew, is downloaded again automatically.

3. `output/`: Holds the results of runs, in one folder per data as-of date (the last NYSE trading day in the run's data). Subfolders are created automatically. The layout mirrors the frontend buckets, where each file's key is its path below `output/<kind>/`:

   ```
   output/portfolio_optimization/YYYY-MM-DD/optimization_metadata.yml
   output/portfolio_optimization/YYYY-MM-DD/portfolio_optimization_plot_data.h5
   output/time_series/YYYY-MM-DD/all_forecasts.csv
   output/time_series/YYYY-MM-DD/all_forecasts_metadata.yml
   ```

   Re-running for the same date overwrites that date's files, except that ARIMA training is skipped when that date's `all_forecasts.csv` already exists (unless `--retrain-arima` is given). See `INTERFACE_CONTRACT.md` for the bucket layout the frontend reads.

An older `input/monthly/` folder, if present, is no longer used and can be deleted.

### Scripts

Run these directly with `uv run python scripts/<filename>` from the repository root.

| Filename                                | Purpose                                                                                                                                                                                                                   | Uploads to front end? |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `scripts/run_optimization_and_time_series.py` | The whole backend in one run: obtain adjusted close prices, fit walk-forward ARIMA forecasts, compute max Sharpe ratio and min variance portfolios, the efficient frontier, and a Monte Carlo simulation, then upload. See "Running the backend" for flags. | Yes |
| `scripts/ticker_download_to_csv.py`     | Download a supported symbol from Massive.com and put the results into a `.csv` file. Usage: `uv run python scripts/ticker_download_to_csv.py <ticker> <date_from> <date_to> <filename>`.                                    | No                    |

### Library modules

These modules are imported by the scripts and are not meant to be executed directly.

| Module (under `src/fuzzy_system_research/`) | Purpose                                                                                                                                                 | Uploads to front end?                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `config.py`                                 | Reads and strictly validates `config/modeling_and_optimization.yml` into `TimeSeriesConfig`.                                                            | No                                   |
| `data/ticker_download_manager.py`           | Downloads new data from Massive.com or retrieves old data from the cache.                                                                               | No                                   |
| `data/date_manager.py`                      | Date and business day calculations on the NYSE trading calendar (via `exchange_calendars`).                                                             | No                                   |
| `data/prices.py`                            | Pivots downloaded prices to a wide table and selects the most recent trading sessions.                                                                 | No                                   |
| `optimization/portfolio.py`                 | Monte Carlo portfolio simulation, minimum variance and max Sharpe ratio optimization, efficient frontier, annualization, and risk-free rate retrieval.  | No                                   |
| `optimization/workflow.py`                  | One full optimization run from prices to annualized results, plus the `.h5` and `.yml` writers.                                                        | No                                   |
| `forecasting/arima_models.py`               | Trains ARIMA models on price data.                                                                                                                      | No                                   |
| `forecasting/ticker_predict.py`             | `ArimaWalkForward`: walk-forward ARIMA training over rolling windows; `load_or_train_forecasts` caches the result.                                      | No                                   |
| `forecasting/metrics.py`                    | Accuracy of the ARIMA and naive forecasts over the configured number of most recent trading days (RMSE, MAE, MAPE, relative MAE, skill, directional accuracy) for `all_forecasts_metadata.yml`. | No |
| `pipeline.py`                               | Orchestrates one backend run and its uploads (`csv`, then `.h5`, then `.yml` last).                                                                    | Yes                                  |
| `storage/layout.py`                         | Builds the dated `output/` paths and matching bucket keys (`YYYY-MM-DD/<file>`).                                                                        | No                                   |
| `storage/s3_uploader.py`                    | Uploads files to the one DigitalOcean Space (`FSF_FRONT_END_BUCKET_NAME`) by full object key; rejects non-region-only endpoints.                       | Yes (other modules use it to upload) |
