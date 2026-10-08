# fuzzy-system-research
Backend code the fuzzy-system-finance app. Contains code to model stock prices with ARIMA and optimize portfolios. Obtains data from Massive.com. Uploads to front end via an S3 bucket. See a full listing of functioanlity in the tables of scripts and modules below.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for Python versions, virtual environments, and dependencies. Install `uv` first (for example, `brew install uv` on macOS). Dependencies are declared in `pyproject.toml`, and `uv.lock` pins the exact versions that were resolved. The project requires Python 3.13 (`.python-version`), which `uv` downloads automatically if it is not already installed.

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

### Running scripts

Run scripts through `uv` so they use the project environment; there is no need to activate the virtual environment manually:

```
uv run python scripts/portfolio_optimization.py
```

Always run scripts from the repository root, because the scripts read and write `input/` and `output/` relative to the current working directory.

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

3. `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME`: Space names for portfolio and ARIMA time series data.

Items 2 and 3 connect the backend to the front end.

## Project structure

The repository separates code that you **run** from code that you **import**:

```
fuzzy-system-research/
├── pyproject.toml                # Dependencies, Python version, build config
├── uv.lock                       # Exact resolved dependency versions
├── scripts/                      # RUN these: entry points that wire the library together
│   ├── portfolio_optimization.py
│   ├── ticker_predict_upload.py
│   └── ticker_download_to_csv.py
├── src/fuzzy_system_research/    # IMPORT these: reusable library code, no entry points
│   ├── data/                     # Market data download, caching, and date logic
│   ├── optimization/             # Portfolio optimization math and plotting
│   ├── forecasting/              # ARIMA models and the walk-forward training workflow
│   └── storage/                  # S3/Spaces uploads
├── tests/                        # Tests (use mocks and temporary directories only)
├── input/                        # Cached market data (not committed)
└── output/                       # Results of runs (not committed)
```

Scripts contain argument handling and orchestration: they decide *what* to run. The package contains the computation and I/O helpers: it knows *how*. Scripts import from the package (for example, `from fuzzy_system_research.optimization.portfolio import calc_min_variance_portfolio`), and the package never imports from `scripts/`. This keeps the quantitative code importable and testable without triggering a download or an upload.

### Data folders

The following folders should be created before running the scripts in the section below. Their content is not committed to GitHub.

1. `input/`: Holds risk free rate data.

2. `input/annual/`: Holds annual ticker data.

3. `input/monthly/`: Holds monthly ticker data.

4. `output/`: Holds output data.

### Scripts

Run these directly with `uv run python scripts/<filename>` from the repository root.

| Filename                                | Purpose                                                                                                                                                                                                                   | Uploads to front end? |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `scripts/portfolio_optimization.py`     | Obtain adjusted close price data for a portfolio of stocks and compute max Sharpe ratio portfolio weights, min variance portfolio weights, efficient frontier, and Monte Carlo simulation of portfolio risks and returns. | Yes                   |
| `scripts/ticker_predict_upload.py`      | Obtain 2 months of data and fit a month's worth of ARIMA models on the adjusted close prices.                                                                                                                             | Yes                   |
| `scripts/ticker_download_to_csv.py`     | Download a supported symbol from Massive.com and put the results into a `.csv` file. Usage: `uv run python scripts/ticker_download_to_csv.py <ticker> <date_from> <date_to> <filename>`.                                    | No                    |

### Library modules

These modules are imported by the scripts and are not meant to be executed directly.

| Module (under `src/fuzzy_system_research/`) | Purpose                                                                                                                                                 | Uploads to front end?                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `data/ticker_download_manager.py`           | Downloads new data from Massive.com or retrieves old data from the cache.                                                                               | No                                   |
| `data/date_manager.py`                      | Date and business day calculations.                                                                                                                     | No                                   |
| `optimization/portfolio.py`                 | Monte Carlo portfolio simulation, minimum variance and max Sharpe ratio optimization, efficient frontier, risk-free rate retrieval, and plotting.       | No                                   |
| `forecasting/arima_models.py`               | Trains ARIMA models on price data.                                                                                                                      | No                                   |
| `forecasting/ticker_predict.py`             | `TickerPredictUpload`: walk-forward ARIMA training over rolling windows, plus the upload of the forecasts.                                              | Yes                                  |
| `storage/s3_uploader.py`                    | Uploads files to S3/Spaces buckets on AWS/DigitalOcean.                                                                                                 | Yes (other modules use it to upload) |
