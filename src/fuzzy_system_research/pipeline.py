"""
One backend run: download (or load cached) prices, ARIMA forecasts (or load
cached), forecast metrics, portfolio optimization, then upload all four
frontend files.

The time series settings (training window, scored days) come from the
configuration file, passed in as a TimeSeriesConfig (see config.py).

Order of operations:
1. Price data: one download of price_download_sessions() trading days,
   cached per run date in input/annual/. A cached download too short for the
   steps below is downloaded again once.
2. ARIMA: trained on the last TimeSeriesConfig.arima_window_sessions of that data,
   cached as output/time_series/<as-of date>/all_forecasts.csv. Metrics on
   those forecasts are written next to it as all_forecasts_metadata.yml.
3. Optimization: on the last OPTIMIZATION_LOOKBACK_SESSIONS sessions, written
   to output/portfolio_optimization/<as-of date>/.
4. Upload, only after every step above succeeded, to the one frontend Space
   under <folder>/<as-of date>/: all_forecasts.csv, all_forecasts_metadata.yml,
   portfolio_optimization_plot_data.h5, then optimization_metadata.yml last,
   so the metadata file marks a complete date (see docs/INTERFACE_CONTRACT.md).

Network, S3, and model training are passed in as dependencies so the
orchestration can be tested without them.
"""

import os
from dataclasses import dataclass
from typing import Callable, Protocol
import numpy as np
import pandas as pd
from fuzzy_system_research.config import TimeSeriesConfig
from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.data.prices import last_n_sessions, pivot_close_wide
from fuzzy_system_research.data.ticker_download_manager import TickerDownloadManager
from fuzzy_system_research.forecasting.ticker_predict import (
    ArimaWalkForward,
    ForecastTrainer,
    load_or_train_forecasts,
)
from fuzzy_system_research.forecasting.metrics import (
    build_forecast_metadata,
    write_forecast_metadata,
)
from fuzzy_system_research.optimization.portfolio import get_risk_free_rate
from fuzzy_system_research.optimization.workflow import (
    build_optimization_metadata,
    run_portfolio_optimization,
    write_optimization_metadata,
    write_plot_data_h5,
)
from fuzzy_system_research.storage.layout import (
    ALL_FORECASTS_FILENAME,
    ALL_FORECASTS_METADATA_FILENAME,
    OPTIMIZATION_METADATA_FILENAME,
    OPTIMIZATION_PLOT_DATA_FILENAME,
    OUTPUT_ROOT,
    PORTFOLIO_OPTIMIZATION,
    TIME_SERIES,
    as_of_date_str,
    ensure_parent_dir,
    local_output_path,
    object_key,
)
from fuzzy_system_research.storage.s3_uploader import BUCKET_NAME_ENV, S3Uploader

# Trading sessions of price data the portfolio optimization uses (one year).
OPTIMIZATION_LOOKBACK_SESSIONS = 252
# Extra sessions requested from the API beyond required_sessions(), so that an
# off-by-one in counting sessions, or a session missing from the data, does
# not leave the run short.
PRICE_DOWNLOAD_MARGIN_SESSIONS = 5


def required_sessions(ts_config: TimeSeriesConfig) -> int:
    """
    Trading sessions every run needs: enough for both the optimization and
    the walk-forward ARIMA (TimeSeriesConfig.arima_window_sessions),
    whichever is larger. Pure function.
    """
    return max(OPTIMIZATION_LOOKBACK_SESSIONS, ts_config.arima_window_sessions)


def price_download_sessions(ts_config: TimeSeriesConfig) -> int:
    """
    Trading sessions requested from the API: required_sessions() plus
    PRICE_DOWNLOAD_MARGIN_SESSIONS. Pure
    function.
    """
    return required_sessions(ts_config) + PRICE_DOWNLOAD_MARGIN_SESSIONS


PRICE_DOWNLOAD_FOLDER = os.path.join("input", "annual")

# Environment variables holding the names of the two top-level folders inside
# the one frontend Space. Dev and production share the Space and differ only
# in these folder names.
PORTFOLIO_FOLDER_ENV = "FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER"
TIME_SERIES_FOLDER_ENV = "FSF_FRONT_END_TIME_SERIES_FOLDER"


class Uploader(Protocol):
    """
    Anything that uploads a local file to a full object key in the frontend
    Space, e.g. S3Uploader.
    """

    def upload_file(self, local_filename: str, object_key: str) -> None: ...


@dataclass(frozen=True)
class PipelineResult:
    """
    What a run produced: the as-of date, the local files written, whether
    ARIMA models were trained (False when the cached forecasts were reused),
    and the object keys uploaded, in upload order (empty when uploads were
    skipped).
    """

    as_of: str
    forecasts_path: str
    forecasts_metadata_path: str
    plot_data_path: str
    metadata_path: str
    trained_arima: bool
    uploaded: tuple[str, ...]


def _upload_folders() -> tuple[str, str]:
    """
    Check the environment for everything an upload needs, and return the
    time series and portfolio optimization folder names. Raises RuntimeError
    naming every variable that is not set: the Space name and both folders.
    """
    envs = (BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV)
    values = {env: os.getenv(env) for env in envs}
    missing = [env for env, value in values.items() if not value]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s) {', '.join(missing)}; set them in .env "
            "or run with --no-upload."
        )
    return values[TIME_SERIES_FOLDER_ENV], values[PORTFOLIO_FOLDER_ENV]


def run_pipeline(
    upload: bool = True,
    retrain_arima: bool = False,
    refresh_data: bool = False,
    arima_workers: int | None = None,
    *,
    ts_config: TimeSeriesConfig,
    tdm: TickerDownloadManager | None = None,
    trainer: ForecastTrainer | None = None,
    uploader: Uploader | None = None,
    risk_free_rate_getter: Callable[[DateManager], dict] = get_risk_free_rate,
    dm: DateManager | None = None,
    output_root: str = OUTPUT_ROOT,
    rng: np.random.Generator | None = None,
) -> PipelineResult:
    """
    Run the whole backend: prices, ARIMA forecasts, portfolio optimization,
    and uploads. Nothing is uploaded unless every computation succeeded. If
    an upload fails, the error propagates immediately; because
    optimization_metadata.yml is uploaded last, a partially uploaded date is
    never marked complete, and re-running uploads it again.

    The Space name (FSF_FRONT_END_BUCKET_NAME) and the two top-level folder
    names are read from the environment (load .env before calling). When
    upload is True they, and the endpoint (via S3Uploader), are checked
    before any work starts, so a configuration error fails fast instead of
    after a long ARIMA run.

    Parameters
    ----------
    upload : bool, optional
        If False, write the local output files but upload nothing.

    retrain_arima : bool, optional
        If True, retrain ARIMA models even if the as-of date's forecasts CSV
        already exists, and overwrite it.

    refresh_data : bool, optional
        If True, download prices again even if today's download is cached.
        Even when False, a cached download with fewer than
        required_sessions() sessions (e.g. one written before the ARIMA
        window grew) is downloaded again once.

    arima_workers : int, optional
        Number of worker processes for ARIMA training. Defaults to None,
        which means half the logical CPUs (see default_worker_count() in
        forecasting/ticker_predict.py). Ignored when trainer is given.

    ts_config : TimeSeriesConfig
        Time series settings from the configuration file: the training
        window length and the number of scored backtest days. Required; it
        has no default, so the configuration file is the one source of
        truth.

    tdm, trainer, uploader, risk_free_rate_getter, dm : optional
        Dependencies, defaulting to the real implementations. Tests pass
        doubles here to avoid network, S3, and model training.

    output_root : str, optional
        Root of the local output folder. Defaults to "output".

    rng : np.random.Generator, optional
        Random number generator for the Monte Carlo simulation.

    Returns
    -------
    PipelineResult
        Summary of the run.
    """
    if upload:
        time_series_folder, portfolio_folder = _upload_folders()
        uploader = uploader if uploader is not None else S3Uploader()

    # 1. Prices
    n_required = required_sessions(ts_config)
    n_download = price_download_sessions(ts_config)
    arima_sessions = ts_config.arima_window_sessions
    tdm = tdm if tdm is not None else TickerDownloadManager(PRICE_DOWNLOAD_FOLDER)
    dm = dm if dm is not None else DateManager()
    long_df, start_date, end_date = tdm.get_latest_tickers(
        days_in_past=n_download, use_cache=not refresh_data
    )
    wide_df = pivot_close_wide(long_df)
    if len(wide_df) < n_required and not refresh_data:
        print(
            f"Cached prices cover {len(wide_df)} sessions, fewer than the "
            f"{n_required} needed; downloading again."
        )
        long_df, start_date, end_date = tdm.get_latest_tickers(
            days_in_past=n_download, use_cache=False
        )
        wide_df = pivot_close_wide(long_df)
    if len(wide_df) < n_required:
        raise ValueError(
            f"Downloaded prices cover {len(wide_df)} sessions, fewer than the "
            f"{n_required} needed for ARIMA ({arima_sessions}) and "
            f"the optimization ({OPTIMIZATION_LOOKBACK_SESSIONS})."
        )
    print(f"Loaded data from {start_date} to {end_date}")
    as_of = as_of_date_str(wide_df.index)
    print(f"Results will be stored under {as_of}/")

    # 2. ARIMA forecasts (cached per as-of date)
    forecasts_path = local_output_path(
        TIME_SERIES, as_of, ALL_FORECASTS_FILENAME, output_root
    )
    arima_window_df = last_n_sessions(wide_df, arima_sessions)
    if trainer is None:
        trainer = ArimaWalkForward(
            training_window_returns=ts_config.training_window_returns,
            n_workers=arima_workers,
        )
    trained_arima = load_or_train_forecasts(
        arima_window_df,
        forecasts_path,
        retrain=retrain_arima,
        trainer=trainer,
    )

    # 2b. Forecast metrics, from the CSV exactly as written (also for cached
    # forecasts). The live forecast's training window is the last window.
    forecasts_df = pd.read_csv(forecasts_path, index_col="pred_date", parse_dates=True)
    window_start, window_end = ArimaWalkForward.training_window_start_end(
        arima_window_df.index, ts_config.training_window_returns
    )[-1]
    forecasts_metadata_path = ensure_parent_dir(
        local_output_path(
            TIME_SERIES, as_of, ALL_FORECASTS_METADATA_FILENAME, output_root
        )
    )
    write_forecast_metadata(
        build_forecast_metadata(
            forecasts_df,
            tdm.tickers,
            window_start,
            window_end,
            eval_days=ts_config.evaluation_window_length,
            training_window_returns=ts_config.training_window_returns,
        ),
        forecasts_metadata_path,
    )
    print(f"Saved {forecasts_metadata_path}")

    # 3. Portfolio optimization
    risk_free_rate_data = risk_free_rate_getter(dm)
    print(risk_free_rate_data)
    result = run_portfolio_optimization(
        last_n_sessions(wide_df, OPTIMIZATION_LOOKBACK_SESSIONS),
        risk_free_rate_data["daily_risk_free_rate"],
        rng=rng,
    )
    plot_data_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, OPTIMIZATION_PLOT_DATA_FILENAME, output_root
        )
    )
    write_plot_data_h5(result, plot_data_path)
    print(f"Saved {plot_data_path}")
    metadata_path = ensure_parent_dir(
        local_output_path(
            PORTFOLIO_OPTIMIZATION, as_of, OPTIMIZATION_METADATA_FILENAME, output_root
        )
    )
    write_optimization_metadata(
        build_optimization_metadata(result, tdm.tickers, risk_free_rate_data),
        metadata_path,
    )
    print(f"Saved {metadata_path}")

    # 4. Uploads, optimization metadata last (it marks the date as complete)
    uploaded: list[str] = []
    if not upload:
        print("Skipping uploads (--no-upload).")
    else:
        for local_path, folder, filename in (
            (forecasts_path, time_series_folder, ALL_FORECASTS_FILENAME),
            (
                forecasts_metadata_path,
                time_series_folder,
                ALL_FORECASTS_METADATA_FILENAME,
            ),
            (plot_data_path, portfolio_folder, OPTIMIZATION_PLOT_DATA_FILENAME),
            (metadata_path, portfolio_folder, OPTIMIZATION_METADATA_FILENAME),
        ):
            key = object_key(folder, as_of, filename)
            uploader.upload_file(local_path, key)
            uploaded.append(key)

    return PipelineResult(
        as_of=as_of,
        forecasts_path=forecasts_path,
        forecasts_metadata_path=forecasts_metadata_path,
        plot_data_path=plot_data_path,
        metadata_path=metadata_path,
        trained_arima=trained_arima,
        uploaded=tuple(uploaded),
    )
