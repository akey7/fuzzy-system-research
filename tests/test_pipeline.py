"""
Orchestration tests for fuzzy_system_research.pipeline.run_pipeline.

Downloads, ARIMA training, the risk-free rate lookup, and S3 are replaced by
in-memory doubles; files are written only under pytest's tmp_path. Nothing
here touches the network, a bucket, or the real input/ and output/ folders.
"""

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import SYNTHETIC_TICKERS, make_long_prices
from fuzzy_system_research.data.prices import pivot_close_wide
from fuzzy_system_research.forecasting.ticker_predict import (
    ARIMA_WINDOW_SESSIONS,
    EVALUATION_DAYS,
    TRAINING_WINDOW_RETURNS,
)
from fuzzy_system_research.pipeline import (
    OPTIMIZATION_LOOKBACK_SESSIONS,
    PORTFOLIO_FOLDER_ENV,
    TIME_SERIES_FOLDER_ENV,
    run_pipeline,
)
from fuzzy_system_research.storage.s3_uploader import BUCKET_NAME_ENV

SPACE = "test-space"
TS_FOLDER = "test-time-series"
PO_FOLDER = "test-portfolio"
RF_DATA = {
    "risk_free_rate": 4.05,
    "daily_risk_free_rate": 4.05 / 252,
    "risk_free_rate_date": "2026-10-08 00:00:00",
}
STUB_UPLIFT = 1.01  # Fake ARIMA forecast = 1 % above the previous close.


class FakeTickerDownloadManager:
    def __init__(self, long_df: pd.DataFrame):
        self.long_df = long_df
        self.tickers = list(SYNTHETIC_TICKERS)
        self.calls: list[dict] = []

    def get_latest_tickers(self, days_in_past=40, use_cache=True):
        self.calls.append({"days_in_past": days_in_past, "use_cache": use_cache})
        return self.long_df, "start", "end"


class FakeTrainer:
    """
    Produces a forecasts table shaped like ArimaWalkForward's: one row per
    window end's next session plus the live row, with T_arima, T_naive (the
    previous close) and T (the actual, empty on the live row).
    """

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[pd.DataFrame] = []

    def train_arima_models(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(df)
        if self.fail:
            raise RuntimeError("ARIMA failed")
        window_ends = df.index[TRAINING_WINDOW_RETURNS:]
        pred_dates = list(df.index[TRAINING_WINDOW_RETURNS + 1 :]) + [
            df.index[-1] + pd.offsets.BDay(1)
        ]
        out = pd.DataFrame(index=pd.DatetimeIndex(pred_dates, name="pred_date"))
        for t in df.columns:
            naive = df.loc[window_ends, t].to_numpy()
            out[f"{t}_arima"] = STUB_UPLIFT * naive
            out[f"{t}_naive"] = naive
            out[t] = df[t].reindex(out.index).to_numpy()
        return out


class FakeUploader:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def upload_file(self, local_filename, object_key):
        # Every file must already be written when it is uploaded.
        assert os.path.isfile(local_filename), local_filename
        self.calls.append((local_filename, object_key))


@pytest.fixture
def upload_env(monkeypatch):
    monkeypatch.setenv(BUCKET_NAME_ENV, SPACE)
    monkeypatch.setenv(TIME_SERIES_FOLDER_ENV, TS_FOLDER)
    monkeypatch.setenv(PORTFOLIO_FOLDER_ENV, PO_FOLDER)


@pytest.fixture
def deps(tmp_path):
    """Fresh doubles plus the keyword arguments run_pipeline needs."""
    tdm = FakeTickerDownloadManager(make_long_prices())
    trainer = FakeTrainer()
    uploader = FakeUploader()
    kwargs = dict(
        tdm=tdm,
        trainer=trainer,
        uploader=uploader,
        risk_free_rate_getter=lambda dm: dict(RF_DATA),
        dm=object(),
        output_root=str(tmp_path / "output"),
        rng=np.random.default_rng(0),
    )
    return tdm, trainer, uploader, kwargs


def wide_prices() -> pd.DataFrame:
    return pivot_close_wide(make_long_prices())


def expected_as_of() -> str:
    return wide_prices().index[-1].strftime("%Y-%m-%d")


def test_full_run_writes_four_files_and_uploads_in_order(upload_env, deps):
    _, _, uploader, kwargs = deps
    result = run_pipeline(**kwargs)
    as_of = expected_as_of()
    assert result.as_of == as_of
    for path in (
        result.forecasts_path,
        result.forecasts_metadata_path,
        result.plot_data_path,
        result.metadata_path,
    ):
        assert os.path.isfile(path)
        assert f"{os.sep}{as_of}{os.sep}" in path
    assert [key for _, key in uploader.calls] == [
        f"{TS_FOLDER}/{as_of}/all_forecasts.csv",
        f"{TS_FOLDER}/{as_of}/all_forecasts_metadata.yml",
        f"{PO_FOLDER}/{as_of}/portfolio_optimization_plot_data.h5",
        f"{PO_FOLDER}/{as_of}/optimization_metadata.yml",
    ]
    assert result.uploaded == tuple(key for _, key in uploader.calls)
    assert result.trained_arima is True


def test_keys_never_contain_the_space_name(upload_env, deps):
    # The Space is the S3 bucket; it must not also appear inside a key.
    _, _, uploader, kwargs = deps
    run_pipeline(**kwargs)
    assert all(SPACE not in key for _, key in uploader.calls)


def test_local_layout_mirrors_object_keys(upload_env, deps, tmp_path):
    # output/<kind>/<date>/<file> and <folder>/<date>/<file> share <date>/<file>.
    _, _, uploader, kwargs = deps
    run_pipeline(**kwargs)
    kinds_and_folders = [
        ("time_series", TS_FOLDER),
        ("time_series", TS_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
        ("portfolio_optimization", PO_FOLDER),
    ]
    for (local, key), (kind, folder) in zip(uploader.calls, kinds_and_folders):
        relative = os.path.relpath(local, tmp_path / "output" / kind)
        assert f"{folder}/" + relative.replace(os.sep, "/") == key


def test_metadata_date_to_equals_folder_date(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with open(result.metadata_path) as f:
        metadata = yaml.safe_load(f)
    assert metadata["date_updated"]["date_to"] == result.as_of
    assert metadata["tickers"] == SYNTHETIC_TICKERS


def test_single_download_of_lookback_sessions_with_cache(upload_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(**kwargs)
    assert tdm.calls == [
        {"days_in_past": OPTIMIZATION_LOOKBACK_SESSIONS, "use_cache": True}
    ]


def test_refresh_data_bypasses_download_cache(upload_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(refresh_data=True, **kwargs)
    assert tdm.calls[-1]["use_cache"] is False


def test_arima_trains_on_last_41_sessions(upload_env, deps):
    assert ARIMA_WINDOW_SESSIONS == 41
    _, trainer, _, kwargs = deps
    run_pipeline(**kwargs)
    (window,) = trainer.calls
    pd.testing.assert_frame_equal(window, wide_prices().iloc[-ARIMA_WINDOW_SESSIONS:])


def test_cached_forecasts_skip_training(upload_env, deps):
    _, trainer, uploader, kwargs = deps
    first = run_pipeline(**kwargs)
    with open(first.forecasts_path) as f:
        cached = f.read()
    trainer.calls.clear()
    second = run_pipeline(**kwargs)
    assert trainer.calls == []
    assert second.trained_arima is False
    with open(second.forecasts_path) as f:
        assert f.read() == cached
    assert len(uploader.calls) == 8  # The cached file is still uploaded.


def test_retrain_arima_ignores_cache(upload_env, deps):
    _, trainer, _, kwargs = deps
    first = run_pipeline(**kwargs)
    with open(first.forecasts_path, "w") as f:
        f.write("stale")
    trainer.calls.clear()
    second = run_pipeline(retrain_arima=True, **kwargs)
    assert len(trainer.calls) == 1
    assert second.trained_arima is True
    with open(second.forecasts_path) as f:
        assert f.read() != "stale"


def test_no_upload_writes_files_without_uploading_or_env(monkeypatch, deps):
    for env in (BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV):
        monkeypatch.delenv(env, raising=False)
    _, _, uploader, kwargs = deps
    result = run_pipeline(upload=False, **kwargs)
    assert uploader.calls == []
    assert result.uploaded == ()
    assert os.path.isfile(result.metadata_path)
    assert os.path.isfile(result.forecasts_metadata_path)


@pytest.mark.parametrize(
    "missing", [BUCKET_NAME_ENV, TIME_SERIES_FOLDER_ENV, PORTFOLIO_FOLDER_ENV]
)
def test_missing_upload_env_fails_before_any_work(monkeypatch, upload_env, deps, missing):
    monkeypatch.delenv(missing)
    tdm, trainer, uploader, kwargs = deps
    with pytest.raises(RuntimeError, match=missing):
        run_pipeline(**kwargs)
    assert tdm.calls == []
    assert trainer.calls == []
    assert uploader.calls == []


def test_arima_failure_uploads_nothing(upload_env, deps):
    _, _, uploader, kwargs = deps
    kwargs["trainer"] = FakeTrainer(fail=True)
    with pytest.raises(RuntimeError, match="ARIMA failed"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_upload_failure_stops_before_completion_marker(upload_env, deps):
    # If an upload fails part way, the error propagates and
    # optimization_metadata.yml (the completion marker) is never uploaded,
    # so the frontend does not list the incomplete date.
    _, _, _, kwargs = deps

    class FailOnH5Uploader(FakeUploader):
        def upload_file(self, local_filename, object_key):
            if object_key.endswith(".h5"):
                raise ConnectionError("upload failed")
            super().upload_file(local_filename, object_key)

    uploader = FailOnH5Uploader()
    kwargs["uploader"] = uploader
    with pytest.raises(ConnectionError):
        run_pipeline(**kwargs)
    keys = [key for _, key in uploader.calls]
    assert [k.rsplit("/", 1)[-1] for k in keys] == [
        "all_forecasts.csv",
        "all_forecasts_metadata.yml",
    ]


def test_optimization_failure_uploads_nothing_but_keeps_forecast_cache(
    upload_env, deps
):
    _, _, uploader, kwargs = deps

    def failing_rf(dm):
        raise RuntimeError("risk-free rate unavailable")

    kwargs["risk_free_rate_getter"] = failing_rf
    with pytest.raises(RuntimeError, match="risk-free rate"):
        run_pipeline(**kwargs)
    assert uploader.calls == []
    forecasts = os.path.join(
        kwargs["output_root"], "time_series", expected_as_of(), "all_forecasts.csv"
    )
    assert os.path.isfile(forecasts)  # A retry will not retrain ARIMA.


# all_forecasts_metadata.yml


def load_forecasts_metadata(result) -> dict:
    with open(result.forecasts_metadata_path) as f:
        return yaml.safe_load(f)


def test_forecasts_metadata_windows_match_the_data(upload_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    meta = load_forecasts_metadata(result)
    window = wide_prices().iloc[-ARIMA_WINDOW_SESSIONS:]
    fw = meta["forecast_window"]
    assert fw["training_window_end_date"] == result.as_of
    assert fw["training_window_start_date"] == window.index[
        -(TRAINING_WINDOW_RETURNS + 1)
    ].strftime("%Y-%m-%d")
    forecasts = pd.read_csv(result.forecasts_path, index_col="pred_date", parse_dates=True)
    assert fw["forecast_date"] == forecasts.index[-1].strftime("%Y-%m-%d")
    assert fw["forecast_date"] > fw["training_window_end_date"]
    assert fw["training_window_returns"] == 20
    assert fw["horizon_trading_days"] == 1
    ev = meta["evaluation_window"]
    assert ev["target_trading_days"] == EVALUATION_DAYS
    assert ev["end_date"] == result.as_of  # The last day with an actual close.


def test_forecasts_metadata_has_lowercase_tickers_scored_on_20_days(upload_env, deps):
    _, _, _, kwargs = deps
    meta = load_forecasts_metadata(run_pipeline(**kwargs))
    assert set(meta["ticker_metrics"]) == {t.lower() for t in SYNTHETIC_TICKERS}
    for metrics in meta["ticker_metrics"].values():
        assert metrics["n_evaluated"] == EVALUATION_DAYS
        # The fake ARIMA is 1 % above the previous close; naive is the close.
        assert metrics["relative_mae"] == pytest.approx(
            metrics["arima"]["mae"] / metrics["naive"]["mae"]
        )
        assert metrics["skill"] == pytest.approx(1 - metrics["relative_mae"])


def test_forecasts_metadata_keys_are_lowercase_snake_case(upload_env, deps):
    _, _, _, kwargs = deps
    meta = load_forecasts_metadata(run_pipeline(**kwargs))

    def keys(d):
        for k, v in d.items():
            yield k
            if isinstance(v, dict):
                yield from keys(v)

    assert all(k == k.lower() and " " not in k for k in keys(meta))


def test_forecasts_metadata_is_written_for_cached_forecasts(upload_env, deps):
    _, trainer, _, kwargs = deps
    run_pipeline(**kwargs)
    trainer.calls.clear()
    second = run_pipeline(**kwargs)  # Reuses the cached CSV.
    assert trainer.calls == []
    assert load_forecasts_metadata(second)["ticker_metrics"]
