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
from fuzzy_system_research.pipeline import (
    OPTIMIZATION_LOOKBACK_SESSIONS,
    PORTFOLIO_SPACE_ENV,
    TIME_SERIES_SPACE_ENV,
    run_pipeline,
)

TS_BUCKET = "test-time-series"
PO_BUCKET = "test-portfolio"
RF_DATA = {
    "risk_free_rate": 4.05,
    "daily_risk_free_rate": 4.05 / 252,
    "risk_free_rate_date": "2026-10-08 00:00:00",
}


class FakeTickerDownloadManager:
    def __init__(self, long_df: pd.DataFrame):
        self.long_df = long_df
        self.tickers = list(SYNTHETIC_TICKERS)
        self.calls: list[dict] = []

    def get_latest_tickers(self, days_in_past=40, use_cache=True):
        self.calls.append({"days_in_past": days_in_past, "use_cache": use_cache})
        return self.long_df, "start", "end"


class FakeTrainer:
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[pd.DataFrame] = []

    def train_arima_models(self, df: pd.DataFrame) -> pd.DataFrame:
        self.calls.append(df)
        if self.fail:
            raise RuntimeError("ARIMA failed")
        next_day = df.index[-1] + pd.offsets.BDay(1)
        out = pd.DataFrame(
            {f"{t}_arima": [1.0] for t in df.columns}, index=[next_day]
        )
        out.index.name = "pred_date"
        return out


class FakeUploader:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []

    def upload_file(self, local_filename, bucket_name, remote_filename):
        # Every file must already be written when it is uploaded.
        assert os.path.isfile(local_filename), local_filename
        self.calls.append((local_filename, bucket_name, remote_filename))


@pytest.fixture
def bucket_env(monkeypatch):
    monkeypatch.setenv(TIME_SERIES_SPACE_ENV, TS_BUCKET)
    monkeypatch.setenv(PORTFOLIO_SPACE_ENV, PO_BUCKET)


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


def expected_as_of() -> str:
    return pivot_close_wide(make_long_prices()).index[-1].strftime("%Y-%m-%d")


def test_full_run_writes_three_files_and_uploads_in_order(bucket_env, deps):
    tdm, trainer, uploader, kwargs = deps
    result = run_pipeline(**kwargs)
    as_of = expected_as_of()
    assert result.as_of == as_of
    for path in (result.forecasts_path, result.plot_data_path, result.metadata_path):
        assert os.path.isfile(path)
        assert f"{os.sep}{as_of}{os.sep}" in path
    assert [(bucket, key) for _, bucket, key in uploader.calls] == [
        (TS_BUCKET, f"{as_of}/all_forecasts.csv"),
        (PO_BUCKET, f"{as_of}/portfolio_optimization_plot_data.h5"),
        (PO_BUCKET, f"{as_of}/optimization_metadata.yml"),
    ]
    assert result.uploaded == tuple((b, k) for _, b, k in uploader.calls)
    assert result.trained_arima is True


def test_local_layout_mirrors_bucket_keys(bucket_env, deps, tmp_path):
    _, _, uploader, kwargs = deps
    run_pipeline(**kwargs)
    kinds = ["time_series", "portfolio_optimization", "portfolio_optimization"]
    for (local, _, key), kind in zip(uploader.calls, kinds):
        relative = os.path.relpath(local, tmp_path / "output" / kind)
        assert relative.replace(os.sep, "/") == key


def test_metadata_date_to_equals_folder_date(bucket_env, deps):
    _, _, _, kwargs = deps
    result = run_pipeline(**kwargs)
    with open(result.metadata_path) as f:
        metadata = yaml.safe_load(f)
    assert metadata["date_updated"]["date_to"] == result.as_of
    assert metadata["tickers"] == SYNTHETIC_TICKERS


def test_single_download_of_lookback_sessions_with_cache(bucket_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(**kwargs)
    assert tdm.calls == [
        {"days_in_past": OPTIMIZATION_LOOKBACK_SESSIONS, "use_cache": True}
    ]


def test_refresh_data_bypasses_download_cache(bucket_env, deps):
    tdm, _, _, kwargs = deps
    run_pipeline(refresh_data=True, **kwargs)
    assert tdm.calls[-1]["use_cache"] is False


def test_arima_trains_on_last_40_sessions(bucket_env, deps):
    _, trainer, _, kwargs = deps
    run_pipeline(**kwargs)
    wide = pivot_close_wide(make_long_prices())
    (window,) = trainer.calls
    pd.testing.assert_frame_equal(window, wide.iloc[-40:])


def test_cached_forecasts_skip_training(bucket_env, deps):
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
    assert len(uploader.calls) == 6  # The cached file is still uploaded.


def test_retrain_arima_ignores_cache(bucket_env, deps):
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
    monkeypatch.delenv(TIME_SERIES_SPACE_ENV, raising=False)
    monkeypatch.delenv(PORTFOLIO_SPACE_ENV, raising=False)
    _, _, uploader, kwargs = deps
    result = run_pipeline(upload=False, **kwargs)
    assert uploader.calls == []
    assert result.uploaded == ()
    assert os.path.isfile(result.metadata_path)


@pytest.mark.parametrize("missing", [TIME_SERIES_SPACE_ENV, PORTFOLIO_SPACE_ENV])
def test_missing_bucket_env_fails_before_any_work(monkeypatch, bucket_env, deps, missing):
    monkeypatch.delenv(missing)
    tdm, trainer, uploader, kwargs = deps
    with pytest.raises(RuntimeError, match=missing):
        run_pipeline(**kwargs)
    assert tdm.calls == []
    assert trainer.calls == []
    assert uploader.calls == []


def test_arima_failure_uploads_nothing(bucket_env, deps):
    _, _, uploader, kwargs = deps
    kwargs["trainer"] = FakeTrainer(fail=True)
    with pytest.raises(RuntimeError, match="ARIMA failed"):
        run_pipeline(**kwargs)
    assert uploader.calls == []


def test_upload_failure_stops_before_completion_marker(bucket_env, deps):
    # If an upload fails part way, the error propagates and
    # optimization_metadata.yml (the completion marker) is never uploaded,
    # so the frontend does not list the incomplete date.
    _, _, _, kwargs = deps

    class FailOnH5Uploader(FakeUploader):
        def upload_file(self, local_filename, bucket_name, remote_filename):
            if remote_filename.endswith(".h5"):
                raise ConnectionError("upload failed")
            super().upload_file(local_filename, bucket_name, remote_filename)

    uploader = FailOnH5Uploader()
    kwargs["uploader"] = uploader
    with pytest.raises(ConnectionError):
        run_pipeline(**kwargs)
    keys = [key for _, _, key in uploader.calls]
    assert len(keys) == 1 and keys[0].endswith("all_forecasts.csv")


def test_optimization_failure_uploads_nothing_but_keeps_forecast_cache(
    bucket_env, deps
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
