"""
Tests for ArimaWalkForward.train_arima_models() date handling.

ARIMA fitting is replaced by a stub that keeps the real date logic
(ArimaModels.next_trading_day, i.e. the NYSE calendar) and
returns a dummy forecast, so these tests run fast, without Dask, and check
only how forecast rows are dated and assembled. In-memory only.
"""

import numpy as np
import pandas as pd
import pytest

from fuzzy_system_research.data.date_manager import DateManager, nyse_holidays
from fuzzy_system_research.forecasting import ticker_predict
from fuzzy_system_research.forecasting.arima_models import ArimaModels
from fuzzy_system_research.forecasting.ticker_predict import ArimaWalkForward


def stub_order(last_date: pd.Timestamp) -> tuple[int, int]:
    """A (p, q) that varies from window to window, never (0, 0)."""
    return (1, 0) if last_date.day % 2 else (0, 1)


class StubArimaModels:
    """
    Real pred_date logic, dummy forecast (last observed price), and an order
    that depends on the window (stub_order). Records every series passed to
    fit() in StubArimaModels.fitted, and (ticker, pred_date, order) for every
    forecast in StubArimaModels.orders.
    """

    fitted: list[pd.Series] = []
    orders: list[tuple[str, pd.Timestamp, tuple[int, int]]] = []

    def __init__(self, n_workers=6):
        self._dates = ArimaModels(n_workers=n_workers)

    def fit(self, original_ts, max_p=5, max_q=5, train_len=90):
        StubArimaModels.fitted.append(original_ts)
        pred_date = self._dates.next_trading_day(original_ts.index[-1])
        order = stub_order(original_ts.index[-1])
        StubArimaModels.orders.append((original_ts.name, pred_date, order))
        return pred_date, float(original_ts.dropna().iloc[-1]), order


@pytest.fixture(autouse=True)
def stub_arima(monkeypatch):
    StubArimaModels.fitted = []
    StubArimaModels.orders = []
    monkeypatch.setattr(ticker_predict, "ArimaModels", StubArimaModels)


def nyse_prices(last_session: str, n_sessions: int = 40) -> pd.DataFrame:
    """Wide prices on the n_sessions NYSE sessions ending at last_session (17:00)."""
    sessions = pd.date_range(
        end=pd.Timestamp(last_session), periods=n_sessions, freq=DateManager().cbd
    )
    index = pd.DatetimeIndex([d.replace(hour=17) for d in sessions])
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "AAA": 100 + rng.normal(0, 1, n_sessions).cumsum(),
            "BBB": 50 + rng.normal(0, 1, n_sessions).cumsum(),
        },
        index=index,
    )


@pytest.mark.parametrize(
    "last_session, expected_final",
    [
        ("2026-04-02", "2026-04-06 17:00"),  # Skips Good Friday (NYSE closed).
        ("2026-10-09", "2026-10-12 17:00"),  # Columbus Day: NYSE open.
        ("2026-11-25", "2026-11-27 17:00"),  # Skips Thanksgiving.
        ("2026-10-14", "2026-10-15 17:00"),  # Ordinary weekday.
    ],
)
def test_final_row_is_next_nyse_trading_day(last_session, expected_final):
    forecasts = ArimaWalkForward().train_arima_models(nyse_prices(last_session))
    assert forecasts.index[-1] == pd.Timestamp(expected_final)


def test_final_row_has_forecasts_but_no_actuals():
    forecasts = ArimaWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    last = forecasts.iloc[-1]
    assert last[["AAA", "BBB"]].isna().all()
    assert last[["AAA_arima", "BBB_arima"]].notna().all()
    # Every earlier row has an actual price to compare against.
    assert forecasts.iloc[:-1][["AAA", "BBB"]].notna().all().all()


def test_every_pred_date_is_a_unique_nyse_session_after_the_window():
    prices = nyse_prices("2026-04-02")
    forecasts = ArimaWalkForward().train_arima_models(prices)
    dates = forecasts.index.normalize()
    assert forecasts.index.is_unique
    assert forecasts.index.is_monotonic_increasing
    assert (dates.dayofweek < 5).all()
    assert not dates.isin(nyse_holidays()).any()
    # The first forecast comes after the first training window, not inside it.
    assert forecasts.index[0] > prices.index[0]


# Training windows are resampled to NYSE sessions, not pandas "B" weekdays.
# Data ending 2026-04-02 with 40 sessions starts in early February, so the
# windows span Presidents Day (2026-02-16), an NYSE holiday on a Monday.

PRESIDENTS_DAY = pd.Timestamp("2026-02-16")


def test_training_windows_skip_nyse_holidays():
    prices = nyse_prices("2026-04-02")
    assert prices.index[0].normalize() < PRESIDENTS_DAY  # The data spans it.
    ArimaWalkForward().train_arima_models(prices)
    assert StubArimaModels.fitted  # fit() was called.
    spans_holiday = 0
    for ts in StubArimaModels.fitted:
        dates = ts.index.normalize()
        assert not dates.isin(nyse_holidays()).any()
        assert ts.notna().all()  # Complete data: no inserted missing rows.
        if dates[0] < PRESIDENTS_DAY < dates[-1]:
            spans_holiday += 1
    assert spans_holiday > 0  # At least one window actually crossed it.


def test_training_windows_use_the_nyse_session_frequency():
    ArimaWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    for ts in StubArimaModels.fitted:
        assert isinstance(ts.index.freq, pd.offsets.CustomBusinessDay)
        # Consecutive entries are consecutive NYSE sessions.
        expected = pd.date_range(ts.index[0], periods=len(ts), freq=DateManager().cbd)
        assert ts.index.equals(expected)


def test_missing_trading_session_still_shows_as_missing():
    # A session absent from the data (e.g. a download gap) must remain
    # visible to the model as a missing value rather than silently vanish.
    prices = nyse_prices("2026-04-02")
    gap = pd.Timestamp("2026-03-10 17:00")  # An ordinary trading Tuesday.
    ArimaWalkForward().train_arima_models(prices.drop(index=gap))
    windows_with_gap = [ts for ts in StubArimaModels.fitted if gap in ts.index]
    assert windows_with_gap
    assert all(np.isnan(ts.loc[gap]) for ts in windows_with_gap)


def test_statsmodels_arima_fits_and_forecasts_on_nyse_frequency():
    # The real model (no Dask): an ARIMA on log returns indexed by NYSE
    # sessions must fit and produce a finite one-step forecast.
    import warnings

    import statsmodels.tsa.api as tsa

    prices = nyse_prices("2026-04-02", n_sessions=30)["AAA"]
    window = prices.asfreq(DateManager().cbd)
    log_returns = ArimaModels().log_diff_prep(window).iloc[1:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tsa.ARIMA(endog=log_returns, order=(1, 0, 0)).fit()
        forecast = model.forecast(steps=1)
    assert len(forecast) == 1
    assert np.isfinite(forecast.iloc[0])


# ARIMA order columns: T_arima_p and T_arima_q


def test_order_columns_follow_each_forecast_column():
    forecasts = ArimaWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    assert list(forecasts.columns) == [
        "AAA_arima", "AAA_arima_p", "AAA_arima_q", "AAA",
        "BBB_arima", "BBB_arima_p", "BBB_arima_q", "BBB",
    ]


def test_order_columns_hold_the_order_of_each_rows_model():
    forecasts = ArimaWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    assert StubArimaModels.orders
    for ticker, pred_date, (p, q) in StubArimaModels.orders:
        assert forecasts.loc[pred_date, f"{ticker}_arima_p"] == p
        assert forecasts.loc[pred_date, f"{ticker}_arima_q"] == q
    # Both stub orders occur, so a constant column would not pass.
    assert set(forecasts["AAA_arima_p"]) == {0, 1}


def test_order_columns_are_integers_on_every_row_including_the_last():
    forecasts = ArimaWalkForward().train_arima_models(nyse_prices("2026-04-02"))
    order_columns = [c for c in forecasts.columns if c.endswith(("_arima_p", "_arima_q"))]
    assert len(order_columns) == 4
    for column in order_columns:
        assert pd.api.types.is_integer_dtype(forecasts[column])
        assert forecasts[column].notna().all()


def test_order_columns_round_trip_through_csv_as_integers(tmp_path):
    # The uploaded CSV is written by load_or_train_forecasts().
    path = str(tmp_path / "all_forecasts.csv")
    ticker_predict.load_or_train_forecasts(
        nyse_prices("2026-04-02"), path, trainer=ArimaWalkForward()
    )
    loaded = pd.read_csv(path, index_col="pred_date", parse_dates=True)
    assert pd.api.types.is_integer_dtype(loaded["AAA_arima_p"])
    assert pd.api.types.is_integer_dtype(loaded["BBB_arima_q"])
    # Forecast columns are still found by the "_arima" suffix alone.
    assert [c for c in loaded.columns if c.endswith("_arima")] == ["AAA_arima", "BBB_arima"]


# ArimaModels.select_order (pure; no model fitting)


def order_results(rows: dict[tuple[int, int], tuple[float, float]]) -> pd.DataFrame:
    """train_arima()-shaped results: (p, q) index, rmse and mean_bic columns."""
    index = pd.MultiIndex.from_tuples(list(rows), names=["p", "q"])
    rmse, bic = zip(*rows.values())
    return pd.DataFrame({"rmse": rmse, "mean_bic": bic}, index=index)


def test_select_order_picks_best_average_rank_and_returns_ints():
    results = order_results(
        {
            (0, 1): (0.030, -50.0),  # Ranks: rmse 3, bic 2 -> average 2.5
            (1, 0): (0.010, -40.0),  # Ranks: rmse 1, bic 3 -> average 2.0
            (1, 1): (0.020, -60.0),  # Ranks: rmse 2, bic 1 -> average 1.5 (best)
        }
    )
    # The best RMSE alone, (1, 0), does not win: (1, 1) has the best average.
    assert ArimaModels().select_order(results) == (1, 1)
    p, q = ArimaModels().select_order(results)
    assert type(p) is int and type(q) is int


def test_select_order_weighs_rmse_and_bic_equally():
    results = order_results(
        {
            (0, 1): (0.010, -10.0),  # Best RMSE, worst BIC -> (1 + 3) / 2 = 2.0
            (1, 0): (0.030, -30.0),  # Worst RMSE, best BIC -> (3 + 1) / 2 = 2.0
            (1, 1): (0.020, -20.0),  # Middle on both -> 2.0
        }
    )
    # A three-way tie goes to the first row.
    assert ArimaModels().select_order(results) == (0, 1)
