"""
Tests for fuzzy_system_research.forecasting.arima_models: the single
differencing step (to_log_returns) and the rolling-origin evaluation of one
(p, q) candidate (p_d_q_fit), including how failed fits are handled.

p_d_q_fit is called directly (no Dask). Most tests replace statsmodels'
ARIMA with a fake whose fits fail on chosen steps; one test uses the real
model. In-memory only.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.linalg import LinAlgError

from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.forecasting import arima_models
from fuzzy_system_research.forecasting.arima_models import ArimaModels


def session_prices(n: int = 21, seed: int = 0) -> pd.Series:
    """Prices on n consecutive NYSE sessions, with the NYSE frequency set."""
    index = pd.date_range("2026-03-02 17:00", periods=n, freq=DateManager().cbd)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=index)


# to_log_returns: the only differencing step


def test_to_log_returns_is_one_difference_of_log_prices():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    expected = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    np.testing.assert_allclose(returns.to_numpy(), expected)


def test_to_log_returns_drops_the_leading_nan():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    assert len(returns) == len(prices) - 1
    assert returns.index[0] == prices.index[1]
    assert returns.notna().all()


def test_to_log_returns_keeps_index_and_frequency():
    prices = session_prices()
    returns = ArimaModels().to_log_returns(prices)
    assert returns.index.equals(prices.index[1:])
    assert returns.index.freq == prices.index.freq


def test_to_log_returns_keeps_a_missing_session_visible():
    prices = session_prices()
    prices.iloc[5] = np.nan  # A session missing from the data.
    returns = ArimaModels().to_log_returns(prices)
    # Both returns touching the missing price are missing; nothing else is.
    assert returns.isna().sum() == 2
    assert np.isnan(returns.loc[prices.index[5]])
    assert np.isnan(returns.loc[prices.index[6]])


# p_d_q_fit with a fake ARIMA


class FakeFitted:
    def __init__(self, next_value: float):
        self._next = next_value
        self.aic = 10.0
        self.bic = 20.0

    def forecast(self, steps: int = 1) -> pd.Series:
        return pd.Series([self._next])


def install_fake_arima(monkeypatch, ts: pd.Series, fail: dict[int, type[Exception]]):
    """
    Replace statsmodels' ARIMA. A successful fit forecasts the true next
    value of ts exactly, so the RMSE over successful steps is 0. Step k (the
    k-th fit, from 0) raises fail[k] if present. If a failed step were scored
    with an earlier step's model, that forecast (an earlier value) would be
    wrong and the RMSE would be positive.
    """
    calls = {"n": 0}

    class FakeARIMA:
        def __init__(self, endog, order):
            self.endog = endog

        def fit(self):
            k = calls["n"]
            calls["n"] += 1
            if k in fail:
                raise fail[k]("forced failure")
            position = ts.index.get_loc(self.endog.index[-1]) + 1
            return FakeFitted(float(ts.iloc[position]))

    monkeypatch.setattr(arima_models.tsa, "ARIMA", FakeARIMA)
    return calls


def task_for(ts: pd.Series, train_len: int = 10) -> dict:
    return {"ts": ts, "d": 0, "train_len": train_len, "p": 1, "q": 0}


@pytest.fixture
def returns() -> pd.Series:
    return ArimaModels().to_log_returns(session_prices(n=21))  # 20 returns


def test_first_fit_failing_does_not_crash(monkeypatch, returns):
    # With the old code this raised, because no previous model existed.
    install_fake_arima(monkeypatch, returns, fail={0: ValueError})
    result = ArimaModels().p_d_q_fit(task_for(returns))
    assert result["n_stationarity_errors"] == 1
    assert result["rmse"] == pytest.approx(0.0)


def test_failed_step_is_excluded_not_scored_with_a_stale_model(monkeypatch, returns):
    install_fake_arima(monkeypatch, returns, fail={3: ValueError, 6: LinAlgError})
    result = ArimaModels().p_d_q_fit(task_for(returns))
    assert result["n_stationarity_errors"] == 1
    assert result["n_convergence_errors"] == 1
    # Successful fits forecast perfectly; reusing a stale model would not.
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mean_aic"] == pytest.approx(10.0)
    assert result["mean_bic"] == pytest.approx(20.0)


def test_one_fit_per_step(monkeypatch, returns):
    calls = install_fake_arima(monkeypatch, returns, fail={})
    ArimaModels().p_d_q_fit(task_for(returns, train_len=10))
    assert calls["n"] == len(returns) - 10


def test_all_fits_failing_scores_infinite_and_ranks_last(monkeypatch, returns):
    n_steps = len(returns) - 10
    install_fake_arima(monkeypatch, returns, fail={k: ValueError for k in range(n_steps)})
    failed = ArimaModels().p_d_q_fit(task_for(returns))
    assert failed["rmse"] == np.inf
    assert failed["mean_bic"] == np.inf
    assert failed["n_stationarity_errors"] == n_steps
    # Next to a working candidate, the failed one is never selected.
    train_result = pd.DataFrame(
        [
            {**failed, "p": 0, "q": 1},
            {"p": 1, "q": 0, "rmse": 0.02, "mean_aic": 5.0, "mean_bic": 7.0},
        ]
    ).set_index(["p", "q"])
    assert ArimaModels().select_order(train_result) == (1, 0)


# p_d_q_fit with the real statsmodels ARIMA (no Dask)


def test_p_d_q_fit_with_real_arima_returns_finite_scores(returns):
    # Small real fits can occasionally fail; only some must succeed.
    result = ArimaModels().p_d_q_fit(task_for(returns))
    n_steps = len(returns) - 10
    n_errors = result["n_convergence_errors"] + result["n_stationarity_errors"]
    assert n_errors < n_steps
    assert np.isfinite(result["rmse"])
    assert np.isfinite(result["mean_bic"])
