"""
Unit tests for fuzzy_system_research.optimization.portfolio: random weights,
weight bounds, the optimizers' constraints (including a binding weight cap),
infeasible efficient-frontier targets, and the risk-free rate.

In-memory only. get_risk_free_rate() is tested inside pytest's tmp_path with
the FRED download replaced by a fake, so the real input/ folder and the
network are never touched.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from fuzzy_system_research.optimization import portfolio
from fuzzy_system_research.optimization.portfolio import (
    calc_efficient_frontier,
    calc_min_variance_portfolio,
    calc_weight_bounds,
    get_risk_free_rate,
    latest_risk_free_rate,
    optimize_sharpe_ratio,
    softmax_random_distribution,
)
from fuzzy_system_research.optimization.workflow import PortfolioAssets


def assets(n: int) -> PortfolioAssets:
    return PortfolioAssets(tickers=tuple(f"T{i}" for i in range(n)))


# softmax_random_distribution


def test_softmax_weights_are_positive_and_sum_to_one():
    w = softmax_random_distribution(8, rng=np.random.default_rng(1))
    assert w.shape == (8,)
    assert (w > 0).all()
    assert w.sum() == pytest.approx(1.0)


def test_softmax_is_reproducible_with_seeded_rng():
    a = softmax_random_distribution(5, rng=np.random.default_rng(42))
    b = softmax_random_distribution(5, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("bad_d", [0, -1, 2.5, "3"])
def test_softmax_rejects_non_positive_integers(bad_d):
    with pytest.raises(ValueError):
        softmax_random_distribution(bad_d)


# calc_weight_bounds


@pytest.mark.parametrize("n", [3, 8, 10])
def test_weight_bounds_are_long_only_and_capped_at_4_over_n(n):
    bounds = calc_weight_bounds(assets(n))
    assert bounds == [(0.0, 4.0 / n)] * n


# Optimizers respect a binding weight cap (8 assets: cap = 4/8 = 50 %)

N_CAPPED = 8
CAP = 4.0 / N_CAPPED


def dominant_asset_inputs():
    """
    Asset 0 has by far the best return and the lowest variance, so without
    the cap both optimizers would put (nearly) everything in it.
    """
    mean_returns = np.array([0.30] + [0.02] * (N_CAPPED - 1))
    cov = np.diag([0.5] + [4.0] * (N_CAPPED - 1))
    return mean_returns, cov


def test_max_sharpe_respects_binding_cap():
    mean_returns, cov = dominant_asset_inputs()
    _, weights, _, _ = optimize_sharpe_ratio(assets(N_CAPPED), 0.01, mean_returns, cov)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()
    assert (weights <= CAP + 1e-6).all()
    assert weights[0] == pytest.approx(CAP, abs=1e-4)  # The cap binds.


def test_min_variance_respects_binding_cap():
    mean_returns, cov = dominant_asset_inputs()
    _, weights, _ = calc_min_variance_portfolio(assets(N_CAPPED), cov, mean_returns)
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()
    assert (weights <= CAP + 1e-6).all()
    assert weights[0] == pytest.approx(CAP, abs=1e-4)  # The cap binds.


def test_max_sharpe_is_optimal_among_random_feasible_portfolios():
    mean_returns, cov = dominant_asset_inputs()
    rf = 0.01
    best_sharpe, _, _, _ = optimize_sharpe_ratio(assets(N_CAPPED), rf, mean_returns, cov)
    rng = np.random.default_rng(3)
    for _ in range(2000):
        w = rng.dirichlet(np.ones(N_CAPPED))
        if w.max() > CAP:
            continue  # Only compare against portfolios the optimizer may choose.
        sharpe = (w @ mean_returns - rf) / np.sqrt(w @ cov @ w)
        assert sharpe <= best_sharpe + 1e-5  # SLSQP convergence tolerance.


# calc_efficient_frontier: infeasible targets become NaN


def test_efficient_frontier_marks_unreachable_targets_as_nan():
    # Three assets, cap 4/3 (not binding): the highest reachable return is
    # max(mean_returns) = 0.3, so targets up to 0.6 are partly infeasible.
    mean_returns = np.array([0.1, 0.2, 0.3])
    cov = np.diag([1.0, 2.0, 3.0])
    min_var_risk, _, min_var_return = calc_min_variance_portfolio(
        assets(3), cov, mean_returns
    )
    risks, targets = calc_efficient_frontier(
        assets(3), min_var_return, 0.6, mean_returns, cov, num_portfolios=50
    )
    risks = np.asarray(risks)
    assert len(risks) == len(targets) == 50
    assert np.isnan(risks[targets > 0.3 + 1e-6]).all()
    # Stay clear of the corner w = (0, 0, 1), where SLSQP may legitimately stop.
    feasible = risks[targets < 0.3 - 0.02]
    assert np.isfinite(feasible).all()
    assert feasible[0] == pytest.approx(min_var_risk, rel=1e-3)
    # Upper (efficient) branch: risk does not decrease as the target rises.
    assert (np.diff(feasible) >= -1e-6).all()


# Risk-free rate


def fred_frame(dates: list[str], rates: list[float]) -> pd.DataFrame:
    """Shaped like pandas_datareader's FRED result: DATE index, DTB3 column."""
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="DATE")
    return pd.DataFrame({"DTB3": rates}, index=index)


def test_latest_risk_free_rate_uses_most_recent_value_and_daily_units():
    df = fred_frame(["2026-10-07", "2026-10-08"], [4.10, 4.05])
    data = latest_risk_free_rate(df)
    assert data["risk_free_rate"] == pytest.approx(4.05)
    assert data["daily_risk_free_rate"] == pytest.approx(4.05 / 252)
    assert data["risk_free_rate_date"] == "2026-10-08 00:00:00"


def test_latest_risk_free_rate_skips_holiday_nan():
    # 2026-10-12 (Columbus Day) has no DTB3 value; use the previous one.
    df = fred_frame(["2026-10-08", "2026-10-09", "2026-10-12"], [4.05, 4.07, np.nan])
    data = latest_risk_free_rate(df)
    assert data["risk_free_rate"] == pytest.approx(4.07)
    assert data["risk_free_rate_date"] == "2026-10-09 00:00:00"
    assert np.isfinite(data["daily_risk_free_rate"])


def test_latest_risk_free_rate_ignores_row_order():
    df = fred_frame(["2026-10-09", "2026-10-07", "2026-10-08"], [4.07, 4.10, 4.05])
    assert latest_risk_free_rate(df)["risk_free_rate"] == pytest.approx(4.07)


def test_latest_risk_free_rate_with_no_valid_value_raises():
    df = fred_frame(["2026-10-12"], [np.nan])
    with pytest.raises(ValueError):
        latest_risk_free_rate(df)


class FakeDateManager:
    def get_today_date(self) -> str:
        return "2026-10-14"


@pytest.fixture
def in_tmp_input(tmp_path, monkeypatch):
    """Run in tmp_path with an empty input/ folder; return the cache path."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("input")
    return os.path.join("input", "Risk Free Rate 2026-10-14.json")


@pytest.fixture
def fake_fred(monkeypatch):
    """Replace the FRED download; set calls['df'] to the frame to return."""
    calls = {"count": 0, "df": fred_frame(["2026-10-09", "2026-10-12"], [4.07, np.nan])}

    def fake_data_reader(name, source, start, end):
        assert (name, source) == ("DTB3", "fred")
        calls["count"] += 1
        return calls["df"]

    monkeypatch.setattr(portfolio.web, "DataReader", fake_data_reader)
    return calls


def test_get_risk_free_rate_downloads_skips_nan_and_writes_cache(in_tmp_input, fake_fred):
    data = get_risk_free_rate(FakeDateManager())
    assert fake_fred["count"] == 1
    assert data["risk_free_rate"] == pytest.approx(4.07)
    with open(in_tmp_input, encoding="utf-8") as f:
        assert json.load(f) == data


def test_get_risk_free_rate_uses_valid_cache_without_download(in_tmp_input, fake_fred):
    cached = {
        "risk_free_rate": 4.0,
        "daily_risk_free_rate": 4.0 / 252,
        "risk_free_rate_date": "2026-10-09 00:00:00",
    }
    with open(in_tmp_input, "w", encoding="utf-8") as f:
        json.dump(cached, f)
    assert get_risk_free_rate(FakeDateManager()) == cached
    assert fake_fred["count"] == 0


def test_get_risk_free_rate_replaces_cached_nan(in_tmp_input, fake_fred):
    # A cache written by the old code on a holiday contains NaN.
    with open(in_tmp_input, "w", encoding="utf-8") as f:
        f.write(
            '{"risk_free_rate": NaN, "daily_risk_free_rate": NaN, '
            '"risk_free_rate_date": "2026-10-12 00:00:00"}'
        )
    data = get_risk_free_rate(FakeDateManager())
    assert fake_fred["count"] == 1
    assert data["risk_free_rate"] == pytest.approx(4.07)
    with open(in_tmp_input, encoding="utf-8") as f:
        assert json.load(f)["risk_free_rate"] == pytest.approx(4.07)


def test_get_risk_free_rate_with_no_valid_rate_raises_and_writes_nothing(
    in_tmp_input, fake_fred
):
    fake_fred["df"] = fred_frame(["2026-10-12"], [np.nan])
    with pytest.raises(ValueError):
        get_risk_free_rate(FakeDateManager())
    assert not os.path.exists(in_tmp_input)
