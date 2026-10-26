"""
Unit tests for fuzzy_system_research.optimization.portfolio: the optimizers
under the shared PortfolioProblem constraints (including a binding weight
cap), convergence checks, the maximum return portfolio, the efficient
frontier and its unreached targets, and the risk-free rate.

In-memory only. get_risk_free_rate() is tested inside pytest's tmp_path with
the FRED download replaced by a fake, so the real input/ folder and the
network are never touched.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from scipy.optimize import OptimizeResult

from fuzzy_system_research.optimization import portfolio
from fuzzy_system_research.optimization.portfolio import (
    OptimizationError,
    efficient_frontier,
    get_risk_free_rate,
    latest_risk_free_rate,
    max_return_weights,
    max_sharpe_weights,
    min_variance_weights,
)
from fuzzy_system_research.optimization.problem import PortfolioProblem

DAILY_RF = 0.01


def make_problem(mean_returns, cov, cap_multiple: float = 4) -> PortfolioProblem:
    n = len(mean_returns)
    return PortfolioProblem(
        tickers=tuple(f"T{i}" for i in range(n)),
        daily_mean_returns=np.asarray(mean_returns, dtype=float),
        daily_cov=np.asarray(cov, dtype=float),
        daily_risk_free_rate=DAILY_RF,
        weight_cap_multiple=cap_multiple,
    )


# Optimizers respect a binding weight cap (8 assets: cap = 4/8 = 50 %)

N_CAPPED = 8
CAP = 4.0 / N_CAPPED


def dominant_asset_problem() -> PortfolioProblem:
    """
    Asset 0 has by far the best return and the lowest variance, so without
    the cap both optimizers would put (nearly) everything in it.
    """
    mean_returns = np.array([0.30] + [0.02] * (N_CAPPED - 1))
    cov = np.diag([0.5] + [4.0] * (N_CAPPED - 1))
    return make_problem(mean_returns, cov)


@pytest.mark.parametrize("optimizer", [max_sharpe_weights, min_variance_weights])
def test_optimizers_respect_binding_cap(optimizer):
    weights = optimizer(dominant_asset_problem())
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()
    assert (weights <= CAP + 1e-6).all()
    assert weights[0] == pytest.approx(CAP, abs=1e-4)  # The cap binds.


def test_optimizers_use_the_problems_bounds_and_constraints(monkeypatch):
    # Every optimizer passes exactly the problem's bounds and its budget
    # constraint to SciPy, so they cannot use different constraint sets.
    problem = dominant_asset_problem()
    seen = []
    real = portfolio.minimize

    def spy(fun, x0, **kwargs):
        seen.append(kwargs)
        return real(fun, x0, **kwargs)

    monkeypatch.setattr(portfolio, "minimize", spy)
    min_variance_weights(problem)
    max_sharpe_weights(problem)
    max_return_weights(problem)
    efficient_frontier(problem, 0.05, 0.1, 3)
    assert len(seen) == 6
    for kwargs in seen:
        assert kwargs["method"] == "SLSQP"
        assert kwargs["bounds"] == problem.bounds()
        budget = kwargs["constraints"][0]
        assert budget["fun"] is problem.constraints()[0]["fun"]


def test_max_sharpe_is_optimal_among_random_feasible_portfolios():
    problem = dominant_asset_problem()
    stats = problem.portfolio_stats(max_sharpe_weights(problem))
    samples = problem.sample_feasible_weights(5000, np.random.default_rng(3))
    sampled = problem.portfolio_stats(samples)
    assert sampled.sharpe_ratio.max() <= stats.sharpe_ratio + 1e-4


def test_min_variance_is_least_risky_among_random_feasible_portfolios():
    problem = dominant_asset_problem()
    risk = problem.portfolio_stats(min_variance_weights(problem)).risk
    samples = problem.sample_feasible_weights(5000, np.random.default_rng(4))
    assert problem.portfolio_stats(samples).risk.min() >= risk * (1 - 1e-4)


def greedy_max_return(problem: PortfolioProblem) -> float:
    """
    Closed-form maximum return under the current constraints (long-only,
    sum to 1, each weight at most the cap): fill the highest-return assets
    up to the cap, in order.
    """
    remaining, total = 1.0, 0.0
    for mu in sorted(problem.daily_mean_returns, reverse=True):
        w = min(problem.weight_cap, remaining)
        total += w * mu
        remaining -= w
    return total


@pytest.mark.parametrize("cap_multiple", [1.5, 2, 4])
def test_max_return_portfolio_matches_closed_form(cap_multiple):
    mean_returns = np.array([0.05, 0.12, 0.08, 0.20, 0.01])
    problem = make_problem(mean_returns, np.eye(5), cap_multiple)
    weights = max_return_weights(problem)
    assert problem.max_constraint_violation(weights) <= 1e-6
    assert problem.daily_mean_return(weights) == pytest.approx(
        greedy_max_return(problem), abs=1e-6
    )
    # The library's closed form, used by the run-time check, agrees too.
    assert problem.daily_mean_return(
        problem.max_return_weights_closed_form()
    ) == pytest.approx(greedy_max_return(problem), abs=1e-12)


# Convergence: an unconverged optimization must not return weights


@pytest.fixture
def failing_minimize(monkeypatch):
    """Replace SciPy's minimize with one that never converges."""

    def fake_minimize(fun, x0, **kwargs):
        x = np.asarray(x0, dtype=float)
        return OptimizeResult(
            x=x, fun=fun(x), success=False, status=9, message="Iteration limit reached"
        )

    monkeypatch.setattr(portfolio, "minimize", fake_minimize)


@pytest.mark.parametrize(
    "optimizer, name",
    [
        (min_variance_weights, "minimum variance.*Iteration limit"),
        (max_sharpe_weights, "max Sharpe ratio.*status 9"),
        (max_return_weights, "maximum return"),
    ],
)
def test_optimizers_raise_if_they_do_not_converge(failing_minimize, optimizer, name):
    with pytest.raises(OptimizationError, match=name):
        optimizer(dominant_asset_problem())


def test_unreached_frontier_targets_stay_nan_instead_of_raising(failing_minimize):
    # The efficient frontier records an unreachable target as NaN (see the
    # contract); only the portfolio optimizations must converge.
    targets, weights = efficient_frontier(dominant_asset_problem(), 0.02, 0.3, 5)
    assert len(targets) == 5
    assert np.isnan(weights).all()


def test_optimization_error_is_a_runtime_error():
    assert issubclass(OptimizationError, RuntimeError)


# efficient_frontier


def test_efficient_frontier_marks_unreachable_targets_as_nan():
    # Three assets, cap 4/3 (not binding): the highest reachable return is
    # max(mean_returns) = 0.3, so targets up to 0.6 are partly infeasible.
    problem = make_problem([0.1, 0.2, 0.3], np.diag([1.0, 2.0, 3.0]))
    min_var = min_variance_weights(problem)
    targets, weights = efficient_frontier(
        problem, problem.daily_mean_return(min_var), 0.6, 50
    )
    risks = problem.portfolio_stats(weights).risk
    assert weights.shape == (50, 3)
    assert np.isnan(risks[targets > 0.3 + 1e-6]).all()
    # Stay clear of the corner w = (0, 0, 1), where SLSQP may legitimately stop.
    feasible = risks[targets < 0.3 - 0.02]
    assert np.isfinite(feasible).all()
    assert feasible[0] == pytest.approx(problem.portfolio_stats(min_var).risk, rel=1e-3)
    # Upper (efficient) branch: volatility does not decrease as the target rises.
    assert (np.diff(feasible) >= -1e-6).all()


def test_efficient_frontier_points_hit_their_targets_within_the_cap():
    problem = make_problem(
        [0.05, 0.12, 0.08, 0.20, 0.01], np.diag([1.0, 2.0, 1.5, 3.0, 0.5]), 2
    )
    low = problem.daily_mean_return(min_variance_weights(problem))
    high = problem.daily_mean_return(max_return_weights(problem))
    targets, weights = efficient_frontier(problem, low, high, 20)
    assert targets[0] == low and targets[-1] == high
    reached = ~np.isnan(weights).any(axis=1)
    assert reached[:-2].all()  # Only points at the corner at the very top may be missed.
    assert problem.max_constraint_violation(weights[reached]).max() <= 1e-6
    np.testing.assert_allclose(
        problem.daily_mean_return(weights[reached]), targets[reached], atol=1e-6
    )


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


def test_get_risk_free_rate_creates_missing_input_folder(tmp_path, monkeypatch, fake_fred):
    # A fresh checkout has no input/ folder at all.
    monkeypatch.chdir(tmp_path)
    data = get_risk_free_rate(FakeDateManager())
    cache_path = os.path.join("input", "Risk Free Rate 2026-10-14.json")
    with open(cache_path, encoding="utf-8") as f:
        assert json.load(f) == data


def test_get_risk_free_rate_with_no_valid_rate_raises_and_writes_nothing(
    in_tmp_input, fake_fred
):
    fake_fred["df"] = fred_frame(["2026-10-12"], [np.nan])
    with pytest.raises(ValueError):
        get_risk_free_rate(FakeDateManager())
    assert not os.path.exists(in_tmp_input)
