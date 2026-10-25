"""
Tests for fuzzy_system_research.optimization.workflow on seeded synthetic
prices. In-memory, except the writers, which write under pytest's tmp_path.
"""

import h5py
import numpy as np
import pytest
import yaml

from conftest import make_long_prices
from fuzzy_system_research.data.prices import pivot_close_wide
from fuzzy_system_research.optimization import workflow
from fuzzy_system_research.optimization.portfolio import OptimizationError
from fuzzy_system_research.optimization.workflow import (
    OptimizationResult,
    build_optimization_metadata,
    run_portfolio_optimization,
    write_optimization_metadata,
    write_plot_data_h5,
)

ANNUAL_RF = 4.05
DAILY_RF = ANNUAL_RF / 252
RF_DATA = {
    "risk_free_rate": ANNUAL_RF,
    "daily_risk_free_rate": DAILY_RF,
    "risk_free_rate_date": "2026-10-08 00:00:00",
}


@pytest.fixture(scope="module")
def wide():
    return pivot_close_wide(make_long_prices())


@pytest.fixture(scope="module")
def result(wide) -> OptimizationResult:
    return run_portfolio_optimization(wide, DAILY_RF, rng=np.random.default_rng(0))


def test_result_shapes(result):
    assert result.frontier_risks.shape == (100,)
    assert result.frontier_returns.shape == (100,)
    assert result.tangency_risks.shape == (100,)
    assert result.tangency_returns.shape == (100,)
    assert result.simulated_risks.shape == (10_000,)
    assert result.simulated_returns.shape == (10_000,)


def test_result_dates_span_the_data(result, wide):
    assert result.date_from == wide.index[0]
    assert result.date_to == wide.index[-1]


def test_weights_are_long_only_and_sum_to_100(result, wide):
    weights = result.max_sharpe_weights_pct
    assert set(weights) == set(wide.columns)
    assert all(w >= -1e-6 for w in weights.values())
    # SLSQP satisfies the sum-to-1 constraint only to its tolerance.
    assert sum(weights.values()) == pytest.approx(100, abs=1e-3)


def test_min_var_weights_are_long_only_and_sum_to_100(result, wide):
    weights = result.min_var_weights_pct
    assert set(weights) == set(wide.columns)
    assert all(w >= -1e-6 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(100, abs=1e-3)
    assert all(isinstance(w, float) for w in weights.values())


def test_min_var_weights_reproduce_min_var_risk_and_return(result, wide):
    # The stored weights must be the ones that produced the uploaded risk and
    # return: recompute both from the same daily percent returns.
    returns = wide.pct_change().iloc[1:] * 100
    w = np.array([result.min_var_weights_pct[t] for t in wide.columns]) / 100
    daily_risk = np.sqrt(w @ returns.cov().to_numpy() @ w)
    daily_return = w @ returns.mean().to_numpy()
    assert result.min_var_risk == pytest.approx(daily_risk * np.sqrt(252), rel=1e-9)
    assert result.min_var_return == pytest.approx(daily_return * 252, rel=1e-9)


def test_min_var_portfolio_is_no_riskier_than_any_simulated_portfolio(result):
    # With 3 assets the weight cap never binds, so every random long-only
    # portfolio is feasible for the optimizer. Allow for SLSQP's tolerance.
    assert result.min_var_risk <= result.simulated_risks.min() * (1 + 1e-3)


def test_sharpe_ratio_identity_in_annual_units(result):
    expected = (result.max_sharpe_return - ANNUAL_RF) / result.max_sharpe_risk
    assert result.sharpe_ratio == pytest.approx(expected, rel=1e-9)


def test_tangency_line_starts_at_annual_rf_with_sharpe_slope(result):
    assert result.tangency_risks[0] == pytest.approx(0.0)
    assert result.tangency_returns[0] == pytest.approx(ANNUAL_RF)
    np.testing.assert_allclose(
        result.tangency_returns,
        ANNUAL_RF + result.sharpe_ratio * result.tangency_risks,
    )
    assert not np.isnan(result.tangency_returns).any()


def test_tangency_line_ends_at_largest_finite_frontier_risk(result):
    assert result.tangency_risks[-1] == pytest.approx(np.nanmax(result.frontier_risks))


def test_min_var_is_least_risky_frontier_point(result):
    finite = result.frontier_risks[~np.isnan(result.frontier_risks)]
    # Two separate SLSQP solves; allow for their convergence tolerance.
    assert result.min_var_risk <= finite.min() * (1 + 1e-3)


def test_seeded_simulation_is_reproducible(wide, result):
    again = run_portfolio_optimization(wide, DAILY_RF, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(again.simulated_risks, result.simulated_risks)
    np.testing.assert_array_equal(again.simulated_returns, result.simulated_returns)


def test_write_plot_data_h5_matches_contract(result, tmp_path):
    path = tmp_path / "portfolio_optimization_plot_data.h5"
    write_plot_data_h5(result, str(path))
    with h5py.File(path, "r") as hf:
        assert hf.attrs["annualization"] == "arithmetic"
        assert hf.attrs["periods_per_year"] == 252
        assert hf.attrs["return_units"] == "percent_per_year"
        assert hf.attrs["risk_units"] == "percent_per_year"
        assert hf.attrs["return_type"] == "simple"
        assert set(hf.keys()) == {
            "efficient_frontier",
            "tangency_line",
            "simulated_portfolios",
            "max_sharpe_ratio",
            "min_var_portfolio",
        }
        for group in hf.values():
            assert set(group.keys()) == {"xs", "ys"}
            assert group["xs"].dtype == np.float64
            assert group["xs"].shape == group["ys"].shape
        assert hf["max_sharpe_ratio/xs"][0] == pytest.approx(result.max_sharpe_risk)
        assert hf["max_sharpe_ratio/ys"][0] == pytest.approx(result.max_sharpe_return)
        assert hf["min_var_portfolio/xs"][0] == pytest.approx(result.min_var_risk)
        assert hf["simulated_portfolios/xs"].shape == (10_000,)


def test_build_optimization_metadata(result):
    tickers = ["XBI", "QQQ", "GLD"]
    metadata = build_optimization_metadata(result, tickers, RF_DATA)
    assert metadata["date_updated"] == {
        "date_from": str(result.date_from.date()),
        "date_to": str(result.date_to.date()),
    }
    assert metadata["tickers"] == tickers
    assert metadata["risk_free_rate"] == RF_DATA
    assert metadata["annualization"] == {"method": "arithmetic", "periods_per_year": 252}
    assert metadata["optimum_portfolio"]["sharpe_ratio"] == result.sharpe_ratio
    assert metadata["optimum_portfolio"]["weights"] == result.max_sharpe_weights_pct
    assert metadata["min_var_portfolio"] == {
        "annualized_return": result.min_var_return,
        "risk": result.min_var_risk,
        "weights": result.min_var_weights_pct,
    }


def test_metadata_yaml_round_trip_keeps_dates_as_strings(result, tmp_path):
    path = tmp_path / "optimization_metadata.yml"
    write_optimization_metadata(
        build_optimization_metadata(result, ["XBI", "QQQ", "GLD"], RF_DATA), str(path)
    )
    loaded = yaml.safe_load(path.read_text())
    assert isinstance(loaded["date_updated"]["date_to"], str)
    assert isinstance(loaded["risk_free_rate"]["risk_free_rate_date"], str)
    assert loaded["optimum_portfolio"]["risk"] == pytest.approx(result.max_sharpe_risk)
    # Both weight maps load as plain floats keyed by ticker, not numpy tags.
    for section in ("optimum_portfolio", "min_var_portfolio"):
        weights = loaded[section]["weights"]
        assert set(weights) == {"XBI", "QQQ", "GLD"}
        assert all(isinstance(w, float) for w in weights.values())
    assert loaded["min_var_portfolio"]["weights"] == pytest.approx(
        result.min_var_weights_pct
    )


# Optimality. With 3 assets the 4/N weight cap is 133 %, so it never binds and
# the unconstrained random portfolios are all feasible for the optimizer.


def test_max_sharpe_beats_every_simulated_and_frontier_portfolio(result):
    # SLSQP converges to ~1e-6 on the daily objective (~1e-5 annualized), and
    # the synthetic Sharpe ratio may be negative, so use an absolute margin.
    tol = 1e-4 * (1 + abs(result.sharpe_ratio))
    sim_sharpe = (result.simulated_returns - ANNUAL_RF) / result.simulated_risks
    assert sim_sharpe.max() <= result.sharpe_ratio + tol
    finite = ~np.isnan(result.frontier_risks)
    frontier_sharpe = (
        result.frontier_returns[finite] - ANNUAL_RF
    ) / result.frontier_risks[finite]
    assert frontier_sharpe.max() <= result.sharpe_ratio + tol


def test_no_simulated_portfolio_dominates_the_frontier(result):
    # Each frontier point is the least risky portfolio with that target
    # return, and the frontier is the upper (rising) branch, so no portfolio
    # with at least that return may have less risk. Tolerance allows for SLSQP
    # convergence.
    finite = ~np.isnan(result.frontier_risks)
    targets = result.frontier_returns[finite]
    risks = result.frontier_risks[finite]
    sim_r = result.simulated_returns[:, None]
    sim_s = result.simulated_risks[:, None]
    dominates = (sim_r >= targets[None, :]) & (sim_s < risks[None, :] * (1 - 1e-3))
    assert not dominates.any()


def test_weight_cap_binds_with_many_assets():
    # 8 assets: cap = 4/8 = 50 % per asset.
    tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
    wide8 = pivot_close_wide(make_long_prices(tickers=tickers, n_sessions=80, seed=11))
    res = run_portfolio_optimization(wide8, DAILY_RF, rng=np.random.default_rng(0))
    for weights_pct in (res.max_sharpe_weights_pct, res.min_var_weights_pct):
        weights = np.array(list(weights_pct.values()))
        assert len(weights) == 8
        assert (weights <= 50 + 1e-4).all()
        assert (weights >= -1e-6).all()
        assert weights.sum() == pytest.approx(100, abs=1e-3)


@pytest.mark.parametrize(
    "optimizer, name",
    [
        ("calc_min_variance_portfolio", "minimum variance"),
        ("optimize_sharpe_ratio", "max Sharpe ratio"),
    ],
)
def test_unconverged_optimization_stops_the_run(wide, monkeypatch, optimizer, name):
    def not_converged(*args, **kwargs):
        raise OptimizationError(f"The {name} optimization did not converge.")

    monkeypatch.setattr(workflow, optimizer, not_converged)
    with pytest.raises(OptimizationError, match=name):
        run_portfolio_optimization(wide, DAILY_RF, rng=np.random.default_rng(0))


def test_tangency_line_survives_nan_first_frontier_point(wide, monkeypatch):
    # Regression test for the v5 fix: max() on a list starting with NaN
    # returned NaN and made the whole tangency line NaN; nanmax must not.
    real = workflow.calc_efficient_frontier

    def frontier_with_leading_nan(*args, **kwargs):
        risks, targets = real(*args, **kwargs)
        return [np.nan] + list(risks[1:]), targets

    monkeypatch.setattr(workflow, "calc_efficient_frontier", frontier_with_leading_nan)
    res = run_portfolio_optimization(wide, DAILY_RF, rng=np.random.default_rng(0))
    assert np.isnan(res.frontier_risks[0])
    assert np.isfinite(res.tangency_risks).all()
    assert np.isfinite(res.tangency_returns).all()
    assert res.tangency_risks[-1] == pytest.approx(np.nanmax(res.frontier_risks))
