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
