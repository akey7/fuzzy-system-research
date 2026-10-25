"""
End-to-end portfolio optimization for one run: from wide adjusted close
prices to annualized results, plus writers for the two files the frontend
reads (see docs/INTERFACE_CONTRACT.md).

Conventions (unchanged from the original script):
- Returns are simple daily returns in percent; the optimizer works in daily
  units with the daily risk-free rate (annual T-bill rate / 252).
- Constraints come from portfolio.calc_weight_bounds(): long-only, weights
  sum to 1, each weight capped at 4 / (number of assets).
- Everything stored in OptimizationResult is annualized arithmetically
  (return × 252, risk × √252, Sharpe × √252), which leaves the optimal
  weights unchanged and keeps the tangency line straight.
- Mean returns and covariance are estimated from about one year of daily
  data, so the results are noisy.
- The minimum variance and max Sharpe ratio optimizations must converge;
  otherwise portfolio.OptimizationError stops the run before the
  optimization files are written and before anything is uploaded. An
  efficient-frontier target return that cannot be reached is recorded as
  NaN instead (see docs/INTERFACE_CONTRACT.md).
"""

from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
import yaml
from fuzzy_system_research.optimization.portfolio import (
    TRADING_DAYS_PER_YEAR,
    annualize_return,
    annualize_risk,
    annualize_sharpe_ratio,
    calc_efficient_frontier,
    calc_min_variance_portfolio,
    optimize_sharpe_ratio,
    simulate_portfolios,
)


# Written directly above the top-level optimum_portfolio key in
# optimization_metadata.yml. The key keeps its name for compatibility with
# readers (see docs/INTERFACE_CONTRACT.md); the comment says what it holds.
OPTIMUM_PORTFOLIO_COMMENT = (
    "# optimum_portfolio is the maximum Sharpe ratio (tangency) portfolio."
)


@dataclass(frozen=True)
class PortfolioAssets:
    """
    The assets being optimized. The functions in portfolio.py only need an
    object with a tickers sequence; this one is built from the price data's
    columns, so it always matches the returns and covariance dimensions.
    """

    tickers: tuple[str, ...]


@dataclass(frozen=True)
class OptimizationResult:
    """
    Results of one portfolio optimization run. All returns and risks are
    annualized arithmetically and in percent; the Sharpe ratio is
    annualized and dimensionless; weights are in percent, keyed by ticker,
    for both the max Sharpe ratio and the minimum variance portfolio.
    """

    date_from: pd.Timestamp
    date_to: pd.Timestamp
    frontier_risks: np.ndarray
    frontier_returns: np.ndarray
    tangency_risks: np.ndarray
    tangency_returns: np.ndarray
    simulated_risks: np.ndarray
    simulated_returns: np.ndarray
    max_sharpe_risk: float
    max_sharpe_return: float
    sharpe_ratio: float
    max_sharpe_weights_pct: dict[str, float]
    min_var_risk: float
    min_var_return: float
    min_var_weights_pct: dict[str, float]


def run_portfolio_optimization(
    wide_df: pd.DataFrame,
    daily_risk_free_rate: float,
    rng: np.random.Generator | None = None,
) -> OptimizationResult:
    """
    Compute the Monte Carlo cloud, minimum variance portfolio, efficient
    frontier, max Sharpe ratio portfolio, and tangency line from adjusted
    close prices, and annualize them for the frontend. Pure computation
    apart from progress printing; does no file or network I/O.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide adjusted close prices (tickers in columns, sorted DatetimeIndex),
        as returned by data.prices.pivot_close_wide().

    daily_risk_free_rate : float
        Daily risk-free rate in percent (annual rate in percent / 252).

    rng : np.random.Generator, optional
        Random number generator for the Monte Carlo simulation. If None,
        NumPy's global random state is used.

    Returns
    -------
    OptimizationResult
        Annualized results of the run.

    Raises
    ------
    portfolio.OptimizationError
        If the minimum variance or the max Sharpe ratio optimization does
        not converge.
    """
    assets = PortfolioAssets(tickers=tuple(wide_df.columns))
    print(
        f"Sanity check: Wide dataframe should have 0 missing values: {wide_df.isna().sum().sum()}"
    )
    returns_df = wide_df.pct_change()
    returns_df = returns_df.iloc[1:] * 100
    mean_returns = returns_df.mean()
    print("Mean portfolio returns:")
    print(mean_returns)
    cov = returns_df.cov()
    print("Covariance matrix:")
    print(cov)
    cov_np = cov.to_numpy()
    simulated_risks, simulated_returns = simulate_portfolios(
        assets, mean_returns, cov_np, rng=rng
    )
    min_var_risk, min_var_weights, min_var_return = calc_min_variance_portfolio(
        assets, cov_np, mean_returns
    )
    min_var_weights_pct_dict = pd.Series(
        min_var_weights * 100, index=mean_returns.index
    ).to_dict()
    print("Minimum variance weights %:")
    print(min_var_weights_pct_dict)
    optimized_risks, target_returns = calc_efficient_frontier(
        assets, min_var_return, max(simulated_returns), mean_returns, cov_np
    )
    best_sharpe_ratio, best_weights, opt_risk, opt_return = optimize_sharpe_ratio(
        assets, daily_risk_free_rate, mean_returns, cov_np
    )
    best_weights_pct_dict = pd.Series(
        best_weights * 100, index=mean_returns.index
    ).to_dict()
    print("Best weights %:")
    print(best_weights_pct_dict)
    # nanmax: infeasible frontier points are NaN and must not propagate.
    tangency_max_risk = np.nanmax(optimized_risks)
    tangency_xs = np.linspace(0, tangency_max_risk, 100)
    tangency_ys = daily_risk_free_rate + best_sharpe_ratio * tangency_xs
    result = OptimizationResult(
        date_from=wide_df.index[0],
        date_to=wide_df.index[-1],
        frontier_risks=annualize_risk(optimized_risks),
        frontier_returns=annualize_return(target_returns),
        tangency_risks=annualize_risk(tangency_xs),
        tangency_returns=annualize_return(tangency_ys),
        simulated_risks=annualize_risk(simulated_risks),
        simulated_returns=annualize_return(simulated_returns),
        max_sharpe_risk=float(annualize_risk(opt_risk)),
        max_sharpe_return=float(annualize_return(opt_return)),
        sharpe_ratio=annualize_sharpe_ratio(best_sharpe_ratio),
        max_sharpe_weights_pct=best_weights_pct_dict,
        min_var_risk=float(annualize_risk(min_var_risk)),
        min_var_return=float(annualize_return(min_var_return)),
        min_var_weights_pct=min_var_weights_pct_dict,
    )
    print("annualized_optimum_return, annualized_optimum_risk, annualized_sharpe_ratio")
    print(result.max_sharpe_return, result.max_sharpe_risk, result.sharpe_ratio)
    return result


def write_plot_data_h5(result: OptimizationResult, path: str) -> None:
    """
    Write portfolio_optimization_plot_data.h5 as specified in
    docs/INTERFACE_CONTRACT.md: five groups of xs (risk) / ys (return) datasets
    plus root attributes describing the units. Overwrites the file.

    Parameters
    ----------
    result : OptimizationResult
        Annualized results of the run.

    path : str
        Local path of the .h5 file to write. Its folder must exist.
    """
    with h5py.File(path, "w") as hf:
        hf.attrs["return_units"] = "percent_per_year"
        hf.attrs["risk_units"] = "percent_per_year"
        hf.attrs["return_type"] = "simple"
        hf.attrs["annualization"] = "arithmetic"
        hf.attrs["periods_per_year"] = TRADING_DAYS_PER_YEAR
        groups = {
            "efficient_frontier": (result.frontier_risks, result.frontier_returns),
            "tangency_line": (result.tangency_risks, result.tangency_returns),
            "simulated_portfolios": (result.simulated_risks, result.simulated_returns),
            "max_sharpe_ratio": ([result.max_sharpe_risk], [result.max_sharpe_return]),
            "min_var_portfolio": ([result.min_var_risk], [result.min_var_return]),
        }
        for name, (xs, ys) in groups.items():
            group = hf.create_group(name)
            group.create_dataset("xs", data=xs)
            group.create_dataset("ys", data=ys)


def build_optimization_metadata(
    result: OptimizationResult,
    tickers: list[str],
    risk_free_rate_data: dict,
) -> dict:
    """
    Build the contents of optimization_metadata.yml as specified in
    docs/INTERFACE_CONTRACT.md, including the weights (in percent) of both
    the max Sharpe ratio portfolio and the minimum variance portfolio. Pure
    function.

    Parameters
    ----------
    result : OptimizationResult
        Annualized results of the run.

    tickers : list[str]
        Tickers of the portfolio, in the backend's own order.

    risk_free_rate_data : dict
        As returned by portfolio.get_risk_free_rate(): risk_free_rate,
        daily_risk_free_rate, and risk_free_rate_date.

    Returns
    -------
    dict
        Nested dictionary ready for yaml.dump().
    """
    return {
        "date_updated": {
            "date_from": str(result.date_from.date()),
            "date_to": str(result.date_to.date()),
        },
        "tickers": list(tickers),
        "risk_free_rate": risk_free_rate_data,
        "annualization": {
            "method": "arithmetic",
            "periods_per_year": TRADING_DAYS_PER_YEAR,
        },
        "optimum_portfolio": {
            "annualized_return": result.max_sharpe_return,
            "risk": result.max_sharpe_risk,
            "sharpe_ratio": result.sharpe_ratio,
            "weights": result.max_sharpe_weights_pct,
        },
        "min_var_portfolio": {
            "annualized_return": result.min_var_return,
            "risk": result.min_var_risk,
            "weights": result.min_var_weights_pct,
        },
    }


def optimization_metadata_yaml(metadata: dict) -> str:
    """
    Render optimization_metadata.yml as text: block style, keys sorted, with
    OPTIMUM_PORTFOLIO_COMMENT on the line directly above the top-level
    optimum_portfolio key, because that key's name does not say which
    portfolio it is. YAML parsers ignore comments, so the parsed contents
    are exactly metadata. Pure function.

    Parameters
    ----------
    metadata : dict
        As returned by build_optimization_metadata().

    Returns
    -------
    str
        The YAML text.
    """
    lines = yaml.dump(metadata, default_flow_style=False).splitlines(keepends=True)
    return "".join(
        f"{OPTIMUM_PORTFOLIO_COMMENT}\n{line}"
        if line.startswith("optimum_portfolio:")
        else line
        for line in lines
    )


def write_optimization_metadata(metadata: dict, path: str) -> None:
    """
    Write optimization_metadata.yml as rendered by
    optimization_metadata_yaml() (block style, keys sorted, with a comment
    identifying optimum_portfolio as the max Sharpe ratio portfolio).
    Overwrites the file.

    Parameters
    ----------
    metadata : dict
        As returned by build_optimization_metadata().

    path : str
        Local path of the .yml file to write. Its folder must exist.
    """
    with open(path, "w") as f:
        f.write(optimization_metadata_yaml(metadata))
