"""
End-to-end portfolio optimization for one run: from wide adjusted close
prices to annualized results, plus writers for the two files the frontend
reads (see docs/INTERFACE_CONTRACT.md).

One definition of the problem: run_portfolio_optimization() builds a single
problem.PortfolioProblem (estimates, risk-free rate, constraints) and uses it
for everything:
- the max Sharpe ratio, minimum variance, and maximum return portfolios and
  the efficient frontier are solved under its bounds() and constraints();
- the random portfolios are drawn from the same constraint set by its
  sample_feasible_weights(), seeded from the configuration file;
- every coordinate in OptimizationResult (and so in the .h5 and .yml files)
  is computed by its portfolio_stats(), except the efficient frontier's
  target returns, which are the target grid and are checked against
  portfolio_stats(). The capital allocation line is built from the max
  Sharpe ratio point.

Conventions:
- Returns are simple daily returns in percent; the optimizers work in daily
  units with the daily risk-free rate (annual T-bill rate / 252).
- Constraints: long-only, weights sum to 1, each weight capped at
  weight_cap_multiple / (number of assets), set in
  config/modeling_and_optimization.yml.
- Everything stored in OptimizationResult is annualized arithmetically
  (return × 252, risk × √252), which leaves the optimal weights unchanged and
  keeps the capital allocation line straight.
- Mean returns and covariance are estimated from about one year of daily
  data, so the results are noisy.
- The minimum variance, max Sharpe ratio, and maximum return optimizations
  must converge; otherwise portfolio.OptimizationError stops the run before
  the optimization files are written and before anything is uploaded. An
  efficient-frontier target return that cannot be reached is recorded as NaN
  instead (see docs/INTERFACE_CONTRACT.md).
- check_optimization_invariants() and verify_optimization_files() raise
  InvariantError (an OptimizationError) if the plot data and the
  optimization disagree, which also stops the run before any upload.
"""

from dataclasses import dataclass
import h5py
import numpy as np
import pandas as pd
import yaml
from fuzzy_system_research.config import PortfolioOptimizationConfig
from fuzzy_system_research.optimization.portfolio import (
    OptimizationError,
    efficient_frontier,
    max_return_weights,
    max_sharpe_weights,
    min_variance_weights,
)
from fuzzy_system_research.optimization.problem import (
    TRADING_DAYS_PER_YEAR,
    PortfolioProblem,
    annualize_return,
)


# Written directly above the top-level optimum_portfolio key in
# optimization_metadata.yml. The key keeps its name for compatibility with
# readers (see docs/INTERFACE_CONTRACT.md); the comment says what it holds.
OPTIMUM_PORTFOLIO_COMMENT = (
    "# optimum_portfolio is the maximum Sharpe ratio (tangency) portfolio."
)

# Points on the capital allocation line (the tangency_line group of the .h5
# file). Two would draw the same straight line; 100 keeps the array shape
# documented in docs/INTERFACE_CONTRACT.md.
CAPITAL_ALLOCATION_LINE_POINTS = 100

# Tolerances of the invariant checks. They allow for SLSQP's convergence
# tolerance (about 1e-6 on the daily objective and on each equality
# constraint) and are far smaller than anything visible in the plot.
# Largest allowed constraint violation of any weight vector (fraction of the
# portfolio, so 1e-5 is 0.001 percentage points of weight).
WEIGHT_TOLERANCE = 1e-5
# No feasible portfolio may beat the max Sharpe ratio by more than this,
# times (1 + |max Sharpe ratio|).
SHARPE_TOLERANCE = 1e-4
# Relative tolerance for comparing risks: no random portfolio may have less
# risk than the efficient frontier allows by more than this fraction, and the
# minimum variance risk must equal the frontier's smallest risk to within it.
RISK_RELATIVE_TOLERANCE = 1e-3
# A frontier point's annualized return must equal its target to within this
# many percentage points per year.
RETURN_TOLERANCE = 1e-3
# Relative tolerance for values that are equal by construction but pass
# through float arithmetic or a file (straightness of the capital allocation
# line, h5 versus yml values).
EXACT_RELATIVE_TOLERANCE = 1e-9


class InvariantError(OptimizationError):
    """The plot data disagree with the optimization, so nothing is uploaded."""


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    """
    Results of one portfolio optimization run. All returns and risks are
    annualized arithmetically and in percent per year; the Sharpe ratio is
    annualized and dimensionless. Weights are fractions (summing to 1) in
    the order of tickers; the *_weights_pct properties give them in percent,
    keyed by ticker, as written to optimization_metadata.yml.

    frontier_weights has a row of NaN (and frontier_risks a NaN) for each
    target return the optimizer did not reach. frontier_returns are the
    target returns. tangency_* is the capital allocation line, from
    (0, annual_risk_free_rate) to the max Sharpe ratio point.
    """

    date_from: pd.Timestamp
    date_to: pd.Timestamp
    tickers: tuple[str, ...]
    annual_risk_free_rate: float
    frontier_risks: np.ndarray
    frontier_returns: np.ndarray
    frontier_weights: np.ndarray
    tangency_risks: np.ndarray
    tangency_returns: np.ndarray
    simulated_risks: np.ndarray
    simulated_returns: np.ndarray
    simulated_weights: np.ndarray
    max_sharpe_risk: float
    max_sharpe_return: float
    sharpe_ratio: float
    max_sharpe_weights: np.ndarray
    min_var_risk: float
    min_var_return: float
    min_var_weights: np.ndarray

    @property
    def max_sharpe_weights_pct(self) -> dict[str, float]:
        """Max Sharpe ratio weights in percent, keyed by ticker."""
        return _weights_pct(self.tickers, self.max_sharpe_weights)

    @property
    def min_var_weights_pct(self) -> dict[str, float]:
        """Minimum variance weights in percent, keyed by ticker."""
        return _weights_pct(self.tickers, self.min_var_weights)


def _weights_pct(tickers: tuple[str, ...], weights: np.ndarray) -> dict[str, float]:
    # Plain floats, so yaml.dump writes numbers rather than numpy tags.
    return {ticker: float(w * 100) for ticker, w in zip(tickers, weights)}


def capital_allocation_line(
    annual_risk_free_rate: float,
    max_sharpe_risk: float,
    max_sharpe_return: float,
    n_points: int = CAPITAL_ALLOCATION_LINE_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    The capital allocation line (CAL) from the risk-free asset to the max
    Sharpe ratio portfolio, and no further: portfolios beyond that point
    would need borrowing (leverage), which is out of scope. Pure function.

    The line is parameterized as t from 0 to 1, so its first point is
    exactly (0, annual_risk_free_rate), its last point is exactly
    (max_sharpe_risk, max_sharpe_return), and the risks are evenly spaced.
    Its slope is the annualized max Sharpe ratio.

    Parameters
    ----------
    annual_risk_free_rate : float
        Annualized risk-free rate, percent per year.

    max_sharpe_risk, max_sharpe_return : float
        Annualized risk and return of the max Sharpe ratio portfolio,
        percent per year.

    n_points : int, optional
        Number of points. Defaults to CAPITAL_ALLOCATION_LINE_POINTS.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Risks (xs) and returns (ys), each of shape (n_points,).
    """
    t = np.linspace(0.0, 1.0, n_points)
    return t * max_sharpe_risk, (1.0 - t) * annual_risk_free_rate + t * max_sharpe_return


def run_portfolio_optimization(
    wide_df: pd.DataFrame,
    daily_risk_free_rate: float,
    settings: PortfolioOptimizationConfig,
) -> OptimizationResult:
    """
    Compute the minimum variance, max Sharpe ratio, and maximum return
    portfolios, the efficient frontier between the first and the last, the
    random feasible portfolios, and the capital allocation line from
    adjusted close prices, all from one PortfolioProblem, and annualize
    them for the frontend. Checks the invariants before returning. Pure
    computation apart from progress printing; does no file or network I/O.

    Parameters
    ----------
    wide_df : pd.DataFrame
        Wide adjusted close prices (tickers in columns, sorted DatetimeIndex),
        as returned by data.prices.pivot_close_wide().

    daily_risk_free_rate : float
        Daily risk-free rate in percent (annual rate in percent / 252).

    settings : PortfolioOptimizationConfig
        The weight cap, the number of frontier points and random portfolios,
        and the random seed, from the configuration file.

    Returns
    -------
    OptimizationResult
        Annualized results of the run.

    Raises
    ------
    portfolio.OptimizationError
        If the minimum variance, max Sharpe ratio, or maximum return
        optimization does not converge.

    InvariantError
        If the results violate check_optimization_invariants().
    """
    print(
        f"Sanity check: Wide dataframe should have 0 missing values: {wide_df.isna().sum().sum()}"
    )
    problem = PortfolioProblem.from_prices(
        wide_df, daily_risk_free_rate, settings.weight_cap_multiple
    )
    print("Mean daily returns (%):")
    print(pd.Series(problem.daily_mean_returns, index=problem.tickers))
    print("Covariance matrix:")
    print(pd.DataFrame(problem.daily_cov, index=problem.tickers, columns=problem.tickers))
    print(f"Weight cap per ticker: {problem.weight_cap:.4f}")

    min_var_w = min_variance_weights(problem)
    max_sharpe_w = max_sharpe_weights(problem)
    max_return_w = max_return_weights(problem)
    targets, frontier_w = efficient_frontier(
        problem,
        problem.daily_mean_return(min_var_w),
        problem.daily_mean_return(max_return_w),
        settings.efficient_frontier_points,
    )
    simulated_w = problem.sample_feasible_weights(
        settings.simulated_portfolios, np.random.default_rng(settings.random_seed)
    )

    min_var = problem.portfolio_stats(min_var_w)
    max_sharpe = problem.portfolio_stats(max_sharpe_w)
    frontier = problem.portfolio_stats(frontier_w)
    simulated = problem.portfolio_stats(simulated_w)
    cal_risks, cal_returns = capital_allocation_line(
        problem.annual_risk_free_rate, max_sharpe.risk, max_sharpe.annualized_return
    )
    result = OptimizationResult(
        date_from=wide_df.index[0],
        date_to=wide_df.index[-1],
        tickers=problem.tickers,
        annual_risk_free_rate=problem.annual_risk_free_rate,
        frontier_risks=frontier.risk,
        frontier_returns=annualize_return(targets, problem.periods_per_year),
        frontier_weights=frontier_w,
        tangency_risks=cal_risks,
        tangency_returns=cal_returns,
        simulated_risks=simulated.risk,
        simulated_returns=simulated.annualized_return,
        simulated_weights=simulated_w,
        max_sharpe_risk=max_sharpe.risk,
        max_sharpe_return=max_sharpe.annualized_return,
        sharpe_ratio=max_sharpe.sharpe_ratio,
        max_sharpe_weights=max_sharpe_w,
        min_var_risk=min_var.risk,
        min_var_return=min_var.annualized_return,
        min_var_weights=min_var_w,
    )
    print("Minimum variance weights %:")
    print(result.min_var_weights_pct)
    print("Best weights %:")
    print(result.max_sharpe_weights_pct)
    print("annualized_optimum_return, annualized_optimum_risk, annualized_sharpe_ratio")
    print(result.max_sharpe_return, result.max_sharpe_risk, result.sharpe_ratio)
    check_optimization_invariants(problem, result)
    return result


def _fail(message: str) -> None:
    raise InvariantError(f"Optimization plot data check failed: {message}")


def check_optimization_invariants(
    problem: PortfolioProblem, result: OptimizationResult
) -> None:
    """
    Check that the plot data agree with the optimization, and raise
    InvariantError (stopping the run before anything is written or
    uploaded) if they do not. The tolerances are the module constants
    WEIGHT_TOLERANCE, SHARPE_TOLERANCE, RISK_RELATIVE_TOLERANCE,
    RETURN_TOLERANCE, and EXACT_RELATIVE_TOLERANCE.

    Checks:
    1. Every weight vector (random portfolios, both optimized portfolios,
       reached frontier points) satisfies the shared constraints.
    2. No random or frontier portfolio has a higher Sharpe ratio than the
       max Sharpe ratio portfolio.
    3. No random portfolio lies left of the efficient frontier: none has
       less risk than a frontier point whose target return is at most its
       own return, and none has less risk than the minimum variance
       portfolio.
    4. The minimum variance risk equals the smallest reached frontier risk.
    5. Each reached frontier point's return (from portfolio_stats()) equals
       its target return.
    6. The capital allocation line starts at (0, annual risk-free rate),
       ends exactly at the max Sharpe ratio point with no risk beyond it,
       has non-decreasing risks, and is straight with slope sharpe_ratio.

    Parameters
    ----------
    problem : PortfolioProblem
        The problem the result was computed from.

    result : OptimizationResult
        The result to check.
    """
    # 1. Constraints
    reached = ~np.isnan(result.frontier_weights).any(axis=1)
    for name, weights in (
        ("random portfolio", result.simulated_weights),
        ("max Sharpe ratio portfolio", result.max_sharpe_weights[None, :]),
        ("minimum variance portfolio", result.min_var_weights[None, :]),
        ("efficient frontier portfolio", result.frontier_weights[reached]),
    ):
        violation = problem.max_constraint_violation(weights)
        if len(violation) and not violation.max() <= WEIGHT_TOLERANCE:
            _fail(f"a {name} violates the weight constraints by {violation.max():.3g}.")

    # 2. Max Sharpe ratio is the best feasible Sharpe ratio
    sharpe_limit = result.sharpe_ratio + SHARPE_TOLERANCE * (1 + abs(result.sharpe_ratio))
    for name, returns, risks in (
        ("random portfolio", result.simulated_returns, result.simulated_risks),
        (
            "efficient frontier point",
            result.frontier_returns[reached],
            result.frontier_risks[reached],
        ),
    ):
        sharpe = (returns - result.annual_risk_free_rate) / risks
        if len(sharpe) and not sharpe.max() <= sharpe_limit:
            _fail(
                f"a {name} has Sharpe ratio {sharpe.max():.6g}, above the max "
                f"Sharpe ratio {result.sharpe_ratio:.6g}."
            )

    # 3. No random portfolio left of the efficient frontier
    if not reached.any():
        _fail("no efficient frontier target was reached.")
    targets = result.frontier_returns[reached]
    frontier_risks = result.frontier_risks[reached]
    sim_returns = result.simulated_returns[:, None]
    sim_risks = result.simulated_risks[:, None]
    left = (sim_returns >= targets[None, :]) & (
        sim_risks < frontier_risks[None, :] * (1 - RISK_RELATIVE_TOLERANCE)
    )
    if left.any():
        _fail(f"{int(left.any(axis=1).sum())} random portfolio(s) lie left of the frontier.")
    if not result.simulated_risks.min() >= result.min_var_risk * (1 - RISK_RELATIVE_TOLERANCE):
        _fail("a random portfolio has less risk than the minimum variance portfolio.")

    # 4. Minimum variance risk is the frontier's smallest risk
    smallest = frontier_risks.min()
    if not abs(smallest - result.min_var_risk) <= RISK_RELATIVE_TOLERANCE * result.min_var_risk:
        _fail(
            f"the minimum variance risk {result.min_var_risk:.6g} differs from the "
            f"smallest efficient frontier risk {smallest:.6g}."
        )

    # 5. Frontier returns equal their targets
    achieved = problem.portfolio_stats(result.frontier_weights[reached]).annualized_return
    gap = np.abs(achieved - targets).max()
    if not gap <= RETURN_TOLERANCE:
        _fail(f"an efficient frontier point misses its target return by {gap:.3g}.")

    # 6. Capital allocation line
    xs, ys = result.tangency_risks, result.tangency_returns
    if not (xs[0] == 0.0 and ys[0] == result.annual_risk_free_rate):
        _fail("the capital allocation line does not start at (0, risk-free rate).")
    if not (xs[-1] == result.max_sharpe_risk and ys[-1] == result.max_sharpe_return):
        _fail("the capital allocation line does not end at the max Sharpe ratio point.")
    if not (xs.max() == result.max_sharpe_risk and (np.diff(xs) >= 0).all()):
        _fail("the capital allocation line extends beyond the max Sharpe ratio point.")
    line = result.annual_risk_free_rate + result.sharpe_ratio * xs
    if not np.allclose(
        ys, line, rtol=EXACT_RELATIVE_TOLERANCE, atol=EXACT_RELATIVE_TOLERANCE
    ):
        _fail("the capital allocation line is not straight with slope sharpe_ratio.")


def write_plot_data_h5(result: OptimizationResult, path: str) -> None:
    """
    Write portfolio_optimization_plot_data.h5 as specified in
    docs/INTERFACE_CONTRACT.md: five groups of xs (risk) / ys (return) datasets
    plus root attributes describing the units. The tangency_line group holds
    the capital allocation line. Writes the result's values unchanged;
    computes nothing. Overwrites the file.

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
            group.create_dataset("xs", data=np.asarray(xs, dtype=np.float64))
            group.create_dataset("ys", data=np.asarray(ys, dtype=np.float64))


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


def verify_optimization_files(plot_data_path: str, metadata_path: str) -> None:
    """
    Read back the two written optimization files and check that they agree,
    raising InvariantError (before anything is uploaded) if they do not:
    the max Sharpe ratio and minimum variance points in the .h5 file equal
    the risk and annualized_return in the .yml file; the capital allocation
    line starts at (0, risk_free_rate) and ends exactly at the .h5 max
    Sharpe ratio point with no risk beyond it.

    Parameters
    ----------
    plot_data_path : str
        Local path of portfolio_optimization_plot_data.h5.

    metadata_path : str
        Local path of optimization_metadata.yml.
    """
    with h5py.File(plot_data_path, "r") as hf:
        points = {
            name: (float(hf[f"{name}/xs"][0]), float(hf[f"{name}/ys"][0]))
            for name in ("max_sharpe_ratio", "min_var_portfolio")
        }
        cal_xs = hf["tangency_line/xs"][()]
        cal_ys = hf["tangency_line/ys"][()]
    with open(metadata_path, encoding="utf-8") as f:
        metadata = yaml.safe_load(f)

    def same(a: float, b: float) -> bool:
        return bool(np.isclose(a, b, rtol=EXACT_RELATIVE_TOLERANCE, atol=0.0))

    for name, section in (
        ("max_sharpe_ratio", "optimum_portfolio"),
        ("min_var_portfolio", "min_var_portfolio"),
    ):
        risk, annualized_return = points[name]
        yml = metadata[section]
        if not (same(risk, yml["risk"]) and same(annualized_return, yml["annualized_return"])):
            _fail(f"the .h5 {name} point differs from {section} in the .yml file.")
    ms_risk, ms_return = points["max_sharpe_ratio"]
    if not (cal_xs[-1] == ms_risk and cal_ys[-1] == ms_return and cal_xs.max() == ms_risk):
        _fail("the .h5 capital allocation line does not end at the max Sharpe ratio point.")
    if not (cal_xs[0] == 0.0 and same(cal_ys[0], metadata["risk_free_rate"]["risk_free_rate"])):
        _fail("the .h5 capital allocation line does not start at (0, risk_free_rate).")
