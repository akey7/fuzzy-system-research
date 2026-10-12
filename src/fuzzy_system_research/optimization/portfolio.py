import os
import json
import datetime
import numpy as np
from scipy.optimize import minimize
import pandas_datareader as web
from dateutil.relativedelta import relativedelta

TRADING_DAYS_PER_YEAR = 252


def annualize_return(
    daily_return: float | np.ndarray | list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray | np.float64:
    """
    Arithmetically annualize mean daily simple returns: daily × periods_per_year.

    Arithmetic (not compounded) annualization is used so that linear
    relationships between risk and return are preserved, e.g. the tangency
    line stays a straight line and the Sharpe ratio scales by
    sqrt(periods_per_year). Units are preserved, so percent per day becomes
    percent per year. NaN values pass through unchanged.

    Parameters
    ----------
    daily_return : float or array-like of float
        Mean daily simple return(s).

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    np.ndarray or np.float64
        Annualized return(s): an array with the same shape as an array-like
        input, or a NumPy scalar for a scalar input.
    """
    return np.asarray(daily_return, dtype=float) * periods_per_year


def annualize_risk(
    daily_risk: float | np.ndarray | list[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray | np.float64:
    """
    Annualize the standard deviation of daily returns: daily σ * sqrt(periods_per_year).

    Assumes daily returns are serially uncorrelated. Units are preserved, so
    percent per day becomes percent per year. NaN values pass through
    unchanged.

    Parameters
    ----------
    daily_risk : float or array-like of float
        Standard deviation(s) of daily returns.

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    np.ndarray or np.float64
        Annualized standard deviation(s): an array with the same shape as an
        array-like input, or a NumPy scalar for a scalar input.
    """
    return np.asarray(daily_risk, dtype=float) * np.sqrt(periods_per_year)


def annualize_sharpe_ratio(
    daily_sharpe_ratio: float, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """
    Annualize a Sharpe ratio computed from daily returns and a daily
    risk-free rate: daily Sharpe * sqrt(periods_per_year).

    This equals (annualized return - annualized risk-free rate) / annualized
    risk when both returns are annualized arithmetically with
    annualize_return() and risk with annualize_risk().

    Parameters
    ----------
    daily_sharpe_ratio : float
        Sharpe ratio from daily excess returns and daily risk.

    periods_per_year : int, optional
        Trading days per year. Defaults to 252.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    return float(daily_sharpe_ratio * np.sqrt(periods_per_year))


def softmax_random_distribution(D, rng: np.random.Generator | None = None):
    """
    Generates a NumPy array of D random floats that sum to 1.0 using the softmax function.

    The process involves:
    1. Generating D random numbers (from a standard normal distribution).
    2. Applying the softmax function to these numbers. Softmax converts a vector
        of numbers into a probability distribution where each element is non-negative
        and all elements sum to 1.0.

    Parameters
    ----------
    D : int
        The desired number of elements (dimension) in the output array.
        Must be a positive integer.

    rng : np.random.Generator, optional
        Random number generator to draw from, e.g. for reproducible tests.
        If None (the default), NumPy's global random state is used.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (D,) containing floats that sum to 1.0.

    Raises
    ------
    ValueError: If D is not a positive integer.
    """
    if not isinstance(D, int) or D <= 0:
        raise ValueError("Dimension D must be a positive integer.")
    random_inputs = np.random.randn(D) if rng is None else rng.standard_normal(D)
    stable_inputs = random_inputs - np.max(random_inputs)
    exponentials = np.exp(stable_inputs)
    softmax_output = exponentials / np.sum(exponentials)
    return softmax_output


def simulate_portfolios(
    tdm,
    mean_returns,
    cov_np,
    n_portfolios=10_000,
    rng: np.random.Generator | None = None,
):
    """
    Monte Carlo simulation of portfolios to plot as dots on the efficient
    frontier.

    Parameters
    ----------
    tdm : TickerDownloadManager
        Ticker data. Only its tickers attribute (its length) is used, so any
        object with a tickers sequence works.

    mean_returns : np.array of float
        Mean returns of the portfolio.

    cov_np : np.array of float
        Covariance matrix.

    n_portfolios : int, optional
        Number of portfolios to simulate, defaults to 10,000.

    rng : np.random.Generator, optional
        Random number generator for reproducible simulations. If None (the
        default), NumPy's global random state is used.

    Returns
    -------
    list[float], list[float]
        Lists of simulated risks and their corresponding returns.
    """
    simulated_returns = np.zeros(n_portfolios)
    simulated_risks = np.zeros(n_portfolios)
    random_weights = []
    rand_range = 1.0
    for i in range(n_portfolios):
        D = len(tdm.tickers)
        # w = np.random.random(D) * rand_range - rand_range / 2  # Allows short selling
        # w[-1] = 1 - w[:-1].sum()
        # np.random.shuffle(w)
        w = softmax_random_distribution(D, rng=rng)  # No short selling
        random_weights.append(w)
        simulated_return = mean_returns.dot(w)
        simulated_risk = np.sqrt(w.dot(cov_np).dot(w))
        simulated_returns[i] = simulated_return
        simulated_risks[i] = simulated_risk
    return simulated_risks, simulated_returns


def calc_weight_bounds(tdm):
    """
    Calculate a list of bounds for the weight of each asset in the portfolio
    defined by the given TickerDownloadManager according to the constraints
    imposed by each of the lines that are commented out.

    Parameters
    ----------
    tdm : TickerDownloadManager
        Where the tickers to be optimized are.
    """
    D = len(tdm.tickers)
    # weight_bounds = [(-0.5, None)] * D  # Allows shorting
    # weight_bounds = [(0.0, 1.0) for _ in range(D)]  # No shorting, no leverage
    return [
        (0.0, 4.0 / D) for _ in range(D)
    ]  # Limit how much can be invested in one asset, no shorting, no leverage


def calc_min_variance_portfolio(tdm, cov_np, mean_returns):
    """
    Optimize the minimum variance portfolio.

    Parameters
    ----------
    tdm : TickerDownloadManager
        Ticker data

    cov_np : np.array of float
        Covariance matrix.

    mean_returns : np.array of float
        Mean returns of the portfolio assets.

    Returns
    -------
    float, float, float
        Minimum variance risk, minimum variance weights, minimum variance return.
    """
    weight_bounds = calc_weight_bounds(tdm)
    D = len(tdm.tickers)

    def get_portfolio_variance(weights):
        return weights.dot(cov_np).dot(weights)

    def portfolio_weights_constraint(weights):
        return weights.sum() - 1

    min_var_result = minimize(
        fun=get_portfolio_variance,
        x0=np.ones(D) / D,
        method="SLSQP",
        bounds=weight_bounds,
        constraints={"type": "eq", "fun": portfolio_weights_constraint},
    )
    print("Minimum variance portfolio optimization result:")
    print(min_var_result)
    min_var_risk = np.sqrt(min_var_result.fun)
    min_var_weights = min_var_result.x
    min_var_return = min_var_weights.dot(mean_returns)
    print(
        f"min_var_risk={min_var_risk}, min_var_weights={min_var_weights}, min_var_return={min_var_return}"
    )
    return min_var_risk, min_var_weights, min_var_return


def calc_efficient_frontier(
    tdm, min_var_return, max_simulated_return, mean_returns, cov_np, num_portfolios=100
):
    """
    Calculates the efficient frontier for assets in TickerDownloadManager tdm.
    Essentially calculates the points of the efficient frontier from the
    minimum variance return and the desired maximum simulated return, with
    the given mean returns and covariance matrix for the portfolio. Prints
    error conditions to stdout.

    Parameters
    ----------
    tdm : TickerDownloadManager
        Contains the prices of the tickers in the download manager.

    min_var_return : float
        The minimum variance return located at one extreme of the frontier

    max_simulated_return : float
        The maximum simulated return (from the Monte Carlo runs) at the other
        extreme of the frontier.

    mean_returns : float
        Mean returns of each asset.

    cov_np : np.array of float
        Covariance matrix of the portfolio.

    num_portfolios : int, optional
        The number of portfolio to calculate on the efficient frontier.

    Returns
    -------
    tuple[list[float], list[float]]
        Returns two lists. The first list has optimized risks, the other
        list has corresponding target returns.
    """
    D = len(tdm.tickers)
    print(f"Possible returns range: {min_var_return:.4f} to {max_simulated_return:.4f}")
    target_returns = np.linspace(min_var_return, max_simulated_return, num_portfolios)

    def target_returns_constraint(weights, target_return):
        return weights.dot(mean_returns) - target_return

    def portfolio_weights_constraint(weights):
        return weights.sum() - 1

    def get_portfolio_variance(weights):
        return weights.dot(cov_np).dot(weights)

    constraints = [
        {"type": "eq", "fun": target_returns_constraint, "args": [target_returns[0]]},
        {"type": "eq", "fun": portfolio_weights_constraint},
    ]
    optimized_risks = []
    for target_return in target_returns:
        constraints[0]["args"] = [target_return]
        result = minimize(
            fun=get_portfolio_variance,
            x0=np.ones(D) / D,
            method="SLSQP",
            bounds=calc_weight_bounds(tdm),
            constraints=constraints,
        )
        if result.status == 0:
            optimized_risks.append(np.sqrt(result.fun))
        else:
            optimized_risks.append(np.nan)
            print(f"Infeasible target return: {target_return:.4f}")
    return optimized_risks, target_returns


def get_risk_free_rate(dm):
    """
    Finds the risk-free rates of the 3-month U.S. treasury bill on the business
    day closest to the current day. Caches the downloaded rates so that repeated
    calls to this function only retrieve the data when necessary.

    Returns
    -------
    dict[str, float]
        Returns the risk free annual rate, the risk free daily rate, and the date
        of those rates.
    """
    today_date = dm.get_today_date()
    risk_free_rate_filename = os.path.join("input", f"Risk Free Rate {today_date}.json")
    if os.path.exists(risk_free_rate_filename):
        print("Reading risk-free rate cache...")
        with open(risk_free_rate_filename, "r", encoding="utf-8") as f:
            risk_free_rate_data = json.load(f)
            print(risk_free_rate_data)
            daily_risk_free_rate = risk_free_rate_data["daily_risk_free_rate"]
    else:
        end_date = datetime.datetime.now()
        start_date = end_date - relativedelta(years=1)
        print(start_date, end_date)
        tb3m_df = web.DataReader("DTB3", "fred", start_date, end_date).sort_values(
            "DATE", ascending=False
        )
        risk_free_rate = float(tb3m_df.iloc[0]["DTB3"])
        daily_risk_free_rate = risk_free_rate / 252
        risk_free_rate_date = str(tb3m_df.index[0])
        print(daily_risk_free_rate)
        risk_free_rate_data = {
            "risk_free_rate": risk_free_rate,
            "daily_risk_free_rate": daily_risk_free_rate,
            "risk_free_rate_date": risk_free_rate_date,
        }
        with open(risk_free_rate_filename, "w", encoding="utf-8") as f:
            json.dump(risk_free_rate_data, f, indent=4)
    return risk_free_rate_data


def optimize_sharpe_ratio(tdm, daily_risk_free_rate, mean_returns, cov_np):
    """
    Optimize a portfolio with the given tickers for the maximum Sharpe ratio
    given the provided mean returns, daily risk-free rate, and covariance
    matrix.

    Parameters
    ----------
    tdm : TickerDownloadManager
        Price data for tickers in the given portfolio.

    daily_risk_free_rate : float
        Daily risk free rate.

    mean_returns : np.array of float
        The mean returns of the portfolio.

    cov_np : np.array of float
        Covariance matrix.

    Returns
    -------
    best_sharpe_ratio, best_weights, opt_risk, opt_return
        Optimized Sharpe ratio, portfolio rates, optimal portfolio risk,
        and optimal return.
    """
    D = len(tdm.tickers)

    def negative_sharpe_ratio(weights):
        mean = weights.dot(mean_returns)
        risk = np.sqrt(weights.dot(cov_np).dot(weights))
        return -(mean - daily_risk_free_rate) / risk

    def portfolio_weights_constraint(weights):
        return weights.sum() - 1

    sharpe_ratio_result = minimize(
        fun=negative_sharpe_ratio,
        x0=np.ones(D) / D,
        method="SLSQP",
        bounds=calc_weight_bounds(tdm),
        constraints={"type": "eq", "fun": portfolio_weights_constraint},
    )
    best_sharpe_ratio = -sharpe_ratio_result.fun
    best_weights = sharpe_ratio_result.x
    opt_risk = np.sqrt(best_weights.dot(cov_np).dot(best_weights))
    opt_return = best_weights.dot(mean_returns)
    return best_sharpe_ratio, best_weights, opt_risk, opt_return
