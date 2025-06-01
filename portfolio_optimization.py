import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from scipy.optimize import minimize
import pandas_datareader as web
from dateutil.relativedelta import relativedelta
import h5py
import yaml
from ticker_download_manager import TickerDownloadManager
from date_manager import DateManager
from ticker_predict_upload import TickerPredictUpload
from s3_uploader import S3Uploader


def softmax_random_distribution(D):
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
    random_inputs = np.random.randn(D)
    stable_inputs = random_inputs - np.max(random_inputs)
    exponentials = np.exp(stable_inputs)
    softmax_output = exponentials / np.sum(exponentials)
    return softmax_output


def simulate_portfolios(tdm, mean_returns, cov_np, n_portfolios=10_000):
    simulated_returns = np.zeros(n_portfolios)
    simulated_risks = np.zeros(n_portfolios)
    random_weights = []
    rand_range = 1.0
    for i in range(n_portfolios):
        D = len(tdm.tickers)
        # w = np.random.random(D) * rand_range - rand_range / 2  # Allows short selling
        # w[-1] = 1 - w[:-1].sum()
        # np.random.shuffle(w)
        w = softmax_random_distribution(D)  # No short selling
        random_weights.append(w)
        simulated_return = mean_returns.dot(w)
        simulated_risk = np.sqrt(w.dot(cov_np).dot(w))
        simulated_returns[i] = simulated_return
        simulated_risks[i] = simulated_risk
    return simulated_risks, simulated_returns


def calc_min_variance_portfolio(tdm, cov_np, mean_returns):
    D = len(tdm.tickers)
    # weight_bounds = [(-0.5, None)] * D  # Allows shorting
    # weight_bounds = [(0.0, 1.0) for _ in range(D)]  # No shorting, no leverage
    weight_bounds = [
        (0.0, 4.0 / D) for _ in range(D)
    ]  # Limit how much can be invested in one asset, no shorting, no leverage

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
    print(f"min_var_risk={min_var_risk}, min_var_weights={min_var_weights}, min_var_return={min_var_return}")
    return min_var_risk, min_var_weights, min_var_return


def main():
    tdm = TickerDownloadManager(os.path.join("input", "annual"))
    dm = DateManager()
    tpu = TickerPredictUpload()
    long_df, start_date, end_date = tdm.get_latest_tickers(
        days_in_past=252, use_cache=True
    )
    print(f"Loaded data from {start_date} to {end_date}")
    wide_df = tpu.pivot_ticker_close_wide(long_df)
    date_from = wide_df.index[0]
    date_to = wide_df.index[-1]
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
    simulated_risks, simulated_returns = simulate_portfolios(tdm, mean_returns, cov_np)
    min_var_risk, min_var_weights, min_var_return = calc_min_variance_portfolio(
        tdm, cov_np, mean_returns
    )


if __name__ == "__main__":
    main()
