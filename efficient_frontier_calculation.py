import os
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas_datareader as web
from dateutil.relativedelta import relativedelta
from dates_and_downloads import DatesAndDownloads


class EfficientFrontierCalculation(DatesAndDownloads):
    def __init__(self):
        super().__init__()
        self.returns_df = None
        self.mean_returns = None
        self.cov_np = None
        self.weight_bounds = None
        self.min_var_risk = None
        self.min_var_weights = None
        self.min_var_return = None

    def download_returns_or_load_from_cache(self, tickers):
        long_df_filename = os.path.join("input", f"Year of Tickers {self.get_today_date()}.csv")
        if os.path.exists(long_df_filename):
            long_df = pd.read_csv(long_df_filename)
            long_df["datetime"] = pd.to_datetime(long_df["datetime"])
            long_df["datetime"] = long_df["datetime"].apply(
                lambda x: pd.Timestamp(x).replace(hour=23, minute=59, second=59)
            )
            long_df.set_index("datetime", inplace=True)
            long_df.sort_index(inplace=True)
        else:
            date_from = self.past_business_day(pd.Timestamp(self.get_today_date()), 253)
            date_to = self.past_business_day(pd.Timestamp(self.get_today_date()), 1).replace(
                hour=23, minute=59, second=59
            )
            print(date_from, date_to)
            long_df = self.get_tickers(tickers, date_from=date_from, date_to=date_to)
            long_df.to_csv(long_df_filename, index=True)
        wide_df = self.pivot_ticker_close_wide(long_df)
        self.returns_df = wide_df.pct_change()
        self.returns_df = self.returns_df.iloc[1:] * 100

    def calc_means_and_covariance(self):
        self.mean_returns = self.returns_df.mean()
        cov = self.returns_df.cov()
        self.cov_np = cov.to_numpy()

    def simulate_portfolios(self, n_portfolios):
        d = len(self.mean_returns)
        simulated_returns = np.zeros(n_portfolios)
        simulated_risks = np.zeros(n_portfolios)
        random_weights = []
        rand_range = 1.0

        for i in range(n_portfolios):
            weights = np.random.random(d) * rand_range - rand_range / 2  # Allows short-selling
            weights[-1] = 1 - weights[:-1].sum()
            np.random.shuffle(weights)
            random_weights.append(weights)
            simulated_return = self.mean_returns.dot(weights)
            simulated_risk = np.sqrt(weights.dot(self.cov_np).dot(weights))
            simulated_returns[i] = simulated_return
            simulated_risks[i] = simulated_risk

    def create_weight_bounds_for_optimization(self, bounds):
        d = len(self.mean_returns)
        self.weight_bounds = [bounds] * d

    def get_portfolio_variance(self, weights):
        return weights.dot(self.cov_np).dot(weights)
    
    def portfolio_weights_constraint(weights):
        return weights.sum() - 1
    
    def calc_min_var_portfolio(self):
        d = len(self.mean_returns)
        min_var_result = minimize(
            fun=self.get_portfolio_variance,
            x0=np.ones(d) / d,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints={"type": "eq", "fun": self.portfolio_weights_constraint},
        )
        print("######################################################")
        print("# MINIMUM VARIANCE PORTFOLIO OPTIMIZATION            #")
        print("######################################################")
        print(min_var_result)
        self.min_var_risk = np.sqrt(min_var_result.fun)
        self.min_var_weights = min_var_result.x
        self.min_var_return = self.min_var_weights.dot(self.mean_returns)
        print(self.min_var_risk, self.min_var_weights, self.min_var_return)

    def run(self):
        tickers = ["I:SPX", "QQQ", "VXUS", "GLD"]
        self.load_returns(tickers)
        self.calc_means_and_covariance()
        self.create_weight_bounds_for_optimization((0.5, None))
        self.calc_min_var_portfolio()
