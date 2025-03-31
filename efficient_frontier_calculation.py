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

    def run(self):
        tickers = ["I:SPX", "QQQ", "VXUS", "GLD"]
        self.load_returns(tickers)
        self.calc_means_and_covariance()
