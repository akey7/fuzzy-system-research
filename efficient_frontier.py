import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import h5py
from dates_and_downloads import DatesAndDownloads


def optimize_min_var_portfolio(mean_returns, cov):
    D = len(mean_returns)
    x0 = np.ones(D) / D
    eigvals = np.linalg.eigvals(cov)
    if not np.all(eigvals > -1e-10):  # Allow for small numerical errors
        print("Warning: Covariance matrix is not positive semi-definite")
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0) for _ in range(D)]
    min_var_result = minimize(
        fun=lambda w: w.dot(cov).dot(w),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'disp': True}
    )
    print(min_var_result)
    return min_var_result.x



class EfficientFrontier(DatesAndDownloads):
    def __init__(self):
        super().__init__()

    def download_returns_or_load_from_cache(self, tickers):
        long_df_filename = os.path.join(
            "input", f"Year of Tickers {self.get_today_date()}.csv"
        )
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
            date_to = self.past_business_day(
                pd.Timestamp(self.get_today_date()), 1
            ).replace(hour=23, minute=59, second=59)
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
        self.simulated_returns = np.zeros(n_portfolios)
        self.simulated_risks = np.zeros(n_portfolios)
        random_weights = []
        rand_range = 1.0

        for i in range(n_portfolios):
            weights = (
                np.random.random(d) * rand_range - rand_range / 2
            )  # Allows short-selling
            weights[-1] = 1 - weights[:-1].sum()
            np.random.shuffle(weights)
            random_weights.append(weights)
            simulated_return = self.mean_returns.dot(weights)
            simulated_risk = np.sqrt(weights.dot(self.cov_np).dot(weights))
            self.simulated_returns[i] = simulated_return
            self.simulated_risks[i] = simulated_risk

    def create_weight_bounds_for_optimization(self, bounds):
        d = len(self.mean_returns)
        self.weight_bounds = [bounds] * d

    def calc_tangency_line(self):
        daily_risk_free_rate = self.get_daily_risk_free_rate()
        tangency_max_risk = max(self.efficient_risks)
        self.tangency_xs = np.linspace(0, tangency_max_risk, 100)
        self.tangency_ys = daily_risk_free_rate + self.sharpe_ratio * self.tangency_xs

    def save_h5(self):
        today_date = self.get_today_date()
        h5_filename = os.path.join(
            "output", f"Efficient Frontier Plot Data {today_date}.h5"
        )
        with h5py.File(h5_filename, "w") as hf:
            mean_returns_group = hf.create_group("mean")
            min_var_group = hf.create_group("min_var")
            simulated_portfolios_group = hf.create_group("simulated_portfolios")
            efficient_frontier_group = hf.create_group("efficient_frontier")
            sharpe_group = hf.create_group("sharpe")
            tangency_line_group = hf.create_group("tangency_line")
            mean_returns_group.create_dataset("returns", data=self.mean_returns)
            min_var_group.create_dataset("risk", data=self.min_var_risk)
            min_var_group.create_dataset("return", data=self.min_var_return)
            simulated_portfolios_group.create_dataset(
                "returns", data=self.simulated_returns
            )
            simulated_portfolios_group.create_dataset(
                "risks", data=self.simulated_risks
            )
            efficient_frontier_group.create_dataset("risks", data=self.efficient_risks)
            efficient_frontier_group.create_dataset(
                "returns", data=self.efficient_returns
            )
            sharpe_group.create_dataset("ratio", data=self.sharpe_ratio)
            sharpe_group.create_dataset("risk", data=self.sharpe_risk)
            sharpe_group.create_dataset("return", data=self.sharpe_return)
            tangency_line_group.create_dataset("xs", data=self.tangency_xs)
            tangency_line_group.create_dataset("ys", data=self.tangency_ys)
        print(f"Saved {h5_filename}")

    def run(self):
        tickers = ["I:SPX", "QQQ", "VXUS", "GLD"]
        self.download_returns_or_load_from_cache(tickers)
        self.calc_means_and_covariance()
        self.simulate_portfolios(1000)
        self.create_weight_bounds_for_optimization((0.5, None))
        optimize_min_var_portfolio(self.mean_returns, self.cov_np)
        # self.calc_sharpe_ratio()
        # self.calc_efficient_frontier()
        # self.calc_tangency_line()
        # self.save_h5()


if __name__ == "__main__":
    ef = EfficientFrontier()
    ef.run()
