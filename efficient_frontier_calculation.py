import os
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import h5py
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
        self.simulated_returns = None
        self.simulated_risks = None
        self.efficient_risks = None
        self.efficient_returns = None
        self.sharpe_ratio = None
        self.sharpe_weights = None
        self.sharpe_risk = None
        self.sharpe_return = None
        self.tangency_xs = None
        self.tangency_ys = None

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

    def get_portfolio_variance(self, weights):
        return weights.dot(self.cov_np).dot(weights)

    def portfolio_weights_constraint(self, weights):
        return weights.sum() - 1

    def target_returns_constraint(self, weights, target_return):
        return weights.dot(self.mean_returns) - target_return

    def negative_sharpe_ratio(self, weights):
        daily_risk_free_rate = self.get_daily_risk_free_rate()
        mean = weights.dot(self.mean_returns)
        risk = np.sqrt(weights.dot(self.cov_np).dot(weights))
        return -(mean - daily_risk_free_rate) / risk

    def calc_min_var_portfolio(self):
        def callback(xk):
            print(f"Current weights: {xk}")
            print(f"Current variance: {self.get_portfolio_variance(xk)}")

        print("######################################################")
        print("# MINIMUM VARIANCE PORTFOLIO OPTIMIZATION            #")
        print("######################################################")
        print(self.cov_np)
        eigenvalues = np.linalg.eigvals(self.cov_np)
        is_valid = np.all(eigenvalues >= -1e-10)  # Small tolerance for numerical issues
        print(is_valid)
        print(self.weight_bounds)
        d = len(self.mean_returns)
        print(d)
        min_var_result = minimize(
            fun=self.get_portfolio_variance,
            x0=np.ones(d) / d,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints={"type": "eq", "fun": self.portfolio_weights_constraint},
            callback=callback
        )
        print(min_var_result)
        self.min_var_risk = np.sqrt(min_var_result.fun)
        self.min_var_weights = min_var_result.x
        self.min_var_return = self.min_var_weights.dot(self.mean_returns)
        print(self.min_var_risk, self.min_var_weights, self.min_var_return)

    def calc_efficient_frontier(self):
        print("######################################################")
        print("# EFFICIENT FRONTIER CALCULATION                     #")
        print("######################################################")
        n_portfolios = 100
        d = len(self.mean_returns)
        self.efficient_returns = np.linspace(
            self.min_var_return, self.simulated_returns.max(), n_portfolios
        )
        constraints = [
            {
                "type": "eq",
                "fun": self.target_returns_constraint,
                "args": [self.efficient_returns[0]],
            },
            {"type": "eq", "fun": self.portfolio_weights_constraint},
        ]
        self.efficient_risks = []
        for target_return in self.efficient_returns:
            constraints[0]["args"] = [target_return]
            result = minimize(
                fun=self.get_portfolio_variance,
                x0=np.ones(d) / d,
                method="SLSQP",
                bounds=self.weight_bounds,
                constraints=constraints,
            )
            if result.status == 0:
                self.efficient_risks.append(np.sqrt(result.fun))
            else:
                print("Optimization error!", result)

    def calc_sharpe_ratio(self):
        print("######################################################")
        print("# SHARPE RATIO CALCULATION                           #")
        print("######################################################")
        d = len(self.mean_returns)
        sharpe_ratio_result = minimize(
            fun=self.negative_sharpe_ratio,
            x0=np.ones(d) / d,
            method="SLSQP",
            bounds=self.weight_bounds,
            constraints={"type": "eq", "fun": self.portfolio_weights_constraint},
        )
        print(sharpe_ratio_result)
        self.sharpe_ratio = -sharpe_ratio_result.fun
        self.sharpe_weights = sharpe_ratio_result.x
        self.sharpe_risk = np.sqrt(
            self.sharpe_weights.dot(self.cov_np).dot(self.sharpe_weights)
        )
        self.sharpe_return = self.sharpe_weights.dot(self.mean_returns)
        print(self.sharpe_ratio, self.sharpe_weights)

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
        self.calc_min_var_portfolio()
        self.calc_sharpe_ratio()
        # self.calc_efficient_frontier()
        # self.calc_tangency_line()
        # self.save_h5()


if __name__ == "__main__":
    efc = EfficientFrontierCalculation()
    efc.run()
