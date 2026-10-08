import os
import h5py
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.data.ticker_download_manager import TickerDownloadManager
from fuzzy_system_research.forecasting.ticker_predict import TickerPredictUpload
from fuzzy_system_research.optimization.portfolio import (
    calc_efficient_frontier,
    calc_min_variance_portfolio,
    get_risk_free_rate,
    optimize_sharpe_ratio,
    plot_optimization_results,
    simulate_portfolios,
)
from fuzzy_system_research.storage.s3_uploader import S3Uploader


def main():
    """
    Run the whole ARIMA modeling and portfolio optimization workflow, including
    uploading results to S3 for the frontend.
    """
    tdm = TickerDownloadManager(os.path.join("input", "annual"))
    dm = DateManager()
    tpu = TickerPredictUpload()
    load_dotenv()
    s3u = S3Uploader()
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
    print(
        f"min_var_risk={min_var_risk}, min_var_weights={min_var_weights}, min_var_return={min_var_return}"
    )
    optimized_risks, target_returns = calc_efficient_frontier(
        tdm, min_var_return, max(simulated_returns), mean_returns, cov_np
    )
    print("Efficient frontier target returns")
    print(target_returns)
    print("Efficient frontier optimized risks")
    print(optimized_risks)
    risk_free_rate_data = get_risk_free_rate(dm)
    print(risk_free_rate_data)
    best_sharpe_ratio, best_weights, opt_risk, opt_return = optimize_sharpe_ratio(
        tdm, risk_free_rate_data["daily_risk_free_rate"], mean_returns, cov_np
    )
    print(f"best_sharpe_ratio={best_sharpe_ratio}, best_weights={best_weights}")
    best_weights_pct_dict = pd.Series(
        best_weights * 100, index=mean_returns.index
    ).to_dict()
    print("Best weights %:")
    print(best_weights_pct_dict)
    tangency_max_risk = max(optimized_risks)
    tangency_xs = np.linspace(0, tangency_max_risk, 100)
    tangency_ys = (
        risk_free_rate_data["daily_risk_free_rate"] + best_sharpe_ratio * tangency_xs
    )
    result_fig = plot_optimization_results(
        optimized_risks,
        target_returns,
        tangency_xs,
        tangency_ys,
        simulated_risks,
        simulated_returns,
        min_var_risk,
        min_var_return,
        opt_risk,
        opt_return,
    )
    result_fig_filename = os.path.join("output", "portfolio_optimization_plot.png")
    result_fig.savefig(result_fig_filename, dpi=300, bbox_inches="tight")
    print(f"{result_fig_filename} saved!")
    portfolio_optimization_plot_data_filename = os.path.join(
        "output", "portfolio_optimization_plot_data.h5"
    )
    with h5py.File(portfolio_optimization_plot_data_filename, "w") as hf:
        efficient_frontier_group = hf.create_group("efficient_frontier")
        tangency_line_group = hf.create_group("tangency_line")
        simulated_portfolios_group = hf.create_group("simulated_portfolios")
        max_sharpe_ratio_group = hf.create_group("max_sharpe_ratio")
        min_var_portfolio_group = hf.create_group("min_var_portfolio")
        efficient_frontier_group.create_dataset("xs", data=optimized_risks)
        efficient_frontier_group.create_dataset("ys", data=target_returns)
        tangency_line_group.create_dataset("xs", data=tangency_xs)
        tangency_line_group.create_dataset("ys", data=tangency_ys)
        simulated_portfolios_group.create_dataset("xs", data=simulated_risks)
        simulated_portfolios_group.create_dataset("ys", data=simulated_returns)
        max_sharpe_ratio_group.create_dataset("xs", data=[opt_risk])
        max_sharpe_ratio_group.create_dataset("ys", data=[opt_return])
        min_var_portfolio_group.create_dataset("xs", data=[min_var_risk])
        min_var_portfolio_group.create_dataset("ys", data=[min_var_return])
    print(f"Saved {portfolio_optimization_plot_data_filename}")
    annualized_optimum_return = ((1 + opt_return / 100) ** 252 - 1) * 100
    annualized_optimum_risk = opt_risk * np.sqrt(252)
    print("annualized_optimum_return, annualized_optimum_risk")
    print(annualized_optimum_return, annualized_optimum_risk)
    space_name = os.getenv("PORTFOLIO_OPTIMIZATION_SPACE_NAME")
    s3u.upload_file(
        portfolio_optimization_plot_data_filename,
        space_name,
        "portfolio_optimization_plot_data.h5",
    )
    metadata = {
        "date_updated": {
            "date_from": str(date_from.date()),
            "date_to": str(date_to.date()),
        },
        "tickers": tdm.tickers,
        "risk_free_rate": risk_free_rate_data,
        "optimum_portfolio": {
            "annualized_return": float(annualized_optimum_return),
            "risk": float(annualized_optimum_risk),
            "weights": best_weights_pct_dict,
        },
    }
    metadata_filename = os.path.join("output", "optimization_metadata.yml")
    with open(metadata_filename, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    s3u.upload_file(
        metadata_filename,
        space_name,
        "optimization_metadata.yml",
    )


if __name__ == "__main__":
    main()
