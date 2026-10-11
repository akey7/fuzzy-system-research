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
    TRADING_DAYS_PER_YEAR,
    annualize_return,
    annualize_risk,
    annualize_sharpe_ratio,
    calc_efficient_frontier,
    calc_min_variance_portfolio,
    get_risk_free_rate,
    optimize_sharpe_ratio,
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
    # nanmax: infeasible frontier points are NaN and must not propagate.
    tangency_max_risk = np.nanmax(optimized_risks)
    tangency_xs = np.linspace(0, tangency_max_risk, 100)
    tangency_ys = (
        risk_free_rate_data["daily_risk_free_rate"] + best_sharpe_ratio * tangency_xs
    )
    # Optimization above is done in daily units. Everything written for the
    # frontend is annualized arithmetically (return × 252, risk × √252), which
    # preserves the optimal weights and keeps the tangency line straight.
    annualized_optimum_return = float(annualize_return(opt_return))
    annualized_optimum_risk = float(annualize_risk(opt_risk))
    annualized_min_var_return = float(annualize_return(min_var_return))
    annualized_min_var_risk = float(annualize_risk(min_var_risk))
    annualized_sharpe_ratio = annualize_sharpe_ratio(best_sharpe_ratio)
    print("annualized_optimum_return, annualized_optimum_risk, annualized_sharpe_ratio")
    print(annualized_optimum_return, annualized_optimum_risk, annualized_sharpe_ratio)
    portfolio_optimization_plot_data_filename = os.path.join(
        "output", "portfolio_optimization_plot_data.h5"
    )
    with h5py.File(portfolio_optimization_plot_data_filename, "w") as hf:
        hf.attrs["return_units"] = "percent_per_year"
        hf.attrs["risk_units"] = "percent_per_year"
        hf.attrs["return_type"] = "simple"
        hf.attrs["annualization"] = "arithmetic"
        hf.attrs["periods_per_year"] = TRADING_DAYS_PER_YEAR
        efficient_frontier_group = hf.create_group("efficient_frontier")
        tangency_line_group = hf.create_group("tangency_line")
        simulated_portfolios_group = hf.create_group("simulated_portfolios")
        max_sharpe_ratio_group = hf.create_group("max_sharpe_ratio")
        min_var_portfolio_group = hf.create_group("min_var_portfolio")
        efficient_frontier_group.create_dataset(
            "xs", data=annualize_risk(optimized_risks)
        )
        efficient_frontier_group.create_dataset(
            "ys", data=annualize_return(target_returns)
        )
        tangency_line_group.create_dataset("xs", data=annualize_risk(tangency_xs))
        tangency_line_group.create_dataset("ys", data=annualize_return(tangency_ys))
        simulated_portfolios_group.create_dataset(
            "xs", data=annualize_risk(simulated_risks)
        )
        simulated_portfolios_group.create_dataset(
            "ys", data=annualize_return(simulated_returns)
        )
        max_sharpe_ratio_group.create_dataset("xs", data=[annualized_optimum_risk])
        max_sharpe_ratio_group.create_dataset("ys", data=[annualized_optimum_return])
        min_var_portfolio_group.create_dataset("xs", data=[annualized_min_var_risk])
        min_var_portfolio_group.create_dataset("ys", data=[annualized_min_var_return])
    print(f"Saved {portfolio_optimization_plot_data_filename}")
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
        "annualization": {
            "method": "arithmetic",
            "periods_per_year": TRADING_DAYS_PER_YEAR,
        },
        "optimum_portfolio": {
            "annualized_return": annualized_optimum_return,
            "risk": annualized_optimum_risk,
            "sharpe_ratio": annualized_sharpe_ratio,
            "weights": best_weights_pct_dict,
        },
        "min_var_portfolio": {
            "annualized_return": annualized_min_var_return,
            "risk": annualized_min_var_risk,
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
