import warnings
from dask.distributed import Client, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt
import statsmodels.tsa.api as tsa
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from scipy.stats import probplot, moment
from sklearn.metrics import mean_squared_error
import seaborn as sns


def p_d_q_fit(task):
    ts = task["ts"]
    p = task["p"]
    d = task["d"]
    q = task["q"]
    train_len = task["train_len"]
    y_true = ts[train_len:]
    y_pred = []
    bics = []
    aics = []
    n_convergence_errors = 0
    n_stationarity_errors = 0
    for bound in range(train_len, len(ts)):
        train_set = ts.iloc[bound - train_len : bound]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                model = tsa.ARIMA(endog=train_set, order=(p, d, q)).fit()
            except LinAlgError:
                n_convergence_errors += 1
            except ValueError:
                n_stationarity_errors += 1
            forecast = model.forecast(steps=1)
            y_pred.append(forecast.iloc[0])
            aics.append(model.aic)
            bics.append(model.bic)
    df = (
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        .replace(np.inf, np.nan)
        .dropna()
    )
    rmse = np.sqrt(mean_squared_error(y_true=df["y_true"], y_pred=df["y_pred"]))
    result = {
        "p": p,
        "q": q,
        "rmse": rmse,
        "mean_aic": np.mean(aics),
        "mean_bic": np.mean(bics),
        "n_convergence_errors": n_convergence_errors,
        "n_stationarity_errors": n_stationarity_errors,
    }
    return result


def train_arima(ts, max_p=5, d=1, max_q=5, train_len=120, n_workers=6):
    tasks = []
    for p in range(max_p):
        for q in range(max_q):
            if p == 0 and q == 0:
                continue
            tasks.append(
                {
                    "ts": ts,
                    "d": d,
                    "train_len": train_len,
                    "p": p,
                    "q": q,
                }
            )
    dask_client = Client(n_workers=n_workers)
    futures = [dask_client.submit(p_d_q_fit, task) for task in tasks]
    arima_fit_results = []
    for future in tqdm(
        as_completed(futures), total=len(futures), desc="Processing", unit="task"
    ):
        arima_fit_results.append(future.result())
    dask_client.close()
    arima_fit_df = pd.DataFrame(arima_fit_results)
    arima_fit_df.set_index(["p", "q"], inplace=True)
    return arima_fit_df


def train_arima_viz(train_result):
    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharex=True, sharey=True)
    sns.heatmap(
        train_result[train_result.rmse < 0.5].rmse.unstack().mul(10),
        fmt=".3f",
        annot=True,
        cmap="Blues",
        ax=axes[0],
        cbar=False,
    )
    sns.heatmap(
        train_result.mean_bic.unstack(),
        fmt=".2f",
        annot=True,
        cmap="Blues",
        ax=axes[1],
        cbar=False,
    )
    axes[0].set_title("Root Mean Squared Error")
    axes[1].set_title("Mean Bayesian Information Criterion")
    fig.tight_layout()


def best_arima_model(ts, train_result):
    best_p, best_q = train_result.rank().loc[:, ["rmse", "mean_bic"]].mean(1).idxmin()
    print(best_p, best_q)
    best_arima_model = tsa.ARIMA(endog=ts, order=(best_p, 0, best_q)).fit()
    return best_arima_model


def plot_correlogram(ts0, nlags, title, residual_rolling=21, acf_plot_ymax=0.1):
    ts = ts0.dropna()
    q_p = np.max(q_stat(acf(ts, nlags=nlags), len(ts))[1])
    stats = f"Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(ts)[1]:>11.2f}"
    mean, var, skew, kurtosis = moment(ts, moment=[1, 2, 3, 4])
    qq_stats = f"Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}"
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    axs[0][0].plot(ts)
    axs[0][0].plot(ts.rolling(residual_rolling).mean(), color="black")
    axs[0][0].text(x=0.02, y=0.85, s=stats, transform=axs[0][0].transAxes)
    axs[0][0].set_title(f"Residuals and {residual_rolling}-day rolling mean")
    probplot(ts, plot=axs[0][1])
    axs[0][1].text(x=0.02, y=0.75, s=qq_stats, transform=axs[0][1].transAxes)
    axs[0][1].set_title("Q-Q")
    plot_acf(ts, lags=nlags, zero=False, ax=axs[1][0])
    axs[1][0].set_xlabel("Lag")
    axs[1][0].set_ylim(-acf_plot_ymax, acf_plot_ymax)
    plot_pacf(ts, lags=nlags, zero=False, ax=axs[1][1])
    axs[1][1].set_xlabel("Lag")
    axs[1][1].set_ylim(-acf_plot_ymax, acf_plot_ymax)
    fig.suptitle(f"{title}")
    fig.tight_layout()
