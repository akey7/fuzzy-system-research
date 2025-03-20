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


def log_diff_prep(original_ts):
    """
    Prepare a time series for modeling by log transforming and
    differencing. The date time index is passed through 
    unaffected.

    Parameters
    ----------
    original_ts : pd.Series
        The original time series to be transformed.

    Returns
    -------
    pd.Series
        The transformed time series.
    """
    return np.log(original_ts).diff()


def train_arima(ts, max_p=5, d=0, max_q=5, train_len=120, n_workers=6):
    """
    Initiate training runs of ARIMA models with ps and qs under the
    given maximums to find the best pa and q values for the modeling
    the given time series.

    Parameters
    ----------
    ts : pd.Series
        The time series to be modeled. The raw data should be transformed
        by log_diff_prep() first.

    max_p : int, optional
        The maximum number of lagged values to test. Defaults to 5.

    max_q : int, optional
        The maximum number of lagged disturbances to test. Defaults to 5.

    train_len : int, optional
        Then length of the training set. Defaults to 120.

    n_workers : int, optional
        Number of cores to use for parallel evaluation of models.
        Defaults to 6.

    Returns
    -------
    pd.DataFrame
        Returns a DataFrame, meant for analysis by other functions in this
        module, that is the results of training with different p and q values.
    """
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


def p_d_q_fit(task):
    """
    Train and evaluate an ARIMA model with the given p, d, and q parameters
    in the task dictionary.

    Meant to be used by the train_arima function to perform parrallizable
    tasks on cores.

    Paremters
    ---------
    task : dict
        Dictionary that defines the model training task from train_arima.

    Returns
    -------
    dict
        Returns a dictionary, suitable to be used as one row of the results
        DataFrame returned by train_arima, that contains the results of the
        training.
    """
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


def train_arima_viz(train_result):
    """
    In a Jupyter notebook, display heatmaps of p and q parameters and their
    RMSE and BIC values to visualize which models did best.

    Parameters
    ----------
    train_result : pd.DataFrame
        The results of training a bunch of ARIMA models as returned by
        train_arima.
    """
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
    """
    Use training results from train_arima and make a final ARIMA
    model that has the best p and q values, as determined by their
    RMSE and BIC metrics.
    """
    best_p, best_q = train_result.rank().loc[:, ["rmse", "mean_bic"]].mean(1).idxmin()
    print(best_p, best_q)
    best_arima_model = tsa.ARIMA(endog=ts, order=(best_p, 0, best_q)).fit()
    return best_arima_model


def plot_correlogram(ts0, nlags, title, residual_rolling=21, acf_plot_ymax=0.1):
    """
    In a Jupyter notebook, display the correlogram with ACF, PACF, and residuals QQ
    plot and time series. Also displays Q and ADF stats and the moments of the
    residual distribution. This can assist with determining proper p and q ranges to
    try and/or looking at the quality of the best fit model.

    Parameters
    ----------
    ts0
        Time series to be plotted.

    nlags : int
        Number of lags in the ACF and PACF diagrams.

    residual_rolling : int, optional
        The window for the rolling average in the residual plot.
        Defaults to 21

    acf_plot_ymax : float, optional
        The y limits on the ACF and PACF plots will match the
        magnitude of this value. Should be > 0.0. Defaults to 0.1
    """
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
