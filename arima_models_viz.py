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


class ArimaModelsViz:
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
