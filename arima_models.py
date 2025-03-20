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


class ArimaModels:
    def __init__(self, n_workers=6):
        self.train_result = None
        self.n_workers = n_workers

    def fit(original_ts, max_p=5, d=0, max_q=5, train_len=120):
        pass

    def log_diff_prep(self, original_ts):
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

    def train_arima(self, ts, max_p=5, d=0, max_q=5, train_len=120, n_workers=6):
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
        futures = [dask_client.submit(self.p_d_q_fit, task) for task in tasks]
        arima_fit_results = []
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing", unit="task"
        ):
            arima_fit_results.append(future.result())
        dask_client.close()
        arima_fit_df = pd.DataFrame(arima_fit_results)
        arima_fit_df.set_index(["p", "q"], inplace=True)
        return arima_fit_df

    def p_d_q_fit(self, task):
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

    def best_arima_model(self, ts, train_result):
        """
        Use training results from train_arima and make a final ARIMA
        model that has the best p and q values, as determined by their
        RMSE and BIC metrics.
        """
        best_p, best_q = (
            train_result.rank().loc[:, ["rmse", "mean_bic"]].mean(1).idxmin()
        )
        print(best_p, best_q)
        best_arima_model = tsa.ARIMA(endog=ts, order=(best_p, 0, best_q)).fit()
        return best_arima_model

    def predict_1_step(self, orginal_ts, best_arima_model):
        """
        Predict the next time step beyond the training of the model.
        Return the prediction in units of the original time series.

        Parameters
        ----------
        original_ts : pd.Series
            Original time series, neither transformed nor differenced

        best_arima_model : ARIMA
            The best trained ARIMA model as returned from best_arima_model.

        Returns
        -------
        np.float64
            The next value in the time series.
        """
        forecast = best_arima_model.forecast(steps=1)
        pred = orginal_ts.iloc[-1] * np.exp(forecast.iloc[0])
        return pred
