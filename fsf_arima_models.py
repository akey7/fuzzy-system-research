import warnings
from dask.distributed import Client, as_completed
from tqdm import tqdm
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error


class ArimaModels:
    def __init__(self, n_workers=6):
        """
        Initialize a new ArimaModels instance.

        Parameters
        ----------
        n_workers : int, optional
            Number of workers to allocate to fitting ARIMA models.
            Optional and defaults to 6.
        """
        self.n_workers = n_workers
        self.train_result = None
        self.final_model = None

    def next_business_day_skip_holidays(self, date):
        """
        Finds the next business day, skipping US federal holidays.

        Parameters
        ----------
        date : pd.Timestamp
            The starting date.

        Returns
        -------
        pd.Timestamp
            The next business day, skipping holidays.
        """

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=date, end=date + pd.Timedelta(days=365))

        next_day = date + pd.Timedelta(days=1)
        while True:
            if next_day.weekday() >= 5 or next_day in holidays:
                next_day += pd.Timedelta(days=1)
            else:
                return next_day

    def fit(self, original_ts, max_p=5, max_q=5, train_len=90):
        """
        Fit many ARIMA models to find the best p and q values.
        Fitting is parrallelized across the number of workers specified
        by n_workers specified during instantation. The original time
        series is log-transformed and differenced with a lag of 1
        before fitting the models. max_p, max_q, and train_len
        are passed to the other fitting functions as appropriate. This
        function does not return a value; rather, it sets instance
        attributes with the results of the operations.

        This assumes the frequency of the index is "B" and will
        predict the next business day at each training.

        Parameters
        ----------
        original_ts : pd.Series
            The original time series.

        max_p : int, optional
            The maximum number of lagged values to test. Defaults to 5.

        max_q : int, optional
            The maximum number of lagged disturbances to test. Defaults to 5.

        train_len : int, optional
            Then length of the training set. Defaults to 120.

        Returns
        -------
        None
        """
        ts = self.log_diff_prep(original_ts)
        self.train_result = self.train_arima(
            ts, max_p=max_p, max_q=max_q, train_len=train_len, d=0
        )
        self.final_model = self.best_arima_model(ts, self.train_result)
        pred_date = self.next_business_day_skip_holidays(ts.index[-1])
        pred = self.predict(original_ts)
        return pred_date, pred

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
        # print(best_p, best_q)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            best_arima_model = tsa.ARIMA(endog=ts, order=(best_p, 0, best_q)).fit()
        return best_arima_model

    def predict(self, orginal_ts):
        """
        Predict the next time step beyond the training of the model.
        Return the prediction in units of the original time series.

        Parameters
        ----------
        original_ts : pd.Series
            Original time series, neither transformed nor differenced

        Returns
        -------
        np.float64
            The next value in the time series.
        """
        forecast = self.final_model.forecast(steps=1)
        pred = orginal_ts.iloc[-1] * np.exp(forecast.iloc[0])
        return pred
