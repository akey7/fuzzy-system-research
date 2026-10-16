import warnings
import pandas as pd
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.tsa.api as tsa
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from fuzzy_system_research.data.date_manager import DateManager

# Number of times prices are differenced before modeling: once, by
# ArimaModels.to_log_returns(). The models are ARIMA(p, 0, q) on log returns,
# which is ARIMA(p, PRICE_DIFFERENCING_ORDER, q) on log prices.
PRICE_DIFFERENCING_ORDER = 1


class ArimaModels:
    def __init__(self):
        """
        Initialize a new ArimaModels instance to manage ARIMA models.

        Everything here runs in the calling process. Parallelism lives one
        level up, in ArimaWalkForward, which fits many windows at once.
        """
        self.train_result = None
        self.final_model = None
        self.order_tie_broken = None

    def next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """
        Finds the next NYSE trading day, skipping weekends and NYSE holidays.
        The time of day of the given date is preserved.

        Parameters
        ----------
        date : pd.Timestamp
            The starting date.

        Returns
        -------
        pd.Timestamp
            The next NYSE trading day after the given date.
        """
        return DateManager().future_business_day(date, 1)

    def fit(self, original_ts, max_p=5, max_q=5, train_len=90):
        """
        Fit many ARIMA models to find the best p and q values, one
        candidate order after another in this process. The prices in original_ts are
        converted to log returns by to_log_returns(), the only differencing
        step, before fitting the models. max_p, max_q, and train_len
        are passed to the other fitting functions as appropriate. It also
        sets instance attributes with the training results, the final model,
        and order_tie_broken (whether select_order() needed its tie-break
        rules; see order_was_tie_broken()).

        This assumes one index entry per NYSE trading session (as produced
        by asfreq(DateManager().cbd)) and predicts the next NYSE trading day
        (see next_trading_day()).

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
        tuple[pd.Timestamp, float, tuple[int, int]]
            The date being forecast (the next NYSE trading day after the
            last index entry), the forecast price in the units of
            original_ts, and the (p, q) order of the model that made the
            forecast (see select_order()). The model is ARIMA(p, 0, q) on
            log returns, equivalent to ARIMA(p, 1, q) on log prices.
        """
        ts = self.to_log_returns(original_ts)
        self.train_result = self.train_arima(
            ts, max_p=max_p, max_q=max_q, train_len=train_len, d=0
        )
        best_order = self.select_order(self.train_result)
        self.order_tie_broken = self.order_was_tie_broken(self.train_result)
        self.final_model = self.best_arima_model(ts, self.train_result)
        pred_date = self.next_trading_day(ts.index[-1])
        pred = self.predict(original_ts)
        return pred_date, pred, best_order

    def to_log_returns(self, original_ts: pd.Series) -> pd.Series:
        """
        Convert prices to daily log returns, log(P_t) - log(P_{t-1}). This
        is the only differencing step in the model: the ARIMA models are fit
        with d = 0 on these returns, which is ARIMA(p, 1, q) on log prices.

        The first value, which has no previous price, is dropped. The
        datetime index (and its frequency) is otherwise passed through, so a
        session missing from the prices still appears as NaN.

        Parameters
        ----------
        original_ts : pd.Series
            Prices, one entry per trading session.

        Returns
        -------
        pd.Series
            Log returns, one entry shorter than original_ts.
        """
        return np.log(original_ts).diff().iloc[1:]

    def train_arima(self, ts, max_p=5, d=0, max_q=5, train_len=120):
        """
        Train and score ARIMA models for every p and q under the given
        maximums (excluding (0, 0)), one after another, to find the best p
        and q values for modeling the given time series.

        Parameters
        ----------
        ts : pd.Series
            The time series to be modeled. The raw prices should be
            transformed by to_log_returns() first.

        max_p : int, optional
            The maximum number of lagged values to test. Defaults to 5.

        max_q : int, optional
            The maximum number of lagged disturbances to test. Defaults to 5.

        train_len : int, optional
            Then length of the training set. Defaults to 120.

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
        arima_fit_results = [self.p_d_q_fit(task) for task in tasks]
        arima_fit_df = pd.DataFrame(arima_fit_results)
        arima_fit_df.set_index(["p", "q"], inplace=True)
        return arima_fit_df

    def p_d_q_fit(self, task):
        """
        Train and evaluate an ARIMA model with the given p, d, and q parameters
        in the task dictionary.

        Rolling-origin evaluation: for each sub-window of train_len
        observations, fit the model and forecast the next observation one
        step ahead. If a fit fails (LinAlgError or ValueError), that step is
        counted as an error and recorded as a missing forecast; it is
        excluded from the RMSE and from the AIC/BIC means rather than scored
        with another step's model. If every fit fails, rmse, mean_aic, and
        mean_bic are infinite so the order ranks last in select_order().

        Called by train_arima once per candidate order.

        Parameters
        ----------
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
        y_true = ts.iloc[train_len:]
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
                    y_pred.append(np.nan)
                    continue
                except ValueError:
                    n_stationarity_errors += 1
                    y_pred.append(np.nan)
                    continue
                forecast = model.forecast(steps=1)
                y_pred.append(forecast.iloc[0])
                aics.append(model.aic)
                bics.append(model.bic)
        df = (
            pd.DataFrame({"y_true": y_true.to_numpy(), "y_pred": y_pred})
            .replace(np.inf, np.nan)
            .dropna()
        )
        if df.empty:
            rmse = mean_aic = mean_bic = np.inf
        else:
            rmse = np.sqrt(
                mean_squared_error(y_true=df["y_true"], y_pred=df["y_pred"])
            )
            mean_aic = np.mean(aics)
            mean_bic = np.mean(bics)
        result = {
            "p": p,
            "q": q,
            "rmse": rmse,
            "mean_aic": mean_aic,
            "mean_bic": mean_bic,
            "n_convergence_errors": n_convergence_errors,
            "n_stationarity_errors": n_stationarity_errors,
        }
        return result

    @staticmethod
    def average_rank(train_result: pd.DataFrame) -> pd.Series:
        """
        Each candidate's average of its rank by out-of-sample RMSE and its
        rank by mean BIC (rank 1 is the lowest value; equal values share the
        average of their ranks). Lower is better. Pure function.

        Parameters
        ----------
        train_result : pd.DataFrame
            Training results from train_arima(), indexed by (p, q), with at
            least "rmse" and "mean_bic" columns.

        Returns
        -------
        pd.Series
            Average rank, indexed like train_result.
        """
        return train_result[["rmse", "mean_bic"]].rank().mean(axis=1)

    def select_order(self, train_result: pd.DataFrame) -> tuple[int, int]:
        """
        Choose the (p, q) order with the best combined rank: each candidate
        is ranked by out-of-sample RMSE and by mean BIC (lower is better for
        both), the two ranks are averaged (see average_rank()), and the
        lowest average wins. Pure function.

        With few candidates, equal average ranks are common (e.g. ranks
        (1, 2) and (2, 1) both average 1.5). Ties are broken, in this order,
        by:

        1. lower out-of-sample RMSE (the forecasting objective; BIC's
           complexity penalty has already counted once in the average),
        2. fewer parameters, p + q (parsimony, e.g. when two candidates'
           fits all failed and both RMSEs are infinite),
        3. lower p (only so that the result never depends on row order).

        The result therefore does not depend on the row order of
        train_result. Ranks ignore the size of differences, so an RMSE lower
        by a tiny amount still ranks ahead; no tolerance is applied.

        Parameters
        ----------
        train_result : pd.DataFrame
            Training results from train_arima(), indexed by (p, q), with at
            least "rmse" and "mean_bic" columns.

        Returns
        -------
        tuple[int, int]
            The selected (p, q).
        """
        p = train_result.index.get_level_values("p").to_numpy()
        q = train_result.index.get_level_values("q").to_numpy()
        # Column names must differ from the index level names "p" and "q",
        # or pandas cannot tell which one a sort key refers to.
        candidates = pd.DataFrame(
            {
                "average_rank": self.average_rank(train_result),
                "rmse": train_result["rmse"],
                "n_params": p + q,
                "ar_order": p,
            },
            index=train_result.index,
        )
        best_p, best_q = candidates.sort_values(
            ["average_rank", "rmse", "n_params", "ar_order"], kind="stable"
        ).index[0]
        return int(best_p), int(best_q)

    def order_was_tie_broken(self, train_result: pd.DataFrame) -> bool:
        """
        Whether select_order() needed its tie-break rules for train_result,
        i.e. more than one candidate shares the lowest average rank. Pure
        function.

        Parameters
        ----------
        train_result : pd.DataFrame
            Training results from train_arima() (see average_rank()).

        Returns
        -------
        bool
            True if two or more candidates tie for the lowest average rank.
        """
        average_rank = self.average_rank(train_result)
        return bool((average_rank == average_rank.min()).sum() > 1)

    def best_arima_model(self, ts, train_result):
        """
        Use training results from train_arima and make a final ARIMA
        model that has the best p and q values, as determined by their
        RMSE and BIC metrics.

        Parameters
        ----------
        ts : ndarray of float64
            Log-transformed and differenced timeseries to model.

        train_result : pd.DataFrame
            DataFrame of training results of ARIMA models.

        Returns
        -------
        ARIMA models trained with the best p and q values.
        """
        best_p, best_q = self.select_order(train_result)
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
