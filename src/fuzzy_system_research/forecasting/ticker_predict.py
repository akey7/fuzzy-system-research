import os
from typing import Protocol
import pandas as pd
from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.forecasting.arima_models import ArimaModels
from fuzzy_system_research.storage.layout import ensure_parent_dir

# Most recent trading sessions of price data the walk-forward ARIMA uses.
ARIMA_WINDOW_SESSIONS = 40


class ArimaWalkForward:
    """
    Walk-forward, one-step-ahead ARIMA forecasts of adjusted close prices
    over sliding windows of trading days. Does no downloading or uploading.
    """

    def __init__(self):
        """
        Instantiate with the DateManager used to build the training windows.
        """
        self.dm = DateManager()

    def training_window_start_end(self, start_timestamp, end_timestamp, num_days=20):
        """
        Return a list of lists, with each inner list containing two pd.Timestamps.
        The first timestamp is the start of a business day, and the second timestamp
        is the end of a business day. These ranges are used to specify intervals
        to train ARIMA and Holt-Winters models on in a walk-forward method.

        Parameters
        ----------
        start_timestamp : pd.Timestamp
            Start day of range.

        end_timestamp : pd.Timestamp
            End of the day range.

        num_days : int, optional
            Number of days in the specified ranges. If not specified, defaults
            to 20 business days.

        Returns
        -------
        List[List[pd.Timestamp, pd.Timestamp]]
            Returns timestamp ranges.
        """
        timestamp_ranges = [
            [start_timestamp, self.dm.future_business_day(start_timestamp, 20)]
        ]
        while timestamp_ranges[-1][1] < pd.Timestamp(end_timestamp.date()):
            next_start_timestamp = self.dm.future_business_day(timestamp_ranges[-1][0], 1)
            next_end_timestamp = self.dm.future_business_day(
                next_start_timestamp, num_days
            )
            timestamp_ranges.append([next_start_timestamp, next_end_timestamp])
        # for timestamp_range in timestamp_ranges:
        #     print(timestamp_range)
        return timestamp_ranges

    def train_arima_models(
        self,
        df,
        n_business_days=20,
        max_p=2,
        max_q=2,
        n_workers=6,
    ):
        """
        Trains ARIMA models along each training window implementing a
        walk-forward backtesting scheme of model evaluation. Each training
        seeks the optimal (p, q) values for the best performance.

        This function passes prices, not returns: the single differencing
        step (prices to daily log returns) happens inside ArimaModels.fit(),
        via ArimaModels.to_log_returns().

        Each window is resampled to the NYSE trading calendar, so one model
        time step is one trading session: holidays are skipped rather than
        treated as missing observations.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame of adjusted close data.

        n_business_days : int, optional
            Number of days in each training window. Default is 20, which is a
            month of business days.

        max_p : int, optional
            Max p of ARIMA models. Defaults to 2.

        max_q : int, optional
            Max q of ARIMA models. Defaults to 2.

        n_workers : int, optional
            Number of cores to dedicate to training models. Defaults to 6

        Returns
        -------
        pd.DataFrame
            The forecasts after each model training, indexed by pred_date.
            For every ticker T there are four columns, in this order:
            T_arima (forecast price), T_arima_p and T_arima_q (integer order
            of the ARIMA(p, 0, q) model on log returns that made the row's
            forecast), and T (the actual price, missing on the final row).
        """
        all_forecast_dfs = []
        timestamp_ranges = self.training_window_start_end(
            df.index[0],
            df.index[-1],
            n_business_days,
        )
        for ticker in df.columns:
            forecast_rows = []
            for start_timestamp, end_timestamp in timestamp_ranges:
                am = ArimaModels(n_workers=n_workers)
                ticker_ts = df[ticker]
                ticker_ts = ticker_ts.loc[start_timestamp:end_timestamp]
                # One row per NYSE trading session. NYSE holidays get no row
                # (pandas "B" would insert them as missing values), while a
                # session missing from the data still appears as NaN.
                ticker_ts = ticker_ts.asfreq(self.dm.cbd)
                pred_date, pred, (p, q) = am.fit(
                    ticker_ts, max_p=max_p, max_q=max_q, train_len=10
                )
                pred_key = f"{ticker}_arima"
                pred_dict = {
                    "pred_date": pred_date,
                    pred_key: pred,
                    f"{pred_key}_p": p,
                    f"{pred_key}_q": q,
                }
                print(pred_dict)
                forecast_rows.append(pred_dict)
            forecast_df = (
                pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            )
            forecast_start_timestamp = forecast_df.index[0]
            forecast_end_timestamp = forecast_df.index[-1]
            forecast_df[ticker] = df.loc[
                forecast_start_timestamp:forecast_end_timestamp, ticker
            ].copy()
            all_forecast_dfs.append(forecast_df)
        all_forecast_df = pd.concat(all_forecast_dfs, axis=1).sort_index()
        return all_forecast_df


class ForecastTrainer(Protocol):
    """
    Anything that can produce the forecasts DataFrame, e.g. ArimaWalkForward
    (or a test double).
    """

    def train_arima_models(self, df: pd.DataFrame) -> pd.DataFrame: ...


def load_or_train_forecasts(
    window_df: pd.DataFrame,
    path: str,
    retrain: bool = False,
    trainer: ForecastTrainer | None = None,
) -> bool:
    """
    Make sure the forecasts CSV for a run exists at path, training ARIMA
    models only when needed. The CSV at path is the cache: if it exists and
    retrain is False, nothing is trained and the file is left as is.
    Otherwise the models are trained on window_df and the CSV is (over)written
    in the format of all_forecasts.csv in INTERFACE_CONTRACT.md.

    Parameters
    ----------
    window_df : pd.DataFrame
        Wide adjusted close prices to train on, normally the last
        ARIMA_WINDOW_SESSIONS trading sessions.

    path : str
        Local path of the forecasts CSV. Missing folders are created.

    retrain : bool, optional
        If True, train and overwrite the CSV even if it exists. Defaults to
        False.

    trainer : ForecastTrainer, optional
        Object that trains the models. Defaults to a new ArimaWalkForward.

    Returns
    -------
    bool
        True if models were trained and the CSV written, False if the cached
        CSV was reused.
    """
    if os.path.exists(path) and not retrain:
        print(f"Using cached forecasts {path}")
        return False
    trainer = trainer if trainer is not None else ArimaWalkForward()
    forecasts_df = trainer.train_arima_models(window_df)
    forecasts_df.to_csv(ensure_parent_dir(path), index=True)
    print(f"Saved {path}")
    return True
