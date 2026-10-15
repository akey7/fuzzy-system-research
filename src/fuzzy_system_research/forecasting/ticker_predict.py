import os
from typing import Protocol
import numpy as np
import pandas as pd
from fuzzy_system_research.data.date_manager import DateManager
from fuzzy_system_research.forecasting.arima_models import (
    PRICE_DIFFERENCING_ORDER,
    ArimaModels,
)
from fuzzy_system_research.storage.layout import ensure_parent_dir

# Returns in each walk-forward training window (about one month).
TRAINING_WINDOW_RETURNS = 20
# Backtest days scored in all_forecasts_metadata.yml.
EVALUATION_DAYS = 20
# Most recent trading sessions of price data the walk-forward ARIMA uses: one
# training window's prices (TRAINING_WINDOW_RETURNS + 1) plus one session per
# scored day, giving EVALUATION_DAYS forecasts with actuals plus the live
# forecast. 20 + 20 + 1 = 41.
ARIMA_WINDOW_SESSIONS = TRAINING_WINDOW_RETURNS + EVALUATION_DAYS + 1
# ARIMA order search: p in range(ARIMA_MAX_P), q in range(ARIMA_MAX_Q),
# excluding (0, 0).
ARIMA_MAX_P = 2
ARIMA_MAX_Q = 2
# Returns in each rolling sub-window used to score candidate orders.
ORDER_SELECTION_SUBWINDOW_RETURNS = 10


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

    def training_window_start_end(
        self, index: pd.DatetimeIndex, num_days: int = 20
    ) -> list[list[pd.Timestamp]]:
        """
        Return the walk-forward training windows as [start, end] timestamp
        pairs. Each window spans num_days + 1 consecutive dates of index (so
        num_days returns), and consecutive windows slide by one date.

        Windows are built from the dates actually present in the data, not
        from the calendar, so every window ends on an observed date and no two
        windows end on the same one. With complete data (every NYSE session
        present) this is one window per session from the (num_days + 1)-th
        onward. If a whole session is missing from the data, no window ends on
        it, so there is no one-step-ahead forecast for the session after it
        (that would need the missing close).

        Parameters
        ----------
        index : pd.DatetimeIndex
            Sorted dates of the price data, one per observed trading session.

        num_days : int, optional
            Number of returns in each window. Defaults to 20, about a month
            of trading days.

        Returns
        -------
        List[List[pd.Timestamp, pd.Timestamp]]
            Returns timestamp ranges, in date order.

        Raises
        ------
        ValueError
            If index has num_days dates or fewer.
        """
        if len(index) <= num_days:
            raise ValueError(
                f"Need more than {num_days} dates for a {num_days}-day window, "
                f"got {len(index)}."
            )
        return [[index[end - num_days], index[end]] for end in range(num_days, len(index))]

    def train_arima_models(
        self,
        df,
        n_business_days=TRAINING_WINDOW_RETURNS,
        max_p=ARIMA_MAX_P,
        max_q=ARIMA_MAX_Q,
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

        Missing data (see training_window_start_end()): windows end only on
        dates present in df, so pred_date values are unique. If one ticker's
        close is missing on a window's last date, that ticker's T_arima and
        order columns are left empty on that row (no two-step-ahead forecast
        is substituted), while T_naive still uses the most recent observed
        close.

        Parameters
        ----------
        df : pd.DataFrame
            Wide format DataFrame of adjusted close data.

        n_business_days : int, optional
            Number of returns in each training window. Defaults to
            TRAINING_WINDOW_RETURNS (20), which is about a
            month of business days.

        max_p : int, optional
            Exclusive upper bound on p. Defaults to ARIMA_MAX_P (2), so p is 0 or 1.

        max_q : int, optional
            Exclusive upper bound on q. Defaults to ARIMA_MAX_Q (2), so q is 0 or 1.

        n_workers : int, optional
            Number of cores to dedicate to training models. Defaults to 6

        Returns
        -------
        pd.DataFrame
            The forecasts after each model training, indexed by pred_date.
            For every ticker T there are six columns, in this order:
            T_arima (forecast price); T_arima_p, T_arima_d, and T_arima_q
            (nullable integer order of the model that made the row's
            forecast, stated on log prices: ARIMA(p, 0, q) on log returns is
            ARIMA(p, 1, q) on log prices, so T_arima_d is always
            PRICE_DIFFERENCING_ORDER = 1; all three are empty where T_arima
            is);
            T_naive (naive baseline forecast: the most recent observed close
            before pred_date, i.e. the previous trading day's close when no
            data is missing); and T (the actual price, missing on the final
            row).
        """
        all_forecast_dfs = []
        timestamp_ranges = self.training_window_start_end(df.index, n_business_days)
        for ticker in df.columns:
            forecast_rows = []
            pred_key = f"{ticker}_arima"
            order_keys = [f"{pred_key}_p", f"{pred_key}_d", f"{pred_key}_q"]
            for start_timestamp, end_timestamp in timestamp_ranges:
                ticker_ts = df[ticker]
                ticker_ts = ticker_ts.loc[start_timestamp:end_timestamp]
                # One row per NYSE trading session. NYSE holidays get no row
                # (pandas "B" would insert them as missing values), while a
                # session missing from the data still appears as NaN.
                ticker_ts = ticker_ts.asfreq(self.dm.cbd)
                if pd.isna(ticker_ts.iloc[-1]):
                    # This ticker's close at the forecast origin is missing, so
                    # there is no one-step-ahead ARIMA forecast: leave the
                    # forecast and its order empty rather than forecast two
                    # steps ahead from an older close.
                    pred_date = self.dm.future_business_day(ticker_ts.index[-1], 1)
                    pred, p, d, q = np.nan, pd.NA, pd.NA, pd.NA
                else:
                    am = ArimaModels(n_workers=n_workers)
                    pred_date, pred, (p, q) = am.fit(
                        ticker_ts,
                        max_p=max_p,
                        max_q=max_q,
                        train_len=ORDER_SELECTION_SUBWINDOW_RETURNS,
                    )
                    d = PRICE_DIFFERENCING_ORDER
                # Naive (random walk) baseline: the most recent observed close
                # in the same window the ARIMA model saw, so both forecasts
                # use exactly the same information.
                naive = float(ticker_ts.dropna().iloc[-1])
                pred_dict = {
                    "pred_date": pred_date,
                    pred_key: pred,
                    order_keys[0]: p,
                    order_keys[1]: d,
                    order_keys[2]: q,
                    f"{ticker}_naive": naive,
                }
                print(pred_dict)
                forecast_rows.append(pred_dict)
            forecast_df = (
                pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            )
            # Nullable integers: whole numbers, empty where there is no forecast.
            forecast_df[order_keys] = forecast_df[order_keys].astype("Int64")
            forecast_df[pred_key] = forecast_df[pred_key].astype(float)
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
