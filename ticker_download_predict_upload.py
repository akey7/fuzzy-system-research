import warnings
import os
import time
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from polygon import RESTClient
from dotenv import load_dotenv
from fsf_arima_models import ArimaModels
from s3_uploader import S3Uploader


class DownloadPredictUpload:
    def __init__(self):
        """
        Instantiate the by preparing the custom business day that skips
        holidays, logging into HuggingFace, and getting a Client for
        Polygon.io API (for ticker values).
        """
        load_dotenv()
        polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")
        self.polygon_client = RESTClient(polygon_io_api_key)
        self.hf_dataset = os.getenv("HF_DATASET")
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays()
        self.cbd = CustomBusinessDay(holidays=holidays)

    def past_business_day(self, reference_date, business_days_past):
        """
        Calculates the business date a specified number of business days in the past,
        skipping US federal holidays.

        Parameters
        ----------
        reference_date : pd.Timestamp
            Reference date to calculate days from.

        business_days_past : int
            The number of business days in the past.

        Returns
        -------
        pd.Timestamp
            The calculated past business date.
        """
        return reference_date - (self.cbd * business_days_past)

    def future_business_day(self, reference_date, business_days_ahead):
        """
        Calculates the business date a specified number of business days in the future,
        skipping US federal holidays.

        Parameters
        ----------
        reference_date : pd.Timestamp
            Reference date to calculate days from.

        business_days_ahead : int
            The number of business days in the future.

        Returns
        -------
        pd.Timestamp
            The calculated future business date.
        """
        return reference_date + (self.cbd * business_days_ahead)

    def create_business_day_range(self, reference_date, num_days):
        """
        Creates a range of business days starting from a given date,
        skipping US federal holidays.

        Parameters
        ----------
        start_date : pd.Timestamp
            The starting date

        num_days : int
            The number of business days to generate.

        Returns
        -------
        pd.DatetimeIndex
            A DatetimeIndex containing the range of business days.
        """
        return pd.date_range(start=reference_date, periods=num_days, freq=self.cbd)

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
            [start_timestamp, self.future_business_day(start_timestamp, 20)]
        ]
        while timestamp_ranges[-1][1] < pd.Timestamp(end_timestamp.date()):
            next_start_timestamp = self.future_business_day(timestamp_ranges[-1][0], 1)
            next_end_timestamp = self.future_business_day(
                next_start_timestamp, num_days
            )
            timestamp_ranges.append([next_start_timestamp, next_end_timestamp])
        # for timestamp_range in timestamp_ranges:
        #     print(timestamp_range)
        return timestamp_ranges

    def get_tickers(self, tickers, date_from, date_to, delay=10):
        """
        Gets ticker data from Polygon.io between the given dates, inclusive
        of the ending date.

        Parameters
        ----------
        tickers : List[str]
            List of valid ticker symbols (like AAPL)

        date_from : str
            String representation of the start date in YYYY-MM-DD format.

        date_to : str
            String representation of the end date in YYYY-MM-DD format.

        delay: int, optional
            Seconds to wait between successive API calls for the tickers.
            If not specifed defaults to 5 seconds.

        Returns
        -------
        pd.DataFrame
            DataFrame of ticker data in long format.
        """
        rows = []
        for ticker in tickers:
            for a in self.polygon_client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=date_from,
                to=date_to,
                adjusted="true",
            ):
                rows.append(
                    {
                        "timestamp": a.timestamp,
                        "ticker": ticker,
                        "open": a.open,
                        "high": a.high,
                        "low": a.low,
                        "close": a.close,
                        "volume": a.volume,
                        "vwap": a.vwap,
                        "transactions": a.transactions,
                    }
                )
            print(f"{ticker}. Acquired {len(rows)} so far. Sleeping 5 seconds...")
            time.sleep(delay)
        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize(None)
        df["datetime"] = df["datetime"].apply(
            lambda x: pd.Timestamp(x).replace(hour=23, minute=59, second=59)
        )
        df.set_index("datetime", inplace=True)
        df.drop("timestamp", axis=1, inplace=True)
        df.sort_index(inplace=True)
        return df

    def pivot_ticker_close_wide(self, long_df):
        """
        Pivots the adjusted close values retrieved by get_tickers()
        to a wide format suitable for modeling.

        Parameters
        ----------
        long_df : pd.DataFrame
            Long DataFrame of ticker data.

        Returns
        -------
        pd.DataFrame
            Wide dataframe with adjusted close prices of tickers in the columns
            and a pd.DatetimeIndex suitable for indexing into ranges of dates.
        """
        wide_df = long_df.reset_index().pivot(
            index="datetime", columns="ticker", values="close"
        )
        wide_df.index = pd.DatetimeIndex(wide_df.index)
        wide_df.index = pd.DatetimeIndex(
            [dt.replace(hour=17, minute=0, second=0) for dt in wide_df.index]
        )
        wide_df.sort_index(inplace=True)
        return wide_df

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

        Though not in this function, the time series are log transformed and
        differenced with a lag of 1 prior to training.

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
            The forecasts after each model training.
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
                pred_date, pred = am.fit(
                    ticker_ts, max_p=max_p, max_q=max_q, train_len=10
                )
                pred_key = f"{ticker}_arima"
                pred_dict = {"pred_date": pred_date, pred_key: pred}
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

    def train_holt_winters_models(self, df, n_business_days=20, retain_actuals=True):
        """
        Train Holt-Winters models in a walk-forward method and track the
        predictions along the way.

        Parameters
        ----------
        df : pd.DataFrame
            The wide-format dataframe that contains ticker adjusted close
            prices.

        n_business_days : int, optional
            The length of the training windows in number of business days,
            excluding holidays. Defaults to 20.

        retain_actuals : bool, optional
            If True (the default) returns the actual value columns alongside
            the predictions. If False, simply returns the predictions only.

        Returns
        -------
        pd.DataFrame
            A DataFrame of predicted values along the walk-forward pattern.
        """
        all_forecast_dfs = []
        timestamp_ranges = self.training_window_start_end(
            df.index[0],
            df.index[-1],
            n_business_days,
        )
        tickers = [x for x in df.columns if "_" not in x]
        for ticker in tickers:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                forecast_rows = []
                for start_timestamp, end_timestamp in timestamp_ranges:
                    train = df[ticker]
                    train = train.loc[start_timestamp:end_timestamp]
                    train = train.resample("D").ffill().dropna()
                    model = ExponentialSmoothing(
                        train, use_boxcox=0, trend="add", seasonal="add"
                    )
                    fit = model.fit()
                    pred = float(fit.forecast(steps=1))
                    pred_key = f"{ticker}_hw"
                    pred_date = self.future_business_day(train.index[-1], 1).replace(
                        hour=17, minute=0, second=0
                    )
                    pred_dict = {"pred_date": pred_date, pred_key: pred}
                    forecast_rows.append(pred_dict)
            forecast_df = (
                pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            )
            if retain_actuals:
                forecast_start_timestamp = forecast_df.index[0]
                forecast_end_timestamp = forecast_df.index[-1]
                forecast_df[ticker] = df.loc[
                    forecast_start_timestamp:forecast_end_timestamp, ticker
                ].copy()
            all_forecast_dfs.append(forecast_df)
        all_forecast_df = pd.concat(all_forecast_dfs, axis=1)
        return all_forecast_df

    def get_today_date(self):
        """
        Return today's date as a YYYY-MM-DD string.
        """
        today = datetime.now()
        return today.strftime("%Y-%m-%d")

    def run(self):
        """
        Download ticker data from Polygon (if needed), process it, and
        upload it to HuggingFace for the front end. Manage caches of
        stock tickers (so that the Polygon API is not accessed unnecessarily)
        and predictions (so that a long running process is not run
        unecessarily).
        """
        tickers = ["I:SPX", "QQQ", "VXUS", "GLD"]
        long_df_filename = os.path.join("input", f"Tickers {self.get_today_date()}.csv")
        date_from = self.past_business_day(pd.Timestamp(self.get_today_date()), 40)
        date_to = self.past_business_day(
            pd.Timestamp(self.get_today_date()), 1
        ).replace(hour=23, minute=59, second=59)
        print(date_from, date_to)
        if os.path.exists(long_df_filename):
            long_df = pd.read_csv(long_df_filename)
        else:
            long_df = self.get_tickers(tickers, date_from=date_from, date_to=date_to)
            long_df.to_csv(long_df_filename, index=True)
        wide_df = self.pivot_ticker_close_wide(long_df)
        all_forecasts_df_local_filename = os.path.join(
            "output", f"All Forecasts {self.get_today_date()}.csv"
        )
        if os.path.exists(all_forecasts_df_local_filename):
            all_forecasts_df = pd.read_csv(all_forecasts_df_local_filename)
        else:
            arima_forecasts_df = self.train_arima_models(wide_df)
            holt_winters_forecasts_df = self.train_holt_winters_models(
                wide_df, retain_actuals=False
            )
            all_forecasts_df = pd.concat(
                [arima_forecasts_df, holt_winters_forecasts_df], axis=1
            )
            all_forecasts_df.to_csv(all_forecasts_df_local_filename, index=True)
        s3u = S3Uploader()
        time_series_space_name = os.getenv("TIME_SERIES_SPACE_NAME")
        s3u.upload_file(
            all_forecasts_df_local_filename, time_series_space_name, "all_forecasts.csv"
        )


if __name__ == "__main__":
    dpu = DownloadPredictUpload()
    dpu.run()
