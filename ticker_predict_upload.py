import os
import pandas as pd
from fsf_arima_models import ArimaModels
from s3_uploader import S3Uploader
from date_manager import DateManager
from ticker_download_manager import TickerDownloadManager


class TickerPredictUpload:
    def __init__(self):
        """
        Instantiate the by preparing the custom business day that skips
        holidays, logging into HuggingFace, and getting a Client for
        Polygon.io API (for ticker values).
        """
        monthly_download_folder = os.path.join("input", "monthly")
        self.dm = DateManager()
        self.tdm = TickerDownloadManager(monthly_download_folder)

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
                ticker_ts = ticker_ts.asfreq("B")  # Business day frequency
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

    def run(self):
        """
        Download or read cached ticker data from Polygon, process it, and
        upload it to DigitalOcean for transfer to the front end. Manage caches of
        stock tickers (so that the Polygon API is not accessed unnecessarily)
        and predictions (so that a long running process is not run
        unecessarily).
        """
        long_df, start_date, end_date = self.tdm.get_latest_tickers()
        print(f"{start_date} to {end_date}")
        wide_df = self.pivot_ticker_close_wide(long_df)
        all_forecasts_df_local_filename = os.path.join(
            "output", f"All Forecasts {self.dm.get_today_date()}.csv"
        )
        if not os.path.exists(all_forecasts_df_local_filename):
            arima_forecasts_df = self.train_arima_models(wide_df)
            arima_forecasts_df.to_csv(all_forecasts_df_local_filename, index=True)
        s3u = S3Uploader()
        time_series_space_name = os.getenv("TIME_SERIES_SPACE_NAME")
        s3u.upload_file(
            all_forecasts_df_local_filename, time_series_space_name, "all_forecasts.csv"
        )


if __name__ == "__main__":
    tpu = TickerPredictUpload()
    tpu.run()
