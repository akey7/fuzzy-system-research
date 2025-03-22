import os
import time
import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from fsf_arima_models import ArimaModels
from sklearn.metrics import mean_squared_error, mean_absolute_error
from huggingface_hub import login
from datasets import Dataset
from polygon import RESTClient
from dotenv import load_dotenv


class DownloadPredictUpload:
    def __init__(self):
        """
        Instantiate the by preparing the custom business day that skips
        holidays, logging into HuggingFace, and getting a Client for
        Polygon.io API (for ticker values).
        """
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token, add_to_git_credential=True)
        polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")
        self.polygon_client = RESTClient(polygon_io_api_key)
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

    def training_window_start_end(self, start_timestamp, num_days=20):
        """
        Return a list of lists, with each inner list containing two pd.Timestamps.
        The first timestamp is the start of a business day, and the second timestamp
        is the end of a business day. These ranges are used to specify intervals
        to train ARIMA models on.

        Parameters
        ----------
        start_timestamp : pd.Timestamp
            Start day of range.

        num_days : int, optional
            Number of days in the specified range. If not specified, defaults
            to 20 business days.

        Returns
        -------
        List[List[pd.Timestamp, pd.Timestamp]]
            Returns timestamp ranges.
        """
        start_timestamps = self.create_business_day_range(
            pd.Timestamp(start_timestamp), num_days
        )
        timestamp_ranges = []
        for start_timestamp in start_timestamps:
            end_timestamp = self.future_business_day(start_timestamp, num_days).replace(
                hour=23, minute=59, second=59
            )
            timestamp_ranges.append([start_timestamp, end_timestamp])
        for timestamp_range in timestamp_ranges:
            print(timestamp_range)
        return timestamp_ranges
    
    def get_tickers(self, tickers, date_from, date_to, delay=5):
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
                rows.append({
                    "timestamp": a.timestamp,
                    "ticker": ticker,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                    "vwap": a.vwap,
                    "transactions": a.transactions,
                })
            print(f"{ticker}. Acquired {len(rows)} so far. Sleeping 5 seconds...")
            time.sleep(delay)
        df = pd.DataFrame(rows)
        return df
