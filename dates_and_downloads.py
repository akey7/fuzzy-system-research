import os
import time
import json
import datetime
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from polygon import RESTClient
from dotenv import load_dotenv
import pandas_datareader as web
from dateutil.relativedelta import relativedelta


class DatesAndDownloads:
    def __init__(self):
        load_dotenv()
        polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")
        self.polygon_client = RESTClient(polygon_io_api_key)
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays()
        self.cbd = CustomBusinessDay(holidays=holidays)

    def get_today_date(self):
        """
        Return today's date as a YYYY-MM-DD string.
        """
        today = datetime.datetime.now()
        return today.strftime("%Y-%m-%d")
    
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
        timestamp_ranges = [[start_timestamp, self.future_business_day(start_timestamp, 20)]]
        while timestamp_ranges[-1][1] < pd.Timestamp(end_timestamp.date()):
            next_start_timestamp = self.future_business_day(timestamp_ranges[-1][0], 1)
            next_end_timestamp = self.future_business_day(next_start_timestamp, num_days)
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
        df.set_index("datetime", inplace=True)
        df.drop("timestamp", axis=1, inplace=True)
        df.sort_index(inplace=True)
        return df
    
    def get_daily_risk_free_rate(self):
        today_date = self.get_today_date()
        risk_free_rate_filename = os.path.join("input", f"Risk Free Rate {today_date}.json")
        if os.path.exists(risk_free_rate_filename):
            print("Reading risk-free rate cache...")
            with open(risk_free_rate_filename, "r", encoding="utf-8") as f:
                risk_free_rate_data = json.load(f)
                print(risk_free_rate_data)
                daily_risk_free_rate = risk_free_rate_data["daily_risk_free_rate"]
        else:
            end_date = datetime.datetime.now()
            start_date = end_date - relativedelta(years=1)
            print(start_date, end_date)
            tb3m_df = web.DataReader("DTB3", "fred", start_date, end_date).sort_values(
                "DATE", ascending=False
            )
            risk_free_rate = float(tb3m_df.iloc[0]["DTB3"])
            daily_risk_free_rate = risk_free_rate / 252
            risk_free_rate_date = str(tb3m_df.index[0])
            print(daily_risk_free_rate)
            risk_free_rate_data = {
                "risk_free_rate": risk_free_rate,
                "daily_risk_free_rate": daily_risk_free_rate,
                "risk_free_rate_date": risk_free_rate_date,
            }
            with open(risk_free_rate_filename, "w", encoding="utf-8") as f:
                json.dump(risk_free_rate_data, f, indent=4)
        return daily_risk_free_rate

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

