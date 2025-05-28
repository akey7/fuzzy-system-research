import os
import glob
import time
import pandas as pd
from dotenv import load_dotenv
from polygon import RESTClient
from date_manager import DateManager


class TickerDownloadManager:
    def __init__(self, download_folder_name):
        load_dotenv()
        polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")
        self.polygon_client = RESTClient(polygon_io_api_key)
        self.dm = DateManager()
        self.download_folder_name = download_folder_name
        self.tickers = ["I:SPX", "QQQ", "VXUS", "GLD"]

    def download_tickers(self, tickers, date_from, date_to, delay=10):
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
            If not specifed defaults to 10 seconds.

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
            print(f"{ticker}. Acquired {len(rows)} so far. Sleeping {delay} seconds...")
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

    def get_files_by_extension_sorted(self, directory, extension, include_path=True):
        """
        Returns a list of filenames with a given extension sorted by descending creation date.

        Parameters
        ----------
        directory : str
            Directory path to search for files

        extension : str
            File extension to filter by (e.g., '.csv', '.txt', 'py')

        include_path : bool, optional
            If True, returns full file paths; if False, returns only filenames, default False

        Returns
        -------
        List[str]
            List of filenames (or full paths) sorted by creation date (newest first)

        Raises
        ------
        FileNotFoundError
            If the specified directory doesn't exist
        ValueError
            If extension is empty or None
        """
        if not extension:
            raise ValueError("Extension cannot be empty or None")
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory '{directory}' does not exist")
        if not extension.startswith("."):
            extension = f".{extension}"
        pattern = os.path.join(directory, f"*{extension}")
        files = glob.glob(pattern)
        files = [f for f in files if os.path.isfile(f)]
        files.sort(key=lambda x: os.path.getctime(x), reverse=True)
        if include_path:
            return files
        else:
            return [os.path.basename(f) for f in files]

    def get_latest_month_of_tickers(self, use_cache=True):
        """
        Retrieve the latest month of tickers.

        Parameters
        ----------
        use_cache : bool
            If True, does not call API. Rather, it uses the most recent download

        Returns
        -------
        Tuple[pd.DataFrame, str, str]
            Returns the latest dataframe, start date, and end date.
        """
        if use_cache:
            latest_filename = self.get_files_by_extension_sorted(
                self.download_folder_name, "csv"
            )[0]
            df = pd.read_csv(latest_filename)
            start_date = df["datetime"].min()
            end_date = df["datetime"].max()
            return df, start_date, end_date
        else:
            date_from = self.dm.past_business_day(
                pd.Timestamp(self.dm.get_today_date()), 40
            )
            date_to = self.dm.past_business_day(
                pd.Timestamp(self.dm.get_today_date()), 1
            ).replace(hour=23, minute=59, second=59)
            long_df_filename = os.path.join(
                self.download_folder_name, f"Tickers {self.dm.get_today_date()}.csv"
            )
            long_df = self.download_tickers(
                self.tickers, date_from=date_from, date_to=date_to
            )
            long_df.to_csv(long_df_filename, index=True)
            return long_df, date_from, date_to


if __name__ == "__main__":
    dm = TickerDownloadManager(os.path.join("input", "monthly"))
    df, start_date, end_date = dm.get_latest_month_of_tickers(use_cache=False)
    print(f"{start_date} to {end_date}")
    print(df.head())
