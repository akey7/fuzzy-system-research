import warnings
import os
from datetime import datetime
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from huggingface_hub import login
from datasets import Dataset
from dotenv import load_dotenv
from fsf_arima_models import ArimaModels
from dates_and_downloads import DatesAndDownloads


class DownloadPredictUpload(DatesAndDownloads):
    def __init__(self):
        """
        Instantiate the by preparing the custom business day that skips
        holidays, logging into HuggingFace, and getting a Client for
        Polygon.io API (for ticker values).
        """
        super().__init__()
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token, add_to_git_credential=True)
        self.hf_dataset = os.getenv("HF_DATASET")

    def train_arima_models(
        self,
        df,
        n_business_days=20,
        max_p=2,
        max_q=2,
        n_workers=6,
    ):
        """
        Trains ARMA models along each training window implementing a
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
            Max p of ARMA models. Defaults to 2.

        max_q : int, optional
            Max q of ARMA models. Defaults to 2.

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
                    model = ExponentialSmoothing(train, use_boxcox=0)
                    fit = model.fit()
                    pred = float(fit.forecast(steps=1))
                    pred_key = f"{ticker}_hw"
                    pred_date = self.future_business_day(train.index[-1], 1)
                    pred_dict = {"pred_date": pred_date, pred_key: pred}
                    forecast_rows.append(pred_dict)
            forecast_df = pd.DataFrame(forecast_rows).set_index("pred_date").sort_index()
            if retain_actuals:
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
        Download ticker data from Polygon (if needed), process it, and
        upload it to HuggingFace for the front end. Manage caches of
        stock tickers (so that the Polygon API is not accessed unnecessarily)
        and predictions (so that a long running process is not run
        unecessarily).
        """
        tickers = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA", "TSLA"]
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
        all_forecasts_df_filename = os.path.join(
            "output", f"All Forecasts {self.get_today_date()}.csv"
        )
        if os.path.exists(all_forecasts_df_filename):
            all_forecasts_df = pd.read_csv(all_forecasts_df_filename)
        else:
            arima_forecasts_df = self.train_arima_models(wide_df)
            holt_winters_forecasts_df = self.train_holt_winters_models(wide_df, retain_actuals=False)
            all_forecasts_df = pd.concat([arima_forecasts_df, holt_winters_forecasts_df], axis=1)
            all_forecasts_df.to_csv(all_forecasts_df_filename, index=True)
        ds = Dataset.from_pandas(all_forecasts_df)
        ds.push_to_hub(self.hf_dataset)


if __name__ == "__main__":
    dpu = DownloadPredictUpload()
    dpu.run()
