import os
import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from fsf_arima_models import ArimaModels
from sklearn.metrics import mean_squared_error, mean_absolute_error
from huggingface_hub import login
from datasets import Dataset
from dotenv import load_dotenv


class DownloadPredictUpload:
    def __init__(self):
        """
        Instantiate the by preparing the custom business day that skips
        holidays and logging into HuggingFace
        """
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        login(hf_token, add_to_git_credential=True)
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays()
        self.cbd = CustomBusinessDay(holidays=holidays)
