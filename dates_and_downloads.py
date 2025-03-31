import warnings
import os
import time
from datetime import datetime
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from huggingface_hub import login
from datasets import Dataset
from polygon import RESTClient
from dotenv import load_dotenv
from fsf_arima_models import ArimaModels


class DatesAndDownloads
