# fuzzy-system-research
EDA and other analysis code to accompany the fuzzy-system-finance repo and app. Being split away because otherwise it would complicate deployment.

## Installation

### Development Dependencies

This project can be installed into a conda environment. First create the conda environment with:

```
conda create -n fuzzy-system-research python=3.12
```

Then install all development and production dependencies:

```
pip install -r requirements-dev.txt
```

To install only production dependencies:

```
pip install -r requirements.txt
```

### Production Dependencies

To install only the production dependencies, activate the virtual environment and run the following:

```
pip install -r requirements.txt
```

### `.env` file

**Never commit the `.env` file to GitHub!** It should never be made publicly accessible because it contains API keys.

Here are the keys the scripts need to function:

1. `POLYGON_IO_API_KEY`: Key to Polygon.io API for stock and index data. Obtain from API provider.

2. `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_RWDELETE`, `FSF_FRONT_END_BUCKET_KEY_ID`, `FSF_FRONT_END_BUCKET_ENDPOINT`: Region, read/write/delete key, key id, and endpoint of the DigitalOcean S3/Spaces bucket. Set as appropriate for development or production environments. Obtain values from DigitalOcean or AWS environments.

3. `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME`: Space names for portfolio and ARIMA time series data.

Items 2 and 3 connect the backend to the front end.

## Purposes of folders and scripts

### Folders

The folloing folders should be created before running the scripts in the section below. Their content is not committed to GitHub.

1. `input/`: Holds risk free rate data.

2. `input/annual/`: Holds annual ticker data for use in portfolio optimization and ARIMA.

3. `input/annual_predictors/`: Index data used as predictors in `returns_regression.ipynb`.

4. `input/annual_targets/`: Stock data used as targets in `returns_regression.ipynb`.

5. `output/`: Holds output data.

### Scripts and notebooks

Here is a list of scripts and notebooks with information on if they are to be executed directly, whether they upload data to the front end, and their purpose.

| Filename                     | Standalone Execution? | Purpose                                                                                                                                                                                                                   | Uploads to front end?                |
| ---------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `portfolio_optimization.py`  | Yes                   | Obtain adjusted close price data for a portfolio of stocks and compute max Sharpe ratio portfolio weights, min variance portfolio weights, efficient frontier, and Monte Carlo simulation of portfolio risks and returns. | Yes                                  |
| `ticker_predict_upload.py`   | Yes                   | Obtain 2 months of data and fit a month's worth of ARIMA models on the adjusted close prices.                                                                                                                             | Yes                                  |
| `ticker_download_to_csv.py`  | Yes                   | Download a supported symbol from Polygon.io and put the results into a `.csv` file.                                                                                                                                       | No                                   |
| `ticker_download_manager.py` | No                    | Support other scripts by downloading new or retrieving old data from the cache.                                                                                                                                           | No                                   |
| `s3_uploader.py`             | No                    | Support other scripts with uploads to S3/Spaces buckets on AWS/DigitalOcean.                                                                                                                                              | Yes (other scripts use it to upload) |
| `fsf_arima_models.py`        | No                    | Supports other scripts by training ARIMS models on price data.                                                                                                                                                            | No                                   |
| `date_manager.py`            | No                    | Supports other scripts by managing date and business day calculation operations.                                                                                                                                          | No                                   |
| `returns_regression.py`      | Yes                   | Experimental. Use market index data as predictors of the prices of four stocks. Does not upload data to front end.                                                                                                        | No                                   |

