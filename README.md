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

### Production Dependencies

To install only the production dependencies, activate the virtual environment and run the following:

```
pip install -r requirements.txt
```

### `.env` file

**Never commit the `.env` file to GitHub!** It should never be made publicly accessible because it contains API keys.

Here are the keys it needs:

1. `POLYGON_IO_API_KEY`: Key to Polygon.io API where stock data comes from.

2. `HF_TOKEN`: Token to HuggingFace to upload processed datasets for the front end.

## Purposes of folders and scripts

### Folders

The folloing folders should be created before running the scripts in the section below. Their content is not committed to GitHub

1. `input/`: For input data.

2. `output/`: For output data.

### Scripts and notebooks

There are several `.py` scripts in this repo. Here is a lists and what they do:

1. `fsf_arima_models.py`: Handles ARIMA model training and parallelization.

2. `fsf_arima_models_viz.py`: Visualized time series analyzed by ARIMA models.

3. `portfolio_decomposition_arima.ipynb` Decomposes and visualizes residuals of the tickers in the portfolio.
