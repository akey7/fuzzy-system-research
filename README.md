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

Here are the API keys it needs:

1. `MARKETSTACK_API_KEY`: Your key to the MarketStack API. It needs to be on a price tier that gives 10 years of history and end-of-day prices.

## Purposes of folders and scripts

### Folders

The folloing folders should be created before running the scripts in the section below. Their content is not committed to GitHub

1. `input/`: For input data (see below).

2. `output/`: For output data (see below).

### Scripts

There are several `.py` scripts in this repo. Here is a lists and what they do:

1. `scrape_training_data.py`: Scrapes raw json from the MarketStack API with key above and places files into the `input/` folder with the following naming `[symbol]_history.json`.

2. `prep_training_data.py`: Takes the json files in `input/` and write csv files in `output/` that are appropriate for training the model. See the docstrings in the file for more information on what is in these csv files.
