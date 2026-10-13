# Interface Contract: fuzzy-system-research (backend) ↔ fuzzy-system-finance (frontend)

| | |
| --- | --- |
| **Contract version** | 13 |
| **Last updated** | 2026-10-12 |
| **Source of truth** | `fuzzy-system-research` (backend). The copy in `fuzzy-system-finance` is a mirror. |

This file is stored **byte-for-byte identically** in both repos. It describes the files the backend uploads to one S3-compatible bucket (a DigitalOcean Space, accessed with `boto3`) and the frontend reads. Treat it as a public API.

## Rules for editing this file (humans and AI agents)

1. **Edit the backend copy first**, then copy it unchanged into the frontend repo. Never edit only one copy, and never let the copies diverge.
2. **Bump the contract version and add a changelog entry** for every change, including clarifications that change meaning.
3. **Breaking changes** (anything in the list below) need explicit human approval. Describe the frontend impact before making the change, and prefer an additive alternative with a migration path.
   - Renaming or removing an object key, top-level folder, file, YAML key, HDF5 group/dataset, or CSV column
   - Changing a data type, units, scaling, or date format
   - Changing the shape of an array or the meaning of a value
   - Changing the set of tickers or the covered date range in a way a reader could not detect from the data itself
4. **Additive changes** (new optional keys, new columns, new files) are allowed without breaking the contract, **because readers must ignore anything they do not recognize** (see "Reader requirements").
5. The backend agent must not edit the frontend repo, and the frontend agent must not edit the backend repo. Each lists what the other must change.

## Status of this document

Version 1 was written from the backend code. Version 2 was checked against real output files from the backend (a real `optimization_metadata.yml`, the `h5dump -H` structure of the `.h5` file, and the header and final row of `all_forecasts.csv`). What was and was not verified is listed under "Verification status" at the end.

## Where the data lives

All data lives in **one DigitalOcean Space**, which is one S3 bucket. Inside it are **two top-level folders** (key prefixes), whose names are the values of two environment variables. Both the backend and the frontend use these variable names:

| Environment variable | Meaning |
| --- | --- |
| `FSF_FRONT_END_BUCKET_NAME` | Name of the Space (the S3 bucket) |
| `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER` | Name of the top-level folder holding the optimization files |
| `FSF_FRONT_END_TIME_SERIES_FOLDER` | Name of the top-level folder holding the ARIMA forecasts |

**Dev and production share the one Space and differ only in the folder names.** Pointing a frontend or backend at dev or production means setting these two folder variables, so both sides must use the same pair of values for the same environment. The folder variables were named `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME` before contract v13; the old names are no longer read.

Every file is stored under a **date prefix** `YYYY-MM-DD/` inside its folder, one date folder per run, so past runs are kept. Full object keys within the Space, where `<X>` means the value of environment variable `X`:

```
<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml
<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_optimization_plot_data.h5
<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv
```

Dev and production are selected by the configured Space and folder names; the key patterns are identical in both. In this document, **"folder"** means one of these two top-level folders, and **"date folder"** means a `YYYY-MM-DD/` prefix inside one.

### Connecting

- **Endpoint:** use the **region-only** endpoint `https://<region>.digitaloceanspaces.com`, pass the **Space name as the S3 `Bucket`**, and pass the full key (starting with the folder name) as the S3 `Key` or `Prefix`.
- **Space name variable:** both the backend and the frontend read the Space name from `FSF_FRONT_END_BUCKET_NAME`. The backend uploads with `Bucket=<space>` and `Key="<folder>/YYYY-MM-DD/<file>"`, and refuses to run with an endpoint that names the Space.
- **Avoid the Space-specific endpoint:** do **not** put the Space name in the endpoint host (`https://<space>.<region>.digitaloceanspaces.com`) and pass a folder name as `Bucket`. Uploads happen to land at the right key that way (boto3 sends path-style requests, so the "bucket" becomes the first key segment), but listings fail with `NoSuchKey`.

### The date prefix

- **Meaning:** `YYYY-MM-DD` is the **data as-of date**: the last NYSE trading day in the price data used by that run, not the day the backend ran. A run on a Sunday stores Friday's results under Friday's date.
  - For the optimization files it equals `date_updated.date_to` in that date folder's `optimization_metadata.yml`.
  - For `all_forecasts.csv` it is the last `pred_date` with a non-empty actual price. The final row (the live forecast) is the next trading day after the folder date.
- **Same-day re-runs** overwrite the files in that date folder. Older dates are never modified or deleted by the backend.
- **Format:** the date prefix is the **first path segment after the top-level folder**. After removing `<folder>/` from the start of a key, the rest must start with exactly `^\d{4}-\d{2}-\d{2}/`, zero-padded. Lexicographic order equals chronological order.
- **Stale objects:** readers must **ignore every key whose segment after the folder is not a valid date**. In particular, objects left directly under a folder by contract v8 and earlier (for example `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/optimization_metadata.yml`, `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/portfolio_optimization_plot_data.h5`, `<FSF_FRONT_END_TIME_SERIES_FOLDER>/all_forecasts.csv`) are stale and are no longer written; the human removes them by hand.
- **Retention:** all dates are kept indefinitely. There is no pruning.

### Which dates are available

A date is **available** only when its data is complete in **both** folders:

1. Key `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml` exists. The backend uploads this file last of all three, so its presence means both optimization files for that date are complete (and, since v10, that the forecasts were uploaded just before).
2. Key `<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv` exists.

Since v10, one backend run computes everything first and then uploads all three files in the order `all_forecasts.csv`, `portfolio_optimization_plot_data.h5`, `optimization_metadata.yml`; if any computation fails, nothing is uploaded. Readers must still check both folders, because dates written before v10 came from two separate scripts. Dates present in only one folder (for example, only one of the two pre-v10 scripts was run) are **not** available and must not be offered to users.

**To find the dates,** list each folder with S3 `ListObjectsV2` using `Bucket=<space>` and `Prefix="<folder>/"`:

- Follow `ContinuationToken` pagination, since a listing returns at most 1,000 entries.
- Strip the `<folder>/` prefix from each returned key, then match the date pattern above.
- `Delimiter="/"` is optional. With it, date folders come back as `CommonPrefixes` such as `<folder>/2026-10-09/`, and you must still strip the folder prefix, then confirm the marker object (a `HeadObject`, or a listing with `Prefix="<folder>/YYYY-MM-DD/"`). Without it, the full keys are returned and the marker objects can be matched directly.

General behavior of every file:

- **History.** Each date folder is a snapshot of one run. Comparing dates is valid (same keys, formats, and units within a contract version), but a date folder written under an older contract version keeps the format of that version.
- **Non-atomic.** The three files are uploaded one after the other (forecasts, `.h5`, then `.yml`). The completion marker rule above prevents reading a half-uploaded date. During a same-day re-run, a date's files can briefly come from different runs.
- **Manual cadence.** The backend is run by hand, so there is no guaranteed update schedule. The frontend should display the data dates (see below) so a stale dataset is visible, rather than assuming data is current.
- **Content type and ACLs.** The backend sets none. Do not rely on the `Content-Type` of an object; decide how to parse from the key name.
- **Not investment advice.** All values are backtested or modeled outputs of a hobby project. Any user-facing text must say so.

## Shared conventions

- **Tickers** are the strings used by Massive.com, e.g. `NVDA`, `QQQ`. At the time of writing the set is `ETHA, FXE, GLD, IBIT, NVDA, QQQ, VOO, VXUS, XBI, XLE`, but **the frontend must not hardcode it**. Read the ticker list from `optimization_metadata.yml` (`tickers`) or from the `all_forecasts.csv` column names. The set can change.
- **Prices** are daily close prices in US dollars as returned by Massive.com with `adjusted=true`, i.e. adjusted by Massive.com's rules. Treat them as "adjusted close" and do not assume dividend adjustment.
- **Returns** are computed from **simple** (not log) daily returns of consecutive adjusted closes and expressed in **percent** (1.5 means 1.5 %).
- **Risk** means the standard deviation (σ) of returns, in percent.
- **Every return and risk uploaded is annualized.** The backend optimizes in daily units and annualizes only when writing files, using 252 trading days per year and **arithmetic** (not compounded) annualization:
  - annualized return = mean daily return × 252
  - annualized risk = daily σ × √252
  - annualized Sharpe ratio = daily Sharpe ratio × √252, which equals (annualized return − annual risk-free rate) / annualized risk

  Arithmetic annualization keeps risk/return relationships linear, so the tangency line is straight, its intercept is the quoted annual risk-free rate, and its slope is the annualized Sharpe ratio. It is not the compounded growth an investor would realize. If user-facing text needs that distinction, say "annualized (arithmetic)" rather than "annual growth".
- The **only** daily quantity uploaded is `risk_free_rate.daily_risk_free_rate`, kept for backward compatibility and named accordingly.
- **Dates** are NYSE trading days (sessions of the New York Stock Exchange calendar, so weekends and NYSE holidays such as Good Friday are excluded, while federal-only holidays such as Columbus Day are included). "Business day" in this document means NYSE trading day. Date-only fields are strings in `YYYY-MM-DD` format.

---

## `optimization_metadata.yml`

**Folder:** `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8. Parse it with a safe YAML loader. Date-like values are written as **quoted strings**, so loaders return strings, not date objects.

### Example

A real file produced by the backend (contract v4+, data through 2026-10-09):

```yaml
annualization:
  method: arithmetic
  periods_per_year: 252
date_updated:
  date_from: '2025-10-07'
  date_to: '2026-10-09'
min_var_portfolio:
  annualized_return: 12.851676947885277
  risk: 6.89459351276483
optimum_portfolio:
  annualized_return: 34.343418898244956
  risk: 12.439838563729614
  sharpe_ratio: 2.4351938928347816
  weights:
    ETHA: 7.59701304408773e-15
    FXE: 3.055998668072217e-14
    GLD: 4.060464045601646e-14
    IBIT: 0.0
    NVDA: 0.0
    QQQ: 19.079171927481532
    VOO: 13.836265883217862
    VXUS: 0.0
    XBI: 27.084562189300787
    XLE: 39.99999999999994
risk_free_rate:
  daily_risk_free_rate: 0.01607142857142857
  risk_free_rate: 4.05
  risk_free_rate_date: '2026-10-08 00:00:00'
tickers:
- XBI
- QQQ
- XLE
- VOO
- NVDA
- VXUS
- GLD
- IBIT
- FXE
- ETHA
```

Things to notice in the example:

- Date strings are quoted, so they load as strings.
- `weights` keys are sorted alphabetically, but `tickers` is in the backend's own order. Neither order is significant.
- Weights carry **floating-point noise**: a weight that is effectively zero can appear as `7.59701304408773e-15`, and a weight at the cap appears as `39.99999999999994` instead of `40.0`. Values can be written in scientific notation.

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `date_updated.date_from` | string `YYYY-MM-DD` | First trading day of price data used in the optimization. |
| `date_updated.date_to` | string `YYYY-MM-DD` | Last trading day of price data used. **This is the "data as of" date to show users.** |
| `tickers` | list of string | Tickers in the optimized portfolio. Order is not significant. |
| `risk_free_rate.risk_free_rate` | float | Latest 3-month U.S. Treasury bill rate (FRED series `DTB3`), **annualized, in percent** (4.39 means 4.39 % per year). |
| `risk_free_rate.daily_risk_free_rate` | float | `risk_free_rate / 252`. **Percent per day**, the rate the backend used internally for its daily optimization. The only daily quantity in the uploaded files; prefer `risk_free_rate` for display. |
| `risk_free_rate.risk_free_rate_date` | string | Observation date of the rate, formatted as a timestamp string such as `2026-10-08 00:00:00`. Parse the first 10 characters as a date. |
| `optimum_portfolio.weights` | map ticker → float | Weights of the maximum Sharpe ratio portfolio, **in percent** (a weight of 22.4 means 22.4 %). Contains one key per ticker. |
| `optimum_portfolio.annualized_return` | float | Expected return of the max Sharpe ratio portfolio, **annualized arithmetically, in percent**: mean daily return (%) × 252. *Changed in v4; it was compounded before.* |
| `optimum_portfolio.risk` | float | Standard deviation of that portfolio's returns, **annualized** (daily σ × √252), **in percent**. |
| `optimum_portfolio.sharpe_ratio` | float | Annualized Sharpe ratio of that portfolio (dimensionless): daily Sharpe × √252 = (`annualized_return` − `risk_free_rate.risk_free_rate`) / `risk`. *Added in v4.* |
| `min_var_portfolio.annualized_return` | float | Expected return of the minimum variance portfolio, **annualized arithmetically, in percent**. *Added in v4.* |
| `min_var_portfolio.risk` | float | Standard deviation of the minimum variance portfolio's returns, **annualized, in percent**. *Added in v4.* |
| `annualization.method` | string | Always `arithmetic` in v4. Describes how every return and risk in both optimization files was annualized. *Added in v4.* |
| `annualization.periods_per_year` | int | Trading days per year used for annualization (`252`). *Added in v4.* |

### Notes

- **Weight noise.** Treat any weight with absolute value below `1e-6` (percent) as zero, and round for display. Never compare a weight to an exact value such as `0` or `40`, and never assume a "zero" weight is written as exactly `0.0`. Do not assume the weights sum to exactly 100; they sum to 100 only within the optimizer's tolerance.
- The optimization is long-only, weights sum to 100 (within optimizer tolerance), and each weight is currently capped at `4 / (number of tickers)` of the portfolio (40 % with 10 tickers). Those constraints are the optimizer's current settings, not a contractual guarantee. Do not hard-code them in the frontend.
- Inputs (mean returns and covariance) are estimated from roughly one year of daily data, so the optimum is noisy and sensitive to the window. Say so in user-facing text.
- Only the **max Sharpe ratio** portfolio's weights are in this file. The minimum variance portfolio has annualized return and risk (here and as a point in the `.h5` file) but no weights.
- `optimum_portfolio` and `min_var_portfolio` values equal the `max_sharpe_ratio` and `min_var_portfolio` points in the `.h5` file from the same run (same units).
- **Detecting older files.** A file without an `annualization` key predates v4; its `annualized_return` is compounded and it has no Sharpe ratio or min-variance section.

---

## `portfolio_optimization_plot_data.h5`

**Folder:** `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`. **Key:** `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/portfolio_optimization_plot_data.h5`. **Format:** HDF5 (written with `h5py`). The file has five top-level groups. Each group has exactly two datasets named `xs` and `ys`: one-dimensional float64 arrays of equal length within the group. `xs` is the horizontal axis (risk), and `ys` is the vertical axis (return). All values are **annualized (arithmetic), in percent** (see "Shared conventions"), and the file contains no dates and no ticker names (use `optimization_metadata.yml` for those).

### Root attributes (added in v4)

The root group `/` carries attributes that describe the units, so a reader can verify them instead of assuming:

| Attribute | Type | Value in v4 |
| --- | --- | --- |
| `return_units` | string | `percent_per_year` |
| `risk_units` | string | `percent_per_year` (annualized standard deviation, in percent) |
| `return_type` | string | `simple` |
| `annualization` | string | `arithmetic` |
| `periods_per_year` | int64 | `252` |

**Detecting older files.** A file with **no** `annualization` attribute predates v4 and contains **daily** values in percent. Readers should check this attribute and refuse to plot (or convert explicitly) rather than mixing daily and annualized data.

### Layout

```
/                             attrs: return_units, risk_units, return_type, annualization, periods_per_year
/efficient_frontier/xs        float64, shape (100,)    annualized risk σ (%) of the minimum-variance portfolio for each target return
/efficient_frontier/ys        float64, shape (100,)    the 100 target annualized returns (%), evenly spaced
/tangency_line/xs             float64, shape (100,)    annualized risk σ (%), evenly spaced from 0 to the largest non-NaN efficient_frontier/xs
/tangency_line/ys             float64, shape (100,)    annualized return (%): risk_free_rate + sharpe_ratio × xs
/simulated_portfolios/xs      float64, shape (10000,)  annualized risk σ (%) of random long-only portfolios (Monte Carlo)
/simulated_portfolios/ys      float64, shape (10000,)  annualized return (%) of those portfolios
/max_sharpe_ratio/xs          float64, shape (1,)      annualized risk σ (%) of the max Sharpe ratio portfolio
/max_sharpe_ratio/ys          float64, shape (1,)      its annualized return (%)
/min_var_portfolio/xs         float64, shape (1,)      annualized risk σ (%) of the minimum variance portfolio
/min_var_portfolio/ys         float64, shape (1,)      its annualized return (%)
```

In the tangency line formula, `risk_free_rate` and `sharpe_ratio` are the YAML fields `risk_free_rate.risk_free_rate` and `optimum_portfolio.sharpe_ratio` from the same run.

### Group meanings

| Group | What to draw |
| --- | --- |
| `efficient_frontier` | A line. Points run from the minimum variance portfolio's return up to the highest simulated return. |
| `tangency_line` | A straight line starting at (0, annual risk-free rate) with slope equal to the annualized max Sharpe ratio. Also called the capital market line. |
| `simulated_portfolios` | A scatter cloud (about 10,000 points) of random portfolios, to show the feasible region. |
| `max_sharpe_ratio` | A single marker: the max Sharpe ratio (tangency) portfolio. |
| `min_var_portfolio` | A single marker: the minimum variance portfolio. |

### Notes

- **Same units as the YAML file.** Since v4, this file and `optimization_metadata.yml` are both annualized (arithmetic) in percent, so YAML values can be plotted directly on the same axes.
- **`NaN` values.** An `efficient_frontier/xs` entry is `NaN` when the optimizer found no feasible portfolio for that target return. Readers must drop or break the line at `NaN` and must not treat it as an error. `tangency_line` never contains `NaN` (since v5).
- **Array lengths** (100 and 10,000) are current settings. Read the shape from the file and do not hard-code them.
- Suggested axis labels are "Annualized Risk (σ, %)" and "Annualized Return (%)".
- The backend does no plotting. All visualization of this data is the frontend's responsibility.

---

## `all_forecasts.csv`

**Folder:** `FSF_FRONT_END_TIME_SERIES_FOLDER`. **Key:** `<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv`. **Format:** CSV with a header row, UTF-8, comma-separated, written by `pandas.DataFrame.to_csv(index=True)` (the index is the first column). Read it with the first column parsed as dates, e.g. `pandas.read_csv(..., index_col="pred_date", parse_dates=True)`.

### Shape

A wide table with one row per forecast date and, for every ticker `T`, two columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `pred_date` (first column, the index) | timestamp string, e.g. `2026-10-09 17:00:00` | The NYSE trading day being forecast. The time of day (`17:00:00`) is a fixed label with no meaning; use only the date. |
| `T_arima` | float | The ARIMA model's one-step-ahead forecast of `T`'s adjusted close (US dollars) for `pred_date`. |
| `T` | float, may be empty | The actual adjusted close (US dollars) of `T` on `pred_date`. **Empty on rows that have no actual value yet** (see below). |

Example header for three tickers (illustrative):

```
pred_date,GLD_arima,GLD,NVDA_arima,NVDA,QQQ_arima,QQQ
```

Do not rely on column order. Identify forecast columns by the `_arima` suffix, and actual-price columns by matching the remaining name against the ticker list.

### How the forecasts are produced

- **Horizon:** one business day ahead. Each row's forecast was made using only data **before** `pred_date`.
- **Walk-forward:** for each forecast date, an ARIMA model is refit on a sliding window of the preceding 20 business days of prices (log-transformed and differenced once), with orders `p` and `q` chosen from 0 to 1 and 0 to 1 (excluding 0,0) by RMSE and BIC rank on the window. The window slides one business day per row.
- **Back to dollars:** the model forecasts a log return. The backend converts it to a price with `last_price × exp(forecasted log return)`, so `T_arima` is in the same units as `T`.
- **Last row:** the final row is the first NYSE trading day after the latest available data. Its `T` values are **empty** because that day has not happened yet. Treat that row as the live forecast and the rows above it as the backtest.
- **Rows** are unique by `pred_date` and sorted ascending. The number of rows is about one month of business days and is not fixed.

### Not included

The file does **not** contain RMSE, the selected `(p, q)` orders, confidence intervals, or a naive-baseline forecast. If the frontend needs an error metric, compute it from the `T` and `T_arima` columns on rows where `T` is not empty, and state which metric and window it uses. If any of these should be uploaded, it needs a contract change (an additive one is preferred).

---

## Reader requirements (for the frontend)

1. **Ignore unknown things.** Unknown YAML keys, HDF5 groups or datasets, and CSV columns must be ignored without error. This is what makes additive changes safe.
2. **Do not hardcode** the ticker list, array lengths, or row counts.
3. **Handle `NaN` and empty cells** as described above.
4. **Show data dates.** Display the selected date (equal to `date_updated.date_to`) so users can tell how current the data is.
5. **Fail soft.** If a file is missing or does not parse, show an explanatory message instead of crashing. Missing or partial data is expected while a run is mid-upload.
6. **Label units.** Daily versus annualized, and percent versus dollars, must be stated wherever a value is shown.
7. **Date selection.** Offer only *available* dates (see "Which dates are available"), newest first, with the newest selected by default. Load all three files from the **same** date in both folders; never mix dates across files. Ignore keys whose segment after the folder is not a valid date, including stale objects directly under a folder from contract v8 and earlier.
8. **Do not assume consecutive dates.** Runs are manual, so dates have gaps (weekends, holidays, days not run).

## Verification status

Checked against real backend output (2026-10-11), unless dated otherwise:

- Storage layout (2026-10-12):
  - **Console (the human):** the account has exactly one Space (bucket), whose name is the value of `FSF_FRONT_END_BUCKET_NAME`, in region `nyc3`.
  - **Programmatic listing** with the recommended configuration (endpoint `https://nyc3.digitaloceanspaces.com`, `ListObjectsV2(Bucket=<space>, Prefix="<folder>/")`) returned exactly these keys:
    - `time-series/2026-10-09/all_forecasts.csv`
    - `portfolio-optimization/2026-10-09/optimization_metadata.yml`
    - `portfolio-optimization/2026-10-09/portfolio_optimization_plot_data.h5`
    - `portfolio-optimization/optimization_metadata.yml`, a stale pre-v9 object directly under the folder, which readers ignore. The human has since deleted it.

    A later listing after the backend switched to `Bucket=<space>` with the folder in the key showed the same three dated keys and no stale objects.

    So the folder values were `time-series` and `portfolio-optimization` at the time.
  - **Folder names as buckets fail:** `ListObjectsV2` with a folder name as `Bucket` fails, with `NoSuchBucket` from the backend's configuration and `NoSuchKey` from the frontend's.

- `optimization_metadata.yml` (v5 output): every key in the example above, including the v4 additions, quoted date strings, and floating-point noise in weights. The values are internally consistent: `sharpe_ratio` = (`annualized_return` − `risk_free_rate`) / `risk` = (34.343 − 4.05) / 12.440 = 2.435.
- `portfolio_optimization_plot_data.h5` (v5 output, `h5dump -A`): the five root attributes with the documented values and types (variable-length UTF-8 strings; `periods_per_year` is `H5T_STD_I64LE`), plus group names, dataset names, dtype (`H5T_IEEE_F64LE`), and shapes (100, 1, 1, 10000, 100).
- `all_forecasts.csv`: first column `pred_date`, the `T_arima` / `T` naming, and empty actuals in the final row.
- The "How the forecasts are produced" description matches the backend's `run()` defaults.
- `.h5` values (v5 output, `h5dump -d`): `max_sharpe_ratio` = (12.4398, 34.3434) and `min_var_portfolio` = (6.89459, 12.8517) equal the YAML `risk` / `annualized_return` of the same run. `tangency_line/ys` starts at 4.05 (= `risk_free_rate`), is evenly spaced (a straight line), and passes through the max Sharpe ratio point with slope `sharpe_ratio`.
- The annualization formulas and the straight tangency line are covered by the backend's unit tests (`tests/test_annualization.py`).

**Not yet** verified against real data (derived from the code):

- That the date folder equals `date_updated.date_to`, and the upload order (forecasts, `.h5`, then `optimization_metadata.yml`). The construction of the date part of the key and of local paths is covered by unit tests (`tests/test_layout.py`).
- The v8 NYSE calendar behavior on real data (covered by unit tests in `tests/test_date_manager.py`).
- `NaN` entries in `efficient_frontier/xs` (inferred from the code for infeasible target returns). Readers must handle `NaN` regardless.

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 13 | 2026-10-12 | **Configuration rename; no objects move and key strings are unchanged.** The environment variables naming the two top-level folders are renamed so their names say what they hold: `PORTFOLIO_OPTIMIZATION_SPACE_NAME` is now `FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER`, and `TIME_SERIES_SPACE_NAME` is now `FSF_FRONT_END_TIME_SERIES_FOLDER`. Values are unchanged. No fallback: the backend reads only the new names, and the human has already renamed them in the backend `.env` for dev and production. Documented that dev and production share the one Space and differ only in folder names, and listed `FSF_FRONT_END_BUCKET_NAME` with the folder variables. Frontend must: read the new names, rename its Hugging Face secrets and any local `.env` (same values), and drop the old names. |
| 12 | 2026-10-12 | Documentation only; no change to keys, formats, or reader behavior. The v11 text was amended in place after it was first written, so a copy taken earlier may differ. This version is the reconciled text for both repos. Amendments since the first v11 text: the "Connecting" section states that the backend also reads `FSF_FRONT_END_BUCKET_NAME` and uploads with `Bucket=<space>`, `Key="<folder>/YYYY-MM-DD/<file>"`; "Verification status" records the programmatic listing of the real keys, the deleted stale object, and a second listing after the backend change; the v11 changelog entry notes the backend change. Frontend: replace its copy with this file. |
| 11 | 2026-10-12 | **Documentation correction; no objects move and key strings are unchanged.** Earlier versions described `PORTFOLIO_OPTIMIZATION_SPACE_NAME` and `TIME_SERIES_SPACE_NAME` as bucket names. They are top-level folder names inside **one** Space (bucket), so the full keys are `<folder>/YYYY-MM-DD/<file>`. Uploads landed correctly only because boto3, given an endpoint containing the Space name, sends path-style requests. Rewrote "Where the data lives", "The date prefix", "Which dates are available", the per-file Folder/Key lines, and reader requirement 7; added "Connecting". Frontend must: add `FSF_FRONT_END_BUCKET_NAME` (the Space name), use the region-only endpoint `https://<region>.digitaloceanspaces.com`, list each folder with `ListObjectsV2(Bucket=<space>, Prefix="<folder>/")` with pagination, strip the folder prefix before matching dates, and read objects with `Bucket=<space>`, `Key="<folder>/YYYY-MM-DD/<file>"`. The backend was changed the same way (it reads `FSF_FRONT_END_BUCKET_NAME` and passes the folder in the key), which produces byte-identical object keys. |
| 10 | 2026-10-12 | Non-breaking; no change to keys, file names, formats, columns, shapes, or units. The backend now has a single entry point that produces all three files in one run: one price download, ARIMA trained on the last 40 trading sessions of that data (the same window as before), then the optimization. Uploads happen only after every computation succeeds, in the order `all_forecasts.csv`, `.h5`, `optimization_metadata.yml`, so the metadata file now marks a complete date across both buckets. The availability rule (both buckets) is unchanged because earlier dates came from two separate scripts. |
| 9 | 2026-10-12 | **Breaking (keys).** Every file moves under a date prefix `YYYY-MM-DD/` (the data as-of date = last NYSE trading day in the run's data), keeping one folder per run instead of overwriting. File names, formats, columns, shapes, and units are unchanged. Root-level keys are **no longer written** (clean break, no dual-write); stale root objects remain until removed by hand. Defined date availability as the intersection of both buckets, with `optimization_metadata.yml` as the optimization completion marker. Frontend must: list date prefixes in both buckets (`ListObjectsV2`, `Delimiter="/"`, paginated), keep only dates complete in both, show them in a dropdown newest-first with the newest selected by default, load all files from the selected date's folder, and ignore root-level objects. Ship the frontend change with the first prefixed upload to production. |
| 8 | 2026-10-12 | Calendar fix, non-breaking (no change to keys, columns, shapes, or units): business days now follow the NYSE trading calendar instead of the US federal holiday calendar. Effects: `pred_date` and the ARIMA training windows skip NYSE closures (e.g. Good Friday) and include federal-only holidays (e.g. Columbus Day, Veterans Day), and downloaded data now ends on the last NYSE session instead of possibly omitting one. |
| 7 | 2026-10-11 | Documentation only: recorded verification of the `.h5` point values and tangency line against real output. No change to keys, shapes, units, or meaning. |
| 6 | 2026-10-11 | Documentation only: replaced the illustrative v4 YAML example with real output and recorded the verification of the v4 YAML keys and `.h5` root attributes. No change to keys, shapes, units, or meaning. |
| 5 | 2026-10-11 | Bug fix, non-breaking: the tangency line's risk range now ends at the largest **non-NaN** efficient-frontier risk. Before, a `NaN` frontier point could make the whole `tangency_line` `NaN`. Documented that `tangency_line` contains no `NaN`. No change to keys, shapes, or units. |
| 4 | 2026-10-11 | **Breaking (units).** Every return and risk in the optimization files is now annualized arithmetically (return × 252, risk × √252). `.h5`: every dataset changes from daily % to annualized %, with the same keys and shapes, plus new root attributes describing units. YAML: `optimum_portfolio.annualized_return` changes from compounded to arithmetic (about 40.9 → 34.3 on the 2026-10-09 data). Additive: `optimum_portfolio.sharpe_ratio`, `min_var_portfolio`, `annualization`. `daily_risk_free_rate` kept unchanged. Frontend must: relabel axes as annualized, stop any daily→annual conversion it does, check the `.h5` `annualization` attribute (absent means a pre-v4 daily file), and update any display of `annualized_return`. |
| 3 | 2026-10-11 | Backend no longer renders a local PNG plot (it was never uploaded). Removed the note about it. No change to uploaded keys, formats, shapes, or units. |
| 2 | 2026-10-11 | Replaced the illustrative YAML example with a real file. Documented floating-point noise in weights (scientific notation, `~1e-14` for zero, `39.99999999999994` at the cap) and reader handling. Recorded verification status. No change to keys, columns, shapes, or units. |
| 1 | 2026-10-11 | Initial contract, derived from the backend code. |
