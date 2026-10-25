# Interface Contract: fuzzy-system-research (backend) ↔ fuzzy-system-finance (frontend)

| | |
| --- | --- |
| **Contract version** | 29 |
| **Last updated** | 2026-10-24 |
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
<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts_metadata.yml   (since v21)
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

1. Key `<FSF_FRONT_END_PORTFOLIO_OPTIMIZATION_FOLDER>/YYYY-MM-DD/optimization_metadata.yml` exists. The backend uploads this file last of all the files for a date, so its presence means both optimization files for that date are complete (and, since v10, that the forecasts were uploaded just before).
2. Key `<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts.csv` exists.

Since v10, one backend run computes everything first and then uploads its files in the order `all_forecasts.csv`, `all_forecasts_metadata.yml` (since v21), `portfolio_optimization_plot_data.h5`, `optimization_metadata.yml`; if any computation fails, nothing is uploaded. Because `optimization_metadata.yml` is last, its presence (for v21+ runs) also means the forecasts metadata is complete. `all_forecasts_metadata.yml` is not part of the availability rule, since older dates do not have it. Readers must still check both folders, because dates written before v10 came from two separate scripts. Dates present in only one folder (for example, only one of the two pre-v10 scripts was run) are **not** available and must not be offered to users.

**To find the dates,** list each folder with S3 `ListObjectsV2` using `Bucket=<space>` and `Prefix="<folder>/"`:

- Follow `ContinuationToken` pagination, since a listing returns at most 1,000 entries.
- Strip the `<folder>/` prefix from each returned key, then match the date pattern above.
- `Delimiter="/"` is optional. With it, date folders come back as `CommonPrefixes` such as `<folder>/2026-10-09/`, and you must still strip the folder prefix, then confirm the marker object (a `HeadObject`, or a listing with `Prefix="<folder>/YYYY-MM-DD/"`). Without it, the full keys are returned and the marker objects can be matched directly.

General behavior of every file:

- **History.** Each date folder is a snapshot of one run. Comparing dates is valid (same keys, formats, and units within a contract version), but a date folder written under an older contract version keeps the format of that version.
- **Non-atomic.** The files are uploaded one after the other (forecasts, forecasts metadata, `.h5`, then optimization metadata). The completion marker rule above prevents reading a half-uploaded date. During a same-day re-run, a date's files can briefly come from different runs.
- **Manual cadence.** The backend is run by hand, so there is no guaranteed update schedule. The frontend should display the data dates (see below) so a stale dataset is visible, rather than assuming data is current.
- **Content type and ACLs.** The backend sets none. Do not rely on the `Content-Type` of an object; decide how to parse from the key name.
- **Not investment advice: one disclaimer, shown once.** Display this wording exactly once, at the bottom of the UI:

  > Not investment advice. Every value shown is a backtest or model output computed from historical prices, and none of it predicts future results or recommends any investment.

  It covers every file in this contract. Do not repeat it in plot captions, tooltips, or metric descriptions; the suggested wording elsewhere in this document deliberately leaves it out.

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

Since v28, `min_var_portfolio` also has a `weights` map, in the same form as `optimum_portfolio.weights` (the example above predates it). Illustrative values:

```yaml
min_var_portfolio:
  annualized_return: 12.851676947885277
  risk: 6.89459351276483
  weights:            # one key per ticker; only three shown here
    ETHA: 1.2e-15
    FXE: 39.99999999999998
    GLD: 21.403
```

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `date_updated.date_from` | string `YYYY-MM-DD` | First trading day of price data used in the optimization. |
| `date_updated.date_to` | string `YYYY-MM-DD` | Last trading day of price data used. **This is the "data as of" date to show users.** |
| `tickers` | list of string | Tickers in the optimized portfolio. Order is not significant. |
| `risk_free_rate.risk_free_rate` | float | Most recent **published** 3-month U.S. Treasury bill rate (FRED series `DTB3`), **annualized, in percent** (4.39 means 4.39 % per year). Days with no published value (bond-market holidays) are skipped, so this is never `NaN`; since v14, a run with no valid rate fails instead of uploading. |
| `risk_free_rate.daily_risk_free_rate` | float | `risk_free_rate / 252`. **Percent per day**, the rate the backend used internally for its daily optimization. The only daily quantity in the uploaded files; prefer `risk_free_rate` for display. |
| `risk_free_rate.risk_free_rate_date` | string | Observation date of the rate (the last day with a published value, which can be earlier than the last FRED row), formatted as a timestamp string such as `2026-10-08 00:00:00`. Parse the first 10 characters as a date. |
| `optimum_portfolio.weights` | map ticker → float | Weights of the maximum Sharpe ratio portfolio, **in percent** (a weight of 22.4 means 22.4 %). Contains one key per ticker. |
| `optimum_portfolio.annualized_return` | float | Expected return of the max Sharpe ratio portfolio, **annualized arithmetically, in percent**: mean daily return (%) × 252. *Changed in v4; it was compounded before.* |
| `optimum_portfolio.risk` | float | Standard deviation of that portfolio's returns, **annualized** (daily σ × √252), **in percent**. |
| `optimum_portfolio.sharpe_ratio` | float | Annualized Sharpe ratio of that portfolio (dimensionless): daily Sharpe × √252 = (`annualized_return` − `risk_free_rate.risk_free_rate`) / `risk`. *Added in v4.* |
| `min_var_portfolio.annualized_return` | float | Expected return of the minimum variance portfolio, **annualized arithmetically, in percent**. *Added in v4.* |
| `min_var_portfolio.risk` | float | Standard deviation of the minimum variance portfolio's returns, **annualized, in percent**. *Added in v4.* |
| `min_var_portfolio.weights` | map ticker → float | Weights of the minimum variance portfolio, **in percent**, one key per ticker, in the same form and with the same floating-point noise as `optimum_portfolio.weights`. *Added in v28; absent in older files, so treat it as optional.* |
| `annualization.method` | string | Always `arithmetic` in v4. Describes how every return and risk in both optimization files was annualized. *Added in v4.* |
| `annualization.periods_per_year` | int | Trading days per year used for annualization (`252`). *Added in v4.* |

### Notes

- **Weight noise.** These rules apply to both weight maps (`optimum_portfolio.weights` and, since v28, `min_var_portfolio.weights`). Treat any weight with absolute value below `1e-6` (percent) as zero, and round for display. Never compare a weight to an exact value such as `0` or `40`, and never assume a "zero" weight is written as exactly `0.0`. Do not assume the weights sum to exactly 100; they sum to 100 only within the optimizer's tolerance.
- **Converged optimizations only** (since v29). If either the max Sharpe ratio or the minimum variance optimization does not converge, the backend run fails and nothing is uploaded for that date, so every uploaded weight, return, risk, and Sharpe ratio comes from a converged optimization. (An efficient-frontier target return the optimizer cannot reach is still recorded as `NaN` in the `.h5` file; see that section.)
- Both portfolios are optimized under the same constraints: long-only, weights sum to 100 (within optimizer tolerance), and each weight is currently capped at `4 / (number of tickers)` of the portfolio (40 % with 10 tickers). Those constraints are the optimizer's current settings, not a contractual guarantee. Do not hard-code them in the frontend.
- Inputs (mean returns and covariance) are estimated from roughly one year of daily data, so the optimum is noisy and sensitive to the window. Say so in user-facing text.
- **Weights of both portfolios.** Since v28 this file has the weights of both the max Sharpe ratio portfolio (`optimum_portfolio.weights`) and the minimum variance portfolio (`min_var_portfolio.weights`). Before v28, only the max Sharpe ratio weights were included; for those dates, show the minimum variance portfolio's return and risk without weights. The minimum variance portfolio depends only on the estimated covariance of returns, not on expected returns or the risk-free rate, so its weights are usually more stable from one date to the next than the max Sharpe ratio weights.
- **Suggested wording near a table of both portfolios' weights:** "The minimum variance portfolio uses only the estimated covariance of returns, not expected returns, so it is usually more stable from one date to the next than the max Sharpe portfolio. Both are estimated from about a year of daily data and share the same constraints (long-only, capped weights)."
- `optimum_portfolio` and `min_var_portfolio` values equal the `max_sharpe_ratio` and `min_var_portfolio` points in the `.h5` file from the same run (same units).
- **Detecting older files.** A file without an `annualization` key predates v4; its `annualized_return` is compounded and it has no Sharpe ratio or min-variance section. A file whose `min_var_portfolio` has no `weights` key predates v28.

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

A wide table with one row per forecast date and, for every ticker `T`, six columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `pred_date` (first column, the index) | timestamp string, e.g. `2026-10-09 17:00:00` | The NYSE trading day being forecast. The time of day (`17:00:00`) is a fixed label with no meaning; use only the date. |
| `T_arima` | float, may be empty | The ARIMA model's one-step-ahead forecast of `T`'s adjusted close (US dollars) for `pred_date`. Empty when `T`'s close on the previous session is missing from the data (since v20; see "Missing data"). |
| `T_arima_p` | integer, may be empty | The autoregressive order `p` of the model that made this row's `T_arima` forecast. Present on every row where `T_arima` is, including the last; empty where `T_arima` is. *Added in v16.* |
| `T_arima_d` | integer, may be empty | The differencing order `d` of that model, stated on **log prices**. Always `1` where `T_arima` has a value (including the last row); empty where `T_arima` is. *Added in v18.* |
| `T_arima_q` | integer, may be empty | The moving-average order `q` of that model. Present on every row where `T_arima` is, including the last; empty where `T_arima` is. *Added in v16.* |
| `T_naive` | float | The **naive (random walk) baseline** forecast of `T`'s adjusted close (US dollars) for `pred_date`: the most recent observed close before `pred_date`, which is the previous trading day's close unless that day is missing from the data. It uses exactly the data the row's ARIMA model saw (never `pred_date`'s own close), so the two forecasts are directly comparable. `T_arima` / `T_naive` = exp(ARIMA's forecast log return). Present on every row, including the last. *Added in v19.* |
| `T` | float, may be empty | The actual adjusted close (US dollars) of `T` on `pred_date`. **Empty on rows that have no actual value yet** (see below). |

Each model is fit as ARIMA(p, 0, q) on daily **log returns**, which is the same model as ARIMA(p, 1, q) on log prices. The three order columns describe it on log prices: `(T_arima_p, T_arima_d, T_arima_q)` is the `(p, d, q)` of ARIMA(p, d, q) on log prices, so `T_arima_d` is always `1` where present. Orders `p` and `q` are re-selected for every row and ticker (see "How the forecasts are produced"), so they can change from row to row. With the current settings each is 0 or 1, and `(p, q) = (0, 0)` never occurs. Do not hard-code that range, and read `d` from the column rather than assuming 1. Date folders written before v16 have no order columns, and those written before v18 have no `T_arima_d` column, so treat all three as optional.

Example header for three tickers (illustrative):

```
pred_date,GLD_arima,GLD_arima_p,GLD_arima_d,GLD_arima_q,GLD_naive,GLD,NVDA_arima,NVDA_arima_p,NVDA_arima_d,NVDA_arima_q,NVDA_naive,NVDA,QQQ_arima,QQQ_arima_p,QQQ_arima_d,QQQ_arima_q,QQQ_naive,QQQ
```

Do not rely on column order. Identify columns by **exact suffix match**:

- Forecast columns end with `_arima`.
- Order columns end with `_arima_p`, `_arima_d`, or `_arima_q`.
- Naive-baseline columns end with `_naive` (since v19; absent in older date folders, so treat them as optional).
- Actual-price columns are the remaining names that match a ticker exactly.

Do **not** treat every column *containing* `_arima` as a forecast column, because the order columns also contain it.

### Suggested user-facing wording

The frontend may adapt this wording, but it must keep its meaning: the model is fit on log returns, the orders are stated on log prices, and the orders can change daily. It assumes readers know what ARIMA(p, d, q) means.

**Tooltip on a forecast point** (fill in the row's `T_arima_p`, `T_arima_d`, `T_arima_q`):

> ARIMA({p}, {d}, {q}) on log prices
> (fit as ARIMA({p}, 0, {q}) on daily log returns)

**Text near the forecast plot** (fill in `{w}`, the number of closes in each training window, as `forecast_window.training_window_returns` + 1 from the same date's `all_forecasts_metadata.yml`; if that file is absent, as for dates before v21, leave out the parenthesis):

> **How these forecasts are made.** Each point is a one-trading-day-ahead forecast of the closing price, made only with data available before that day. For every day and ticker, an ARIMA model is refit on a window of the most recent closes before that day ({w} trading days), and its orders p and q are chosen again, so they can change from one day to the next. Hover over a forecast to see the model behind it.
>
> **Why d = 1.** The model is fit to daily log returns as ARIMA(p, 0, q). Turning prices into returns is one round of differencing, so the same model on log prices is ARIMA(p, 1, q). The tooltip shows the log-price version.
>
> **Naive forecast:** today's close = yesterday's close. A model adds value only if its errors are smaller than the naive baseline.

### How the forecasts are produced

- **Horizon:** one business day ahead. Each row's forecast was made using only data **before** `pred_date`.
- **Walk-forward:** for each forecast date, an ARIMA model is refit on a sliding window of the closes ending on the last session before `pred_date`. The window length is a backend setting that can change between runs, so **do not assume it**: read `forecast_window.training_window_returns` (daily log returns per window; closes = that value + 1) and `forecast_window.training_window_start_date` from the same date's `all_forecasts_metadata.yml`. Orders `p` and `q` are chosen from 0 to 1 and 0 to 1 (excluding 0,0): since v24, each candidate is fit once on the whole window and the one with the **lowest BIC** wins, and its model makes the forecast. Exactly equal BICs (in practice only when fits fail) go to fewer parameters (`p + q`), then to the lower `p`, so the choice is reproducible. A candidate whose fit fails ranks last; if every candidate's fit fails, the run fails instead of uploading. The window slides one business day per row. Each window contains only NYSE trading sessions (since v15), so one model time step is one session; NYSE holidays are skipped, not treated as missing observations. Each model includes a constant, so its forecast includes the window's average daily log return (a drift term).
- **Before v24** the training window was much shorter, and orders were scored by one-step-ahead RMSE and BIC rank on rolling sub-windows of it (with the v17 and v23 rules for failed fits and ties). For dates from v21 on, the metadata records the window (`training_window_returns`) and sub-window (`models.arima.order_selection_subwindow_returns`) used; the changelog gives the earlier values.
- **Back to dollars:** the model forecasts a log return. The backend converts it to a price with `last_price × exp(forecasted log return)`, so `T_arima` is in the same units as `T`.
- **Last row:** the final row is the first NYSE trading day after the latest available data. Its `T` values are **empty** because that day has not happened yet. Treat that row as the live forecast and the rows above it as the backtest.
- **Rows** are unique by `pred_date` and sorted ascending. There is normally one backtest row with an actual close per scored day, plus the live row, so normally `evaluation_window.target_trading_days` + 1 rows. The count is a backend setting and can be lower when data is missing, so **do not hard-code it**: count the rows of the CSV (the backtest is the rows with a non-empty actual close), and read `target_trading_days` from the metadata for the number that was scored.
- **Missing data** (since v20). Forecasts are made only from closes that are present. If a whole session is absent from the data, there is no row for the session after it, because a one-step-ahead forecast for that day would need the missing close; the row for the missing session itself has an empty `T`. If only one ticker's close is missing on a session, the next row keeps its `pred_date`, but that ticker's `T_arima` and order columns are empty, while its `T_naive` still uses the most recent observed close. Rows are always unique by `pred_date`. Because the order columns can be empty, a CSV reader may load them as floats; treat their non-empty values as whole numbers.

### Not included

The file does **not** contain error metrics or confidence intervals. The naive baseline is included since v19 (`T_naive`). The selected model orders are included: `T_arima_p` and `T_arima_q` since v16, `T_arima_d` since v18. Since v21, error metrics for both forecasts are in `all_forecasts_metadata.yml` (next section). Use those rather than computing metrics in the frontend, so every surface reports the same numbers.

---

## `all_forecasts_metadata.yml`

**Folder:** `FSF_FRONT_END_TIME_SERIES_FOLDER`. **Key:** `<FSF_FRONT_END_TIME_SERIES_FOLDER>/YYYY-MM-DD/all_forecasts_metadata.yml`, in the same date folder as `all_forecasts.csv`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8; parse with a safe YAML loader. *Added in v21; date folders written before v21 do not have it, so treat it as optional.*

It describes the most recent forecasts in that date's `all_forecasts.csv`: the training window of the live forecast, and how accurate the ARIMA and naive forecasts were over the last `evaluation_window.target_trading_days` trading days with an actual close. Both window lengths are backend settings that can change between runs; this file records the values each run used, so read them here rather than assuming them. The backend computes it from the CSV exactly as uploaded.

**All keys are lowercase snake_case, including the ticker keys under `ticker_metrics`.** Ticker symbols are lowercased here (`xbi`, `qqq`, ...), unlike every other file, where they are uppercase. Match them with a case-insensitive comparison, e.g. `ticker.lower()`.

### Example

Illustrative values; the structure matches the backend's output. Values that depend on the window lengths are shown as `<placeholders>`, because those lengths are backend settings: read them from the real file.

```yaml
evaluation_window:
  end_date: '2026-10-09'
  start_date: '<first scored session, YYYY-MM-DD>'
  target_trading_days: <number of scored days>
forecast_window:
  forecast_date: '2026-10-12'
  horizon_trading_days: 1
  training_window_end_date: '2026-10-09'
  training_window_returns: <daily log returns per training window>
  training_window_start_date: '<first session of the live training window, YYYY-MM-DD>'
models:
  arima:
    d: 1
    description: ARIMA(p, 0, q) on daily log returns, which is ARIMA(p, 1, q) on log prices
    excluded_orders:
    - - 0
      - 0
    order_selection: lowest BIC, each candidate order fit once on the whole training window
    order_selection_subwindow_returns: <same as training_window_returns>
    order_selection_tie_break: fewer parameters (p + q), then lower p
    p_values:
    - 0
    - 1
    q_values:
    - 0
    - 1
  naive:
    description: the most recent observed close (random walk)
ticker_metrics:
  xbi:
    arima:
      directional_accuracy: 0.55
      mae: 1.21
      mape: 1.32
      n_directional: <scored days with a nonzero move>
      rmse: 1.52
    evaluation_end_date: '2026-10-09'
    evaluation_start_date: '<first scored session, YYYY-MM-DD>'
    n_evaluated: <normally target_trading_days>
    naive:
      mae: 1.25
      mape: 1.37
      rmse: 1.58
    relative_mae: 0.968
    skill: 0.032
units:
  directional_accuracy: fraction
  mae: usd_per_share
  mape: percent
  relative_mae: ratio
  rmse: usd_per_share
  skill: ratio
```

### Fields

| Path | Type | Meaning |
| --- | --- | --- |
| `forecast_window.training_window_start_date` | string `YYYY-MM-DD` | First trading session of the live forecast's training window. |
| `forecast_window.training_window_end_date` | string `YYYY-MM-DD` | Last session of that window, the latest close used. Equals the date folder. |
| `forecast_window.forecast_date` | string `YYYY-MM-DD` | The date of the live forecast: the CSV's last `pred_date`, one NYSE trading day after `training_window_end_date`. |
| `forecast_window.training_window_returns` | int | Daily log returns in each training window; the window holds one more close than this. A backend setting that can change between runs, so read it from each file and **use it wherever user-facing text states the training window length** (see the suggested wording for the forecast plot). Also use it to tell methods apart when comparing dates. |
| `forecast_window.horizon_trading_days` | int | Forecast horizon (`1`: one trading day ahead). |
| `evaluation_window.target_trading_days` | int | Number of most recent days with an actual close that are scored. A backend setting that can change between runs, so **read this value wherever user-facing text states the number of scored days** (see the suggested wording below) instead of writing a fixed number. |
| `evaluation_window.start_date`, `.end_date` | string `YYYY-MM-DD` or null | Earliest start and latest end of the per-ticker evaluation windows. Normally the same for every ticker. |
| `units.*` | string | Units of each metric: `usd_per_share` (RMSE, MAE), `percent` (MAPE), `ratio` (relative MAE, skill), `fraction` (directional accuracy, from 0 to 1). |
| `models.arima.*`, `models.naive.description` | mixed | How each forecast is made: the ARIMA order search (`p_values`, `q_values`, `excluded_orders`), `d`, the order-selection rule and its sub-window length, the tie-break rule (`order_selection_tie_break`, since v23; absent in older files), and the naive rule. These are taken from the backend's settings. Since v24, candidates are scored on one window, the whole training window, so `order_selection_subwindow_returns` equals `forecast_window.training_window_returns` (it was the rolling sub-window length before). |
| `ticker_metrics.<ticker>.n_evaluated` | int | Days actually scored. Normally equal to `evaluation_window.target_trading_days`; fewer if a ticker's forecast or close was missing on some of those days. Both models are always scored on the same days. |
| `ticker_metrics.<ticker>.evaluation_start_date`, `.evaluation_end_date` | string `YYYY-MM-DD` or null | First and last of this ticker's `target_trading_days` most recent days with an actual close. |
| `ticker_metrics.<ticker>.arima.rmse`, `.naive.rmse` | float or null | Root mean squared error of the forecast against the actual close, **US dollars per share**. |
| `ticker_metrics.<ticker>.arima.mae`, `.naive.mae` | float or null | Mean absolute error, **US dollars per share**. |
| `ticker_metrics.<ticker>.arima.mape`, `.naive.mape` | float or null | Mean absolute percentage error, **in percent** (1.3 means 1.3 %). Scale-free, so it can be compared across tickers. |
| `ticker_metrics.<ticker>.relative_mae` | float or null | ARIMA MAE ÷ naive MAE over the same days. **Below 1 means ARIMA was more accurate than the naive forecast.** `null` if naive MAE is 0 or nothing was scored. |
| `ticker_metrics.<ticker>.skill` | float or null | `1 − relative_mae`. **Above 0 means ARIMA beat the naive forecast**, 0 means no better, below 0 means worse. `null` when `relative_mae` is. |
| `ticker_metrics.<ticker>.arima.directional_accuracy` | float or null | Fraction of scored days on which ARIMA predicted the direction of the move from the previous close correctly (0 to 1). Days with no move are skipped; `null` if no day moved. The naive forecast predicts no move, so it has no directional accuracy. |
| `ticker_metrics.<ticker>.arima.n_directional` | int | Days scored for directional accuracy. |

### Notes

- **Comparing tickers.** RMSE and MAE are in each ticker's own price units, so a $1 error on a $500 stock and on a $30 stock are not alike. Compare tickers with `mape`, `relative_mae`, or `skill`, which are scale-free.
- **What "relative MAE" is.** It is the ratio of out-of-sample MAEs over the same scored days. It is related to, but not the same as, Hyndman's MASE, which scales by the naive forecast's in-sample error. Label it "relative MAE" (or "skill" for `1 − relative_mae`), not "MASE".
- **Small samples.** The scored days (`target_trading_days`) are a modest sample, so the metrics are noisy and can change from day to day. Consecutive forecasts are not independent, which makes the effective sample smaller still. Say so in user-facing text.
- **Comparing dates.** The number of scored days and the training window can differ between date folders, so their metrics can come from different sample sizes and methods. Compare each file's `target_trading_days` and `training_window_returns` before comparing metrics across dates.
- **Suggested wording near the metrics** (fill in `{n}` from `evaluation_window.target_trading_days`): "Accuracy over the last {n} trading days, comparing each forecast with the actual close. Relative MAE below 1 (skill above 0) means ARIMA's errors were smaller than the naive forecast's. RMSE and MAE are in dollars per share; MAPE is in percent and can be compared across tickers. {n} trading days is a modest sample, so treat these as indicative."

---

## Reader requirements (for the frontend)

1. **Ignore unknown things.** Unknown YAML keys, HDF5 groups or datasets, and CSV columns must be ignored without error. This is what makes additive changes safe.
2. **Do not hardcode** the ticker list, array lengths, row counts, or the training and scoring window lengths. Read the window lengths from `all_forecasts_metadata.yml` (`forecast_window.training_window_returns`, `evaluation_window.target_trading_days`) and the rows from `all_forecasts.csv`.
3. **Handle `NaN` and empty cells** as described above.
4. **Show data dates.** Display the selected date (equal to `date_updated.date_to`) so users can tell how current the data is.
5. **Fail soft.** If a file is missing or does not parse, show an explanatory message instead of crashing. Missing or partial data is expected while a run is mid-upload.
6. **Label units.** Daily versus annualized, and percent versus dollars, must be stated wherever a value is shown.
7. **Date selection.** Offer only *available* dates (see "Which dates are available"), newest first, with the newest selected by default. Load all files from the **same** date in both folders; never mix dates across files. Ignore keys whose segment after the folder is not a valid date, including stale objects directly under a folder from contract v8 and earlier.
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
- The "How the forecasts are produced" description matched the backend's defaults at the time (the pre-v24 method). The current method (v24+) is checked against the code and unit tests (`tests/test_arima_models.py`, `tests/test_ticker_predict.py`), not yet against real output.
- `.h5` values (v5 output, `h5dump -d`): `max_sharpe_ratio` = (12.4398, 34.3434) and `min_var_portfolio` = (6.89459, 12.8517) equal the YAML `risk` / `annualized_return` of the same run. `tangency_line/ys` starts at 4.05 (= `risk_free_rate`), is evenly spaced (a straight line), and passes through the max Sharpe ratio point with slope `sharpe_ratio`.
- The annualization formulas and the straight tangency line are covered by the backend's unit tests (`tests/test_annualization.py`).

**Not yet** verified against real data (derived from the code):

- That the date folder equals `date_updated.date_to`, and the upload order (`all_forecasts.csv`, `all_forecasts_metadata.yml`, `.h5`, then `optimization_metadata.yml`). The construction of the date part of the key and of local paths is covered by unit tests (`tests/test_layout.py`).
- The v8 NYSE calendar behavior on real data (covered by unit tests in `tests/test_date_manager.py`).
- `all_forecasts_metadata.yml` (v21) on real output. The metric formulas are covered by unit tests on hand-computed tables (`tests/test_metrics.py`), and the file's windows and upload order by `tests/test_pipeline.py`.
- `NaN` entries in `efficient_frontier/xs` (inferred from the code for infeasible target returns). Readers must handle `NaN` regardless.

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 29 | 2026-10-24 | Backend safeguard; no change to files, keys, columns, types, units, or the meaning of any value. The backend now checks that the max Sharpe ratio and minimum variance optimizations converge; if either does not, the run fails before uploading, as a missing risk-free rate already does (v14). Before v29 an unconverged optimization's weights could have been uploaded without warning. Efficient-frontier points the optimizer cannot reach are still `NaN`, as before. Added the "Converged optimizations only" note to `optimization_metadata.yml`. Frontend: no change required; replace its copy with this file. |
| 28 | 2026-10-24 | **Additive.** `optimization_metadata.yml` gains `min_var_portfolio.weights`: the minimum variance portfolio's weights in percent, one key per ticker, in the same form (and with the same floating-point noise) as `optimum_portfolio.weights`. The backend already computed these weights but did not upload them. No other key, file, value, unit, or upload order changes; the `.h5` file is unchanged. Added the field, a short example, the note that both weight maps follow the weight-noise rules, and suggested wording for showing both portfolios. Frontend: optionally show the minimum variance weights next to the max Sharpe ratio weights (applying the weight-noise rules to both); treat the key as optional, since dates before v28 lack it. |
| 27 | 2026-10-19 | Documentation only; no change to files, keys, columns, types, units, values, or reader behavior. Corrected three stale passages left from before v21 and v24: "Which dates are available" said `optimization_metadata.yml` is uploaded "last of all three" (there are four files per date; it is still last); "Verification status" listed the upload order without `all_forecasts_metadata.yml`; and it said the forecast description matches the backend's `run()` defaults, a function that no longer exists, so it now says that check applied to the pre-v24 method and the current method is checked by code and unit tests only. Frontend: no change required; replace its copy with this file. |
| 26 | 2026-10-18 | Documentation only; no change to files, keys, columns, types, units, or values. The training window length and the number of scored days are backend settings that can change between runs, so the contract no longer states their current values anywhere outside this changelog. Instead it points readers at the data: `forecast_window.training_window_returns` and `evaluation_window.target_trading_days` in `all_forecasts_metadata.yml`, and the rows of `all_forecasts.csv`. The metadata example shows window-dependent values as placeholders. Reader requirement 2 now also forbids hard-coding the window lengths. The suggested forecast-plot text now fills in the training window length (`{w}`, from `training_window_returns` + 1) instead of a fixed figure, like the metrics wording already does with `{n}`. Also fixes a leftover "same 20 days" in the relative MAE note. Frontend must: fill `{w}` in the forecast-plot text from the metadata (or leave the parenthesis out when the file is absent), and make sure no window length or row count is hard-coded. |
| 25 | 2026-10-18 | **Configuration change**; no key, column, file name, type, or unit is renamed, removed, or changed. The backtest now scores the last **60** trading days instead of 20 (set in the backend's `config/modeling_and_optimization.yml`). `all_forecasts.csv` normally has 61 rows (60 backtest + live) instead of 21, covering about three months, and the backend models the last 312 sessions instead of 272. In `all_forecasts_metadata.yml`, `evaluation_window.target_trading_days` is 60, `n_evaluated` and `n_directional` are normally 60, and the evaluation start dates move about three months back; the metrics are computed over 60 days, so they are less noisy than over 20 but still a modest sample. The training window (252 closes) and the optimization files are unchanged. The contract now states the number of scored days and the row count in terms of `target_trading_days` rather than as a fixed number, so a later change to the setting needs no frontend text change. Date folders written before v25 keep their 20-day metrics and 21 rows unless re-run with `--retrain-arima`. Frontend must: fill the number in the suggested metrics wording from `evaluation_window.target_trading_days` (it said "20 trading days" and "Twenty days" before); check that nothing assumes 21 rows or about one month of backtest on the x-axis. |
| 24 | 2026-10-17 | **Methodology change**; no key, column, file name, format, or unit is renamed, removed, or retyped. Each ARIMA forecast is now trained on the preceding trading year (252 closes, 251 daily log returns) instead of 20 returns, and orders are chosen by the **lowest BIC** of one fit per candidate on that whole window instead of by RMSE and BIC rank on rolling 10-return sub-windows; the winning model makes the forecast. Ties (exactly equal BICs) go to fewer parameters, then lower `p`. The backend now models the last 272 sessions; `all_forecasts.csv` still normally has 21 rows (20 backtest + live) and its columns are unchanged. In `all_forecasts_metadata.yml`, values change but keys do not: `forecast_window.training_window_returns` is 251, `training_window_start_date` is about a year before the date folder, `models.arima.order_selection` and `order_selection_tie_break` describe the new rule, and `order_selection_subwindow_returns` equals the training window (251). The optimization files are unchanged: they use exactly the last 252 sessions (previously all downloaded rows, which could be one more session, so `date_updated.date_from` can move one session later). Date folders uploaded before v24 keep their 20-return forecasts unless re-run with `--retrain-arima`, so comparisons across dates mix the two methods; `training_window_returns` tells them apart. Frontend must: update the forecast-plot text to say the model is refit on the previous trading year (see "Suggested user-facing wording"); show any order-selection text from the metadata rather than hard-coding the old rule. |
| 23 | 2026-10-16 | Methodology fix plus one **additive** key; no key, column, format, or unit is renamed, removed, or changed. ARIMA orders are chosen by the lowest average of the RMSE rank and the BIC rank, and with three candidates equal averages are common. Ties used to go to whichever candidate came first in an internal table whose order depended on which parallel job finished first, so the chosen order (and `T_arima`) could differ between runs on the same data. Ties now go to the lower one-step-ahead RMSE, then fewer parameters (`p + q`), then the lower `p`, so runs are reproducible. `all_forecasts_metadata.yml` gains `models.arima.order_selection_tie_break` describing this rule. `T_arima` and the order columns change only on rows where a tie was broken. Date folders uploaded before v23 keep their forecasts unless re-run with `--retrain-arima`, and their orders on tied rows reflect the old, run-dependent tie-break. Frontend: no change required; displaying the new key is optional, and it is absent in older files. |
| 22 | 2026-10-15 | Documentation only; no change to files, keys, or values. The investment-advice disclaimer is consolidated: the frontend shows one fixed wording ("Not investment advice. Every value shown is a backtest or model output computed from historical prices, and none of it predicts future results or recommends any investment.") once, at the bottom of the UI, and it is no longer repeated in the suggested plot, tooltip, and metrics wording. Frontend: keep the single footer disclaimer with this wording and drop any repeats. |
| 21 | 2026-10-15 | **Additive.** New file `all_forecasts_metadata.yml` next to `all_forecasts.csv` in every date folder: the live forecast's training window and forecast date, and, per ticker, RMSE, MAE and MAPE for ARIMA and naive forecasts over the last 20 trading days with an actual close, `relative_mae` (ARIMA MAE ÷ naive MAE), `skill` (1 − relative MAE), and ARIMA directional accuracy, plus units and a description of both models. All keys are lowercase snake_case, **including ticker keys** (`xbi`, not `XBI`). Upload order is now forecasts CSV, forecasts metadata, `.h5`, optimization metadata (still last); the availability rule is unchanged. To score 20 days, the ARIMA input grows from 40 to 41 trading sessions, so `all_forecasts.csv` normally has 21 rows (20 backtest + live) instead of 20; columns are unchanged. Frontend must: read the metrics from this file (match tickers case-insensitively) instead of computing RMSE itself; treat the file as optional for older dates; use the suggested wording; label `relative_mae` as "relative MAE" or show `skill`, not "MASE". |
| 20 | 2026-10-15 | Bug fix, non-breaking for complete data (no change to keys, column names, or units; with no missing closes the output is unchanged). Two gaps in the price data were mishandled. (1) If a whole session was missing, two rows got the same `pred_date` (the day of the gap) and the session after it had no row, violating "rows are unique by `pred_date`"; training windows are now built on the dates actually present, so `pred_date` is always unique, and the session after a whole missing session has no row. (2) If one ticker's close was missing, its next `T_arima` came out empty by accident (the back-transform multiplied by the missing price); it is now deliberately empty, as are that row's order columns (`T_arima_p`, `T_arima_d`, `T_arima_q`), and `T_naive` still uses the most recent observed close. `T_arima` and the order columns are now documented as possibly empty. Added "Missing data" under "How the forecasts are produced". Frontend: handle empty `T_arima` and empty (or float-typed) order columns on such rows. |
| 19 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains a float column `T_naive` per ticker: the naive (random walk) baseline forecast, equal to the most recent observed close before `pred_date` (normally the previous trading day's close), computed from the same data the row's ARIMA model saw. Present on every row. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_d`, `T_arima_q`, `T_naive`, `T` (order is still not part of the contract). Added the `_naive` suffix rule and a naive-forecast sentence to the suggested wording. No error metrics are uploaded. Frontend: treat `T_naive` as optional (absent before v19); optionally plot it and show the wording. |
| 18 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains an integer column `T_arima_d` per ticker, always `1`, so that `(T_arima_p, T_arima_d, T_arima_q)` is the model's full `(p, d, q)` on log prices: ARIMA(p, 0, q) on log returns is ARIMA(p, 1, q) on log prices. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_d`, `T_arima_q`, `T` (order is still not part of the contract). Added "Suggested user-facing wording" for the forecast tooltip and the text near the plot. Frontend: the exact-suffix rule now includes `_arima_d`; the column is optional (absent before v18); show the wording if displaying model orders. |
| 17 | 2026-10-15 | Methodology fix, non-breaking (no change to keys, columns, formats, or units). Two corrections to how ARIMA orders are chosen: (1) when a model fit failed on a rolling sub-window, the previous sub-window's model was reused to score it (or, on the first sub-window, the run crashed); failed sub-windows are now excluded from the order's RMSE and BIC, and an order whose fits all fail ranks last. (2) The leading missing value left by differencing is dropped, so every sub-window holds 10 actual log returns instead of the first holding 9. Selected orders and `T_arima` values can differ slightly from earlier runs; already-uploaded dates keep their forecasts unless re-run with `--retrain-arima`. Frontend: no change required. |
| 16 | 2026-10-15 | **Additive** (no existing column renamed, removed, or changed). `all_forecasts.csv` gains two integer columns per ticker, `T_arima_p` and `T_arima_q`: the order of the ARIMA(p, 0, q) model on log returns that made each row's forecast, present on every row. Each ticker's columns are now ordered `T_arima`, `T_arima_p`, `T_arima_q`, `T` (order is still not part of the contract). Documented the exact-suffix rule for telling forecast, order, and actual columns apart. Frontend must: check that it identifies forecast columns by the exact `_arima` suffix (not by containing `_arima`) and actual columns by exact ticker match; otherwise the new columns could be misread as forecasts. Displaying the orders is optional. Older date folders lack these columns, so readers must treat them as optional. |
| 15 | 2026-10-15 | Methodology fix, non-breaking (no change to keys, columns, formats, or units; `pred_date` was already the next NYSE trading day). ARIMA training windows were resampled to pandas weekdays (`"B"`), which inserted NYSE holidays such as Presidents Day as missing observations inside the window. Windows now contain only NYSE trading sessions, so a model time step is always one session. `T_arima` values for windows that span an NYSE holiday can differ slightly from earlier runs; date folders already uploaded keep their old forecasts unless the backend is re-run with `--retrain-arima`. Frontend: no change required. |
| 14 | 2026-10-14 | Bug fix, non-breaking (no change to keys, formats, or units). FRED leaves `DTB3` empty on bond-market holidays (e.g. Columbus Day); the backend used the latest row even when it was empty, which could make `risk_free_rate`, `daily_risk_free_rate`, `sharpe_ratio`, and the tangency line `NaN`. It now uses the most recent non-missing value, so `risk_free_rate_date` can be earlier than the last FRED row, and a run with no valid rate fails before uploading. Clarified the two `risk_free_rate` field descriptions. Frontend: no change required. |
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
