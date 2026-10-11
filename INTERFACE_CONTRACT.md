# Interface Contract: fuzzy-system-research (backend) ↔ fuzzy-system-finance (frontend)

| | |
| --- | --- |
| **Contract version** | 2 |
| **Last updated** | 2026-10-11 |
| **Source of truth** | `fuzzy-system-research` (backend). The copy in `fuzzy-system-finance` is a mirror. |

This file is stored **byte-for-byte identically** in both repos. It describes the files the backend uploads to S3-compatible buckets (DigitalOcean Spaces, accessed with `boto3`) and the frontend reads. Treat it as a public API.

## Rules for editing this file (humans and AI agents)

1. **Edit the backend copy first**, then copy it unchanged into the frontend repo. Never edit only one copy, and never let the copies diverge.
2. **Bump the contract version and add a changelog entry** for every change, including clarifications that change meaning.
3. **Breaking changes** (anything in the list below) need explicit human approval. Describe the frontend impact before making the change, and prefer an additive alternative with a migration path.
   - Renaming or removing a bucket key, file, YAML key, HDF5 group/dataset, or CSV column
   - Changing a data type, units, scaling, or date format
   - Changing the shape of an array or the meaning of a value
   - Changing the set of tickers or the covered date range in a way a reader could not detect from the data itself
4. **Additive changes** (new optional keys, new columns, new files) are allowed without breaking the contract, **because readers must ignore anything they do not recognize** (see "Reader requirements").
5. The backend agent must not edit the frontend repo, and the frontend agent must not edit the backend repo. Each lists what the other must change.

## Status of this document

Version 1 was written from the backend code. Version 2 was checked against real output files from the backend (a real `optimization_metadata.yml`, the `h5dump -H` structure of the `.h5` file, and the header and final row of `all_forecasts.csv`). What was and was not verified is listed under "Verification status" at the end.

## Where the data lives

| Environment variable (backend `.env`) | Bucket (Space) contents |
| --- | --- |
| `PORTFOLIO_OPTIMIZATION_SPACE_NAME` | `optimization_metadata.yml`, `portfolio_optimization_plot_data.h5` |
| `TIME_SERIES_SPACE_NAME` | `all_forecasts.csv` |

Both buckets are on the same DigitalOcean endpoint. Dev and production are selected by the `.env` values the backend and the frontend are each configured with; the key names below are identical in both.

General behavior of every file:

- **Latest only.** Each backend run overwrites the same key. There are no dated keys and no history in the bucket.
- **Non-atomic.** The two optimization files are uploaded one after the other (`.h5` first, then `.yml`), so for a few seconds the files can come from different runs. The metadata `date_updated` and the `.h5` data are from the same run once both uploads finish.
- **Manual cadence.** The backend is run by hand, so there is no guaranteed update schedule. The frontend should display the data dates (see below) so a stale dataset is visible, rather than assuming data is current.
- **Content type and ACLs.** The backend sets none. Do not rely on the `Content-Type` of an object; decide how to parse from the key name.
- **Not investment advice.** All values are backtested or modeled outputs of a hobby project. Any user-facing text must say so.

## Shared conventions

- **Tickers** are the strings used by Massive.com, e.g. `NVDA`, `QQQ`. At the time of writing the set is `ETHA, FXE, GLD, IBIT, NVDA, QQQ, VOO, VXUS, XBI, XLE`, but **the frontend must not hardcode it**. Read the ticker list from `optimization_metadata.yml` (`tickers`) or from the `all_forecasts.csv` column names. The set can change.
- **Prices** are daily close prices in US dollars as returned by Massive.com with `adjusted=true`, i.e. adjusted by Massive.com's rules. Treat them as "adjusted close" and do not assume dividend adjustment.
- **Returns** are **simple** (not log) daily returns, expressed in **percent** (1.5 means 1.5 %), computed from consecutive adjusted closes.
- **Annualization** uses 252 trading days per year.
- **Risk** means the standard deviation (σ) of daily returns, in percent per day, unless a field says it is annualized.
- **Dates** are US trading days. Date-only fields are strings in `YYYY-MM-DD` format.

---

## `optimization_metadata.yml`

**Bucket:** `PORTFOLIO_OPTIMIZATION_SPACE_NAME`. **Format:** YAML (block style, keys sorted alphabetically), UTF-8. Parse it with a safe YAML loader. Date-like values are written as **quoted strings**, so loaders return strings, not date objects.

### Example

A real file produced by the backend (run for data through 2026-10-09):

```yaml
date_updated:
  date_from: '2025-10-07'
  date_to: '2026-10-09'
optimum_portfolio:
  annualized_return: 40.945115872014924
  risk: 12.439838563729614
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
| `risk_free_rate.daily_risk_free_rate` | float | `risk_free_rate / 252`. **Percent per day**, same units as the daily returns used in the optimization. |
| `risk_free_rate.risk_free_rate_date` | string | Observation date of the rate, formatted as a timestamp string such as `2026-10-08 00:00:00`. Parse the first 10 characters as a date. |
| `optimum_portfolio.weights` | map ticker → float | Weights of the maximum Sharpe ratio portfolio, **in percent** (a weight of 22.4 means 22.4 %). Contains one key per ticker. |
| `optimum_portfolio.annualized_return` | float | Expected return of that portfolio, **annualized, in percent**: `((1 + r_daily/100)^252 − 1) × 100`, where `r_daily` is the portfolio's mean daily return in percent. |
| `optimum_portfolio.risk` | float | Standard deviation of that portfolio's daily returns, **annualized** (daily σ × √252), **in percent**. |

### Notes

- **Weight noise.** Treat any weight with absolute value below `1e-6` (percent) as zero, and round for display. Never compare a weight to an exact value such as `0` or `40`, and never assume a "zero" weight is written as exactly `0.0`. Do not assume the weights sum to exactly 100; they sum to 100 only within the optimizer's tolerance.
- The optimization is long-only, weights sum to 100 (within optimizer tolerance), and each weight is currently capped at `4 / (number of tickers)` of the portfolio (40 % with 10 tickers). Those constraints are the optimizer's current settings, not a contractual guarantee. Do not hard-code them in the frontend.
- Inputs (mean returns and covariance) are estimated from roughly one year of daily data, so the optimum is noisy and sensitive to the window. Say so in user-facing text.
- Only the **max Sharpe ratio** portfolio's weights are in this file. The minimum variance portfolio appears only as a point in the `.h5` file (risk and return), without weights.

---

## `portfolio_optimization_plot_data.h5`

**Bucket:** `PORTFOLIO_OPTIMIZATION_SPACE_NAME`. **Format:** HDF5 (written with `h5py`). The file has five top-level groups. Each group has exactly two datasets named `xs` and `ys`: one-dimensional float64 arrays of equal length within the group. `xs` is the horizontal axis (risk), and `ys` is the vertical axis (return). All values are **daily, in percent**, and the file contains no dates and no ticker names (use `optimization_metadata.yml` for those).

### Layout

```
/efficient_frontier/xs        float64, shape (100,)    daily risk σ (%) of the minimum-variance portfolio for each target return
/efficient_frontier/ys        float64, shape (100,)    the 100 target daily returns (%), evenly spaced
/tangency_line/xs             float64, shape (100,)    daily risk σ (%), evenly spaced from 0
/tangency_line/ys             float64, shape (100,)    daily return (%): daily_risk_free_rate + max_sharpe_ratio × xs
/simulated_portfolios/xs      float64, shape (10000,)  daily risk σ (%) of random long-only portfolios (Monte Carlo)
/simulated_portfolios/ys      float64, shape (10000,)  daily return (%) of those portfolios
/max_sharpe_ratio/xs          float64, shape (1,)      daily risk σ (%) of the max Sharpe ratio portfolio
/max_sharpe_ratio/ys          float64, shape (1,)      its daily return (%)
/min_var_portfolio/xs         float64, shape (1,)      daily risk σ (%) of the minimum variance portfolio
/min_var_portfolio/ys         float64, shape (1,)      its daily return (%)
```

### Group meanings

| Group | What to draw |
| --- | --- |
| `efficient_frontier` | A line. Points run from the minimum variance portfolio's return up to the highest simulated return. |
| `tangency_line` | A straight line starting at (0, risk-free rate) with slope equal to the max Sharpe ratio. Also called the capital market line. |
| `simulated_portfolios` | A scatter cloud (about 10,000 points) of random portfolios, to show the feasible region. |
| `max_sharpe_ratio` | A single marker: the max Sharpe ratio (tangency) portfolio. |
| `min_var_portfolio` | A single marker: the minimum variance portfolio. |

### Notes

- **Units differ from the YAML file.** This file is **daily**, while `optimization_metadata.yml` reports **annualized** risk and return for the optimum portfolio. Do not plot one against the other without converting.
- **`NaN` values.** An `efficient_frontier/xs` entry is `NaN` when the optimizer found no feasible portfolio for that target return. Readers must drop or break the line at `NaN` and must not treat it as an error.
- **Array lengths** (100 and 10,000) are current settings. Read the shape from the file and do not hard-code them.
- Axis labels used by the backend's own plot are "Daily Risk (σ)" and "Daily Returns (%)".
- The backend also writes `portfolio_optimization_plot.png` locally. It is **not uploaded** and is not part of this contract.

---

## `all_forecasts.csv`

**Bucket:** `TIME_SERIES_SPACE_NAME`. **Format:** CSV with a header row, UTF-8, comma-separated, written by `pandas.DataFrame.to_csv(index=True)` (the index is the first column). Read it with the first column parsed as dates, e.g. `pandas.read_csv(..., index_col="pred_date", parse_dates=True)`.

### Shape

A wide table with one row per forecast date and, for every ticker `T`, two columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `pred_date` (first column, the index) | timestamp string, e.g. `2026-10-09 17:00:00` | The business day being forecast. The time of day (`17:00:00`) is a fixed label with no meaning; use only the date. |
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
- **Last row:** the final row is the first business day after the latest available data. Its `T` values are **empty** because that day has not happened yet. Treat that row as the live forecast and the rows above it as the backtest.
- **Rows** are unique by `pred_date` and sorted ascending. The number of rows is about one month of business days and is not fixed.

### Not included

The file does **not** contain RMSE, the selected `(p, q)` orders, confidence intervals, or a naive-baseline forecast. If the frontend needs an error metric, compute it from the `T` and `T_arima` columns on rows where `T` is not empty, and state which metric and window it uses. If any of these should be uploaded, it needs a contract change (an additive one is preferred).

---

## Reader requirements (for the frontend)

1. **Ignore unknown things.** Unknown YAML keys, HDF5 groups or datasets, and CSV columns must be ignored without error. This is what makes additive changes safe.
2. **Do not hardcode** the ticker list, array lengths, or row counts.
3. **Handle `NaN` and empty cells** as described above.
4. **Show data dates.** Display `date_updated.date_to` (and the last non-empty actual date in `all_forecasts.csv`) so staleness is visible.
5. **Fail soft.** If a file is missing or does not parse, show an explanatory message instead of crashing. Missing or partial data is expected while a run is mid-upload.
6. **Label units.** Daily versus annualized, and percent versus dollars, must be stated wherever a value is shown.

## Verification status

Checked against real backend output (2026-10-11):

- `optimization_metadata.yml`: date and timestamp string formats (all quoted), key names, nesting, units, and the floating-point noise in weights.
- `portfolio_optimization_plot_data.h5`: group names, dataset names, dtype (`H5T_IEEE_F64LE`), and shapes (100, 1, 1, 10000, 100), from `h5dump -H`.
- `all_forecasts.csv`: first column `pred_date`, the `T_arima` / `T` naming, and empty actuals in the final row.
- The "How the forecasts are produced" description matches the backend's `run()` defaults.

Still **not** verified against real data:

- `NaN` entries in `efficient_frontier/xs`. This is inferred from the code (infeasible target returns); `h5dump -H` shows structure only, not values. Readers must handle `NaN` regardless.
- The numeric values and units inside the `.h5` datasets (daily, percent). These come from the code.

## Changelog

| Version | Date | Change |
| --- | --- | --- |
| 2 | 2026-10-11 | Replaced the illustrative YAML example with a real file. Documented floating-point noise in weights (scientific notation, `~1e-14` for zero, `39.99999999999994` at the cap) and reader handling. Recorded verification status. No change to keys, columns, shapes, or units. |
| 1 | 2026-10-11 | Initial contract, derived from the backend code. |
