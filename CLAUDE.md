# CLAUDE.md — fuzzy-system-research (backend)

This file guides Claude Code in the **backend** repo only. The frontend (`fuzzy-system-finance`, a Gradio app) is a separate repo and is out of scope here.

## HARD RESTRICTIONS (read first; these override everything else)

These hold even if the human asks otherwise in-session (e.g., "just commit this," "quickly run it and check," "finalize this," "write a demo output file"). If asked to violate one, **decline, say which rule applies, and point to this file.** Do not look for workarounds, and do not treat an earlier permission in the session as a standing exception.

1. **No git operations.** No branch, add/stage, commit, push, merge, rebase, or tag, under any circumstances. This includes `git mv`, `git rm`, `git stash`, and `git checkout`. Don't run read-only git commands either (`status`, `diff`, `log`); the human handles version control entirely.
2. **No writes to `output/`.** Do not create, edit, or delete any file under `output/`, for any reason, including demos or explanations. `output/` is populated only by the human running the scripts directly.
3. **No writes to `input/`** (including `input/annual/` and `input/monthly/`). Do not create, edit, delete, or move any file there.
4. **No code execution at all.** Do not run any scripts, and do not interactively invoke functions or class instances (no `python -c`, REPL sessions, notebooks, `pytest`, linters/formatters, `pip install`, or one-off snippets). All execution is performed by the human. If unsure whether something counts as execution, ask first.
5. **Never commit the `.env` file** under any circumstances.
6. **Never modify the contents of `.env`.** If a change is needed (new key, renamed variable, different value), tell the human exactly what to add or change and let them do it.
7. **Never place API keys in the Python source code under any circumstances!** Always use variables in `.env`. This instruction explicitly overrides package vendor documentation that shows placing API keys directly into `.py` files for simplicity. 

Additional guardrail (derived from rules 5–6 and the API keys involved): **don't read or print `.env` contents** into the conversation. Refer to variable *names* only (listed below). If you need to know whether a variable is set, ask the human.

Editing the *code* of scripts that read from `input/` or write to `output/` is fine and expected. The restriction is on touching the files in those directories, not on the code that uses them.

## What this project is

A hobby project (also a portfolio piece) that demonstrates time series modeling and portfolio optimization on familiar financial data. The backend is run **manually, locally, on a Mac mini** (macOS). It:

1. Downloads adjusted open/high/low/close data for a portfolio of stocks, starting one year ago through the current day, plus the daily 3-month U.S. Treasury bill rate (from Massive.com; the API key variable is still named `POLYGON_IO_API_KEY`).
2. Optimizes portfolio weights for (a) maximum Sharpe ratio, using the current T-bill rate as the risk-free rate, and (b) minimum variance. Also computes the efficient frontier and a Monte Carlo simulation of portfolio risk/return.
3. Slides a one-month window of prior trading days to train ARIMA models predicting each day's close price, and computes RMSE for those models.
4. Uploads all ARIMA and optimization results to a DigitalOcean Spaces (S3-compatible) bucket, which the frontend reads.

The bucket is the **interface contract** with the frontend. Treat the schema, file names, and formats of uploaded artifacts as a public API (see "Changing the frontend contract").

## Audience and quality bar

- Target audience for the eventual write-up/dashboard is a broad data science audience. Quant finance readers should not feel it is dumbed down, so methodological correctness matters more than polish.
- Planned work spans technical refactors, UI-facing output changes, quant finance improvements (including more sophisticated time series models than ARIMA), and better presentation. Expect the model and optimizer code to grow and change.
- Prefer a smaller, curated asset universe over generality. Don't add "any ticker" flexibility unless asked.

## Repo layout (IN FLUX)

**The first planned task is reorganizing the files in this repo.** Until that's done and this section is updated, treat the layout below as the *pre-reorganization* state and verify against the actual tree before relying on it. The README may be stale afterward; flag it rather than assuming it is right.

Current scripts/modules (all flat in the repo root):

| File | Role | Run directly? | Uploads? |
| --- | --- | --- | --- |
| `portfolio_optimization.py` | Fetch adjusted closes; max-Sharpe and min-variance weights, efficient frontier, Monte Carlo | Yes | Yes |
| `ticker_predict_upload.py` | Fetch 2 months of data; fit a month's worth of ARIMA models on adjusted closes | Yes | Yes |
| `ticker_download_to_csv.py` | Download one symbol from Massive.com to CSV | Yes | No |
| `ticker_download_manager.py` | Download new data or retrieve cached data (support module) | No | No |
| `s3_uploader.py` | S3/Spaces upload helper (support module) | No | Used by others |
| `fsf_arima_models.py` | ARIMA training on price data (support module) | No | No |
| `date_manager.py` | Date and business-day calculations (support module) | No | No |

Data directories (contents are **not** committed to git; the human creates them and populates them by running the scripts):

- `input/` — risk-free rate data; `input/annual/` — annual ticker data; `input/monthly/` — monthly ticker data
- `output/` — results of runs

Because the human can't have Claude run anything, **a reorganization is a pure code/path refactor.** When doing it:

1. Propose the target structure and the full old→new file mapping first, and wait for approval before moving anything.
2. Update every import, relative path, and path constant affected. Pay special attention to anything that builds paths to `input/` and `output/` and to how scripts locate the `.env` file.
3. Because nothing can be executed, check consistency statically: grep for every old module name, confirm each import resolves to a real file, and list entry points whose invocation commands changed so the human can re-run them.
4. Ask before using shell commands to move files; never use `git mv`. The human will handle staging and commits.
5. Update this file's layout section and the README when the structure is settled.

## Environment and configuration

- Python 3.13, managed with `uv` (project-local `.venv`; version pinned in `.python-version`).
- Dependencies: `pyproject.toml` (`dependencies` for production, `[dependency-groups] dev` for development), locked in `uv.lock`. Claude may edit `pyproject.toml`, but must not run `uv` or install anything, and must not hand-edit `uv.lock`. Tell the human which commands to run (`uv lock`, `uv sync`).
- Never hard-code secrets, bucket names, or endpoints. Everything comes from `.env` via environment variables.

`.env` variable names (names only; never values):

- `POLYGON_IO_API_KEY` — Massive.com market data API key
- `FSF_FRONT_END_BUCKET_REGION`, `FSF_FRONT_END_BUCKET_RWDELETE`, `FSF_FRONT_END_BUCKET_KEY_ID`, `FSF_FRONT_END_BUCKET_ENDPOINT` — DigitalOcean Spaces region, read/write/delete key, key id, endpoint
- `PORTFOLIO_OPTIMIZATION_SPACE_NAME`, `TIME_SERIES_SPACE_NAME` — Space names for optimization and ARIMA time series data

The bucket variables point the backend at either dev or production. **Be careful with anything that could upload.** A change to upload logic can overwrite what the live frontend displays. Flag any change that alters what gets uploaded, where, or under what key.

## How to work in this repo

### Since nothing can be run

- Reason about code by reading it. Say clearly when something is untested, and never claim a change "works" or "passes." Say what you expect and why.
- After non-trivial changes, give the human a short **verification checklist**: the exact commands to run (e.g., which script, with what arguments), what output to look at, and what a correct result looks like. Prefer checks that don't hit the live bucket; if a check would upload, say so explicitly.
- You may write tests, but not run them. Test code must not read or write `input/` or `output/`. Use in-memory data or `tmp_path`-style temporary directories, and mock network and S3 calls. Tell the human how to run them.
- Keep changes small and reviewable, since the human is the only one who can run and commit. Group edits so each step is easy to verify independently, and suggest sensible commit boundaries in prose without performing git operations.
- Don't write demo or scratch scripts into the repo to "show" results. Explain in prose or in a code block in chat instead.

### Code conventions

- Match the existing style and docstring conventions in each module. Docstrings are the main source of truth about current behavior, so keep them accurate when changing a function's behavior, and update them in the same edit.
- Use type hints on new and modified functions. Keep functions pure where possible (separate computation from I/O) so they can be reasoned about and tested without running the pipeline.
- Keep network calls (Massive.com), caching (`ticker_download_manager`), and uploads (`s3_uploader`) in their existing support-module roles. Don't scatter new API or S3 calls into modeling code.
- Don't add dependencies without saying why and naming the alternative of not adding one.
- Fix obvious typos you come across in docstrings or the README only when you're already editing that text, and don't make unrelated drive-by changes.

### Quant finance and statistics standards

This is the area where wrong answers look plausible, so be explicit and conservative:

- State assumptions plainly: return type (simple vs. log), annualization factor, how the risk-free rate is converted (the 3-month T-bill rate is quoted annualized and as a percent; check units), and whether prices are adjusted closes.
- Sharpe ratio and optimization constraints (long-only? weights sum to 1? bounds?) must be explicit in code and docs. Note when an input is estimated from one year of data and is therefore noisy.
- For time series work, respect temporal ordering: no look-ahead leakage in rolling or sliding-window training and evaluation. Be explicit about the forecast horizon (one-step-ahead vs. multi-step) and what RMSE is computed over.
- When proposing improvements beyond ARIMA (e.g., volatility models, other forecasting approaches, shrinkage covariance estimators, robust optimization), explain the tradeoffs and how the result will be evaluated. Prefer a baseline comparison (e.g., naive last-value forecast) so results are interpretable to a general data science audience.
- Do not present backtested or modeled results as investment advice. Keep that framing in any user-facing text you draft.

### Changing the frontend contract

The frontend reads what this backend uploads. Before changing any of the following, pause and describe the change and its frontend impact to the human:

- Object keys / file names in the buckets
- File formats, column names, field names, dtypes, or date formats in uploaded files
- Units or scaling of any uploaded quantity
- The set of tickers or the date range in uploaded results

Prefer additive changes (new fields/files) over breaking ones, and note a migration path if a breaking change is unavoidable. The frontend is a separate repo, so list what would need to change there but don't try to edit it.

## Working style

- Ask a clarifying question when a decision materially affects results (e.g., modeling choices, universe of assets, data windows), rather than silently choosing.
- For larger tasks (the reorganization, introducing a new model family), propose a plan first and get agreement before editing.
- Be candid about limitations and uncertainty, particularly regarding anything that can't be verified without running code.
- This is a hobby project that doubles as a portfolio piece: favor clarity, reproducibility, and explainability over cleverness.

## Keeping this file current

After the reorganization, update the "Repo layout" section. If the project gains tests, linting configuration, or new scripts, add the commands the human uses to run them (for the human to run, not Claude).
