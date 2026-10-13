"""
Shared, deterministic synthetic price data for tests. In-memory only.
"""

import numpy as np
import pandas as pd
import pytest

SYNTHETIC_TICKERS = ["XBI", "QQQ", "GLD"]  # Deliberately not alphabetical.
SYNTHETIC_SESSIONS = 60


def make_long_prices(
    tickers: list[str] = SYNTHETIC_TICKERS,
    n_sessions: int = SYNTHETIC_SESSIONS,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Long-format adjusted closes shaped like a cached download read back with
    pandas.read_csv: columns datetime, ticker, close (plus open etc. omitted).
    Prices follow seeded geometric random walks with different drifts and
    volatilities so the optimization problem is well posed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-07-01", periods=n_sessions)
    rows = []
    for i, ticker in enumerate(tickers):
        daily_log_returns = rng.normal(0.0004 * (i + 1), 0.01 * (i + 1), n_sessions)
        closes = 100 * np.exp(np.cumsum(daily_log_returns))
        for date, close in zip(dates, closes):
            rows.append(
                {
                    "datetime": date.strftime("%Y-%m-%d 23:59:59"),
                    "ticker": ticker,
                    "close": close,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def long_prices() -> pd.DataFrame:
    return make_long_prices()
