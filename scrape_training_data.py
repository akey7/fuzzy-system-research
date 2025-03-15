import os
from dotenv import load_dotenv
from polygon import RESTClient
import pandas as pd
import time


def main():
    # Load the .env file
    load_dotenv()

    # Paremters common to all API calls
    polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")

    # Run example code
    client = RESTClient(polygon_io_api_key)

    # Date range
    date_from = "2020-04-01"
    date_to = "2025-03-13"

    # List Aggregates (Bars)
    tickers = ["AMZN", "GOOG", "MSFT", "AAPL", "NVDA", "TSLA"]
    rows = []
    for ticker in tickers:
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=date_from,
            to=date_to,
            adjusted="true",
        ):
            rows.append({
                "timestamp": a.timestamp,
                "ticker": ticker,
                "open": a.open,
                "high": a.high,
                "low": a.low,
                "close": a.close,
                "volume": a.volume,
                "vwap": a.vwap,
                "transactions": a.transactions,
            })
        print(f"{ticker}. Acquired {len(rows)} so far. Sleeping 5 seconds...")
        time.sleep(5)

    # Return DataFrame
    ticker_histories = pd.DataFrame(rows)
    ticker_histories_filename = os.path.join("input", "ticker_histories.csv")
    ticker_histories.to_csv(ticker_histories_filename, index=False)
    print(f"Wrote {ticker_histories_filename}")


if __name__ == "__main__":
    main()
