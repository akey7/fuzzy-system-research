import os
from dotenv import load_dotenv
from polygon import RESTClient


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    # Paremters common to all API calls
    polygon_io_api_key = os.getenv("POLYGON_IO_API_KEY")

    # Run example code
    client = RESTClient(polygon_io_api_key)

    # List Aggregates (Bars)
    ticker = "AMZN"
    aggs = []
    for a in client.list_aggs(ticker=ticker, multiplier=1, timespan="day", from_="2022-06-01", to="2022-06-10", limit=100):
        aggs.append(a)

    for agg in aggs:
        print(agg)
