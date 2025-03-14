import os
import json
import requests
import time
from dotenv import load_dotenv


def fetch_all_results(base_url, params, delay=5):
    """
    Fetches paginated data from MarketStack API
    and returns a concatenated list of all results.

    Parameters
    ----------
    base_url : str
        The base URL of the request

    params : dict
        Dictionary of parameters for the request

    delay : int, optional
        Defaults to 5. Delay between API calls in seconds.

    Returns
    -------
    list
        List with all results concatenated.
    """
    all_results = []
    offset = 0
    limit = params["limit"]

    while True:
        params["offset"] = offset
        print(f"Fetching offset {offset}")
        time.sleep(delay)
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            break

        data = response.json()
        results = data.get("data", [])
        pagination = data.get("pagination", {})
        print(pagination)

        all_results.extend(results)

        total = pagination.get("total", 0)

        # If we've fetched all available records, stop
        if offset + limit >= total:
            break

        # Move to the next offset
        offset += limit

        # Delay for the next API call
        time.sleep(delay)

    return all_results


if __name__ == "__main__":
    # Load the .env file
    load_dotenv()

    date_from = "2015-04-01"
    date_to = "2025-03-12"

    # Paremters common to all API calls
    marketstack_api_key = os.getenv("MARKETSTACK_API_KEY")
    base_url = "http://api.marketstack.com/v2/eod"

    # Setup the API call for AAPL history
    # params = {
    #     "access_key": marketstack_api_key,
    #     "symbols": "AAPL",
    #     "date_from": date_from,
    #     "date_to": date_to,
    #     "limit": 3000,
    # }

    # # Fetch the dadta and report the number of records
    # all_data = fetch_all_results(base_url, params)
    # print(f"Fetched {len(all_data)} records")

    # # Write the AAPL history
    # aapl_history_filename = os.path.join("output", "AAPL_history.json")
    # with open(aapl_history_filename, "w") as file:
    #     json.dump(all_data, file, indent=4, sort_keys=True)

    # Scrape the other 5
    symbols = ["GOOG", "AMZN", "MSFT", "NVDA", "TSLA"]
    for symbol in symbols:
        params = {
            "access_key": marketstack_api_key,
            "symbols": symbol,
            "date_from": date_from,
            "date_to": date_to,
            "limit": 3000,
        }
        all_data = fetch_all_results(base_url, params)
        print(f"Fetched {len(all_data)} records")
        history_filename = os.path.join("output", f"{symbol}_history.json")
        with open(history_filename, "w") as file:
            json.dump(all_data, file, indent=4, sort_keys=True)
