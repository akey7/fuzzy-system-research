"""
Run the whole backend in one command: download prices (or use today's
cached download), fit walk-forward ARIMA forecasts (or reuse the cached
forecasts for the data's as-of date), optimize the portfolio, and upload all
results for the frontend.

Run from the repository root:

    uv run python scripts/run_optimization_and_time_series.py [--no-upload]
        [--retrain-arima] [--refresh-data]
"""

import argparse
from dotenv import load_dotenv
from fuzzy_system_research.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """
    Parse the command line options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download prices, fit ARIMA forecasts, optimize the portfolio, and "
            "upload the results under a YYYY-MM-DD/ prefix (the data as-of date)."
        )
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Write the output/ files but do not upload anything.",
    )
    parser.add_argument(
        "--retrain-arima",
        action="store_true",
        help="Retrain ARIMA models even if forecasts for the as-of date are cached.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download prices again even if today's download is cached.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Load .env and run the pipeline with the command line options.
    """
    args = parse_args()
    load_dotenv()
    result = run_pipeline(
        upload=not args.no_upload,
        retrain_arima=args.retrain_arima,
        refresh_data=args.refresh_data,
    )
    print(f"Done: {result.as_of} (ARIMA {'trained' if result.trained_arima else 'cached'})")
    for key in result.uploaded:
        print(f"  uploaded {key}")


if __name__ == "__main__":
    main()
