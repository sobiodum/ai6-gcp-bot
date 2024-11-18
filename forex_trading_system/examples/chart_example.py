# File: examples/chart_example.py
# Path: forex_trading_system/examples/chart_example.py

from data_management.dataset_manager import DatasetManager
from visualization.chart_manager import ChartManager
from datetime import datetime, timedelta
import pytz


def main():
    # Initialize managers
    dataset_manager = DatasetManager()
    chart_manager = ChartManager()

    # Load data for EUR_USD
    ticker = "EUR_USD"

    # Example: Load last 30 days of data
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=30)

    print(f"Loading and updating data for {ticker}...")
    df = dataset_manager.load_and_update_dataset(
        currency_pair=ticker,
        timeframe="15min",  # We'll use 15-min timeframe for charting
        start_time=start_time,
        end_time=end_time
    )

    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total rows: {len(df)}")

    print("Creating charts...")
    chart_manager.create_charts(
        df,
        start_time=start_time,
        end_time=end_time,
        show_candlesticks=False
    )

    print("Done!")


if __name__ == "__main__":
    main()
