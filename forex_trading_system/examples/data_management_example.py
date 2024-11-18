# File: examples/data_management_example.py
# Path: forex_trading_system/examples/data_management_example.py

from datetime import datetime, timedelta
import pytz
import pandas as pd
from data_management.dataset_manager import DatasetManager
from data_management.data_quality import QualityMetrics


def example_data_pipeline():
    """Example of loading and monitoring forex data quality."""

    # Initialize the dataset manager
    config_path = "config/"
    dataset_manager = DatasetManager(config_path)

    # Define time range
    end_time = datetime.now(pytz.UTC)
    start_time = end_time - timedelta(days=30)

    # Load and process data for EUR/USD
    currency_pair = "EUR_USD"
    timeframe = "15min"

    try:
        # Load the dataset
        print(f"Loading {currency_pair} data...")
        df = dataset_manager.load_dataset(
            currency_pair=currency_pair,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time
        )

        # Check data quality
        df, quality_metrics = dataset_manager.update_dataset_manager(
            df, timeframe)

        # Print quality metrics
        print("\nData Quality Report:")
        print(f"Missing Values: {quality_metrics.missing_values}")
        print(f"Number of Gaps: {len(quality_metrics.gaps_detected)}")
        print(f"Data Freshness: {quality_metrics.data_freshness}")

        # Print statistics for close price
        if 'close' in quality_metrics.statistics:
            stats = quality_metrics.statistics['close']
            print("\nClose Price Statistics:")
            print(f"Mean: {stats['mean']:.4f}")
            print(f"Std Dev: {stats['std']:.4f}")
            print(f"Skewness: {stats['skewness']:.4f}")
            print(f"Kurtosis: {stats['kurtosis']:.4f}")

        # Example of dealing with detected issues
        if quality_metrics.gaps_detected:
            print("\nDetected gaps:")
            for start, end in quality_metrics.gaps_detected:
                gap_duration = end - start
                print(f"Gap from {start} to {end} (duration: {gap_duration})")

        if quality_metrics.outliers_detected:
            print("\nOutliers detected:")
            for column, count in quality_metrics.outliers_detected.items():
                if count > 0:
                    print(f"{column}: {count} outliers")

        return df, quality_metrics

    except Exception as e:
        print(f"Error in data pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    df, metrics = example_data_pipeline()
