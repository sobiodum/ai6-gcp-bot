# File: data_management/dataset_manager.py
# Path: forex_trading_system/data_management/dataset_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging
import h5py
from datetime import datetime, timedelta
import pytz

from .data_fetcher import OandaDataFetcher
from .preprocessor import DataPreprocessor
from .indicator_manager import IndicatorManager


class DatasetManager:
    """Manages loading, updating, and preprocessing of forex datasets."""

    def __init__(self, base_path: str = "/Volumes/ssd_fat2/ai6_trading_bot/datasets/1min"):
        """
        Initialize the dataset manager.

        Args:
            base_path: Path to historical parquet files
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path)
        self.data_dir = Path("data")
        self.cache_dir = self.data_dir / "cache"

        # Get project root directory
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize components
        self.fetcher = OandaDataFetcher(
            account_id='101-004-30348600-001',
            access_token='9317ace4596d61e3e98b1a53b2342483-45d3ad4084c80b111727a9fada9ef0ff'
        )
        self.preprocessor = DataPreprocessor(window_size=252)
        self.indicator_manager = IndicatorManager(
            config_path=str(config_path),
            cache_dir=str(self.cache_dir)
        )

        # Standard timeframes we support
        self.timeframes = ["1min", "5min", "15min", "1h"]

    def load_and_update_dataset(
        self,
        currency_pair: str,
        timeframe: str = "1min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True,
        normalize: bool = False  # New parameter
    ) -> pd.DataFrame:
        """
        Load historical data and update with latest from OANDA if needed.

        Args:
            currency_pair: Currency pair symbol (e.g., "EUR_USD")
            timeframe: Time granularity ("1min", "5min", "15min", "1h")
            start_time: Start of data range
            end_time: End of data range
            use_cache: Whether to use cached data
        """
        # Load historical data from parquet
        file_path = self.base_path / f"{currency_pair}.parquet"

        try:
            df = pd.read_parquet(file_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Ensure index is UTC-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            # Get the last timestamp from historical data
            last_historical_timestamp = df.index[-1]

            # Apply time filters before checking for updates
            if start_time:
                start_ts = pd.Timestamp(start_time)
                if start_ts.tz is None:
                    start_ts = start_ts.tz_localize('UTC')
                df = df[df.index >= start_ts]

            if end_time:
                end_ts = pd.Timestamp(end_time)
                if end_ts.tz is None:
                    end_ts = end_ts.tz_localize('UTC')
                df = df[df.index <= end_ts]

            # Only fetch new data if we're requesting recent data
            current_time = datetime.now(pytz.UTC)
            end_time_ts = pd.Timestamp(end_time) if end_time else None
            print(f"end_time is {end_time_ts} ")
            if end_time_ts is not None:
                if end_time_ts.tz is None:
                    end_time_ts = end_time_ts.tz_localize('UTC')

            should_update = (end_time is None or end_time_ts > (current_time - timedelta(days=1))) and \
                (current_time - last_historical_timestamp > timedelta(days=1))

            if should_update:
                self.logger.info(
                    f"Fetching recent data for {currency_pair} from OANDA")

                recent_start = last_historical_timestamp - timedelta(days=1)
                recent_data = self.fetcher.fetch_candles(
                    currency_pair,
                    timeframe,
                    start_time=recent_start,
                    end_time=current_time
                )

                if not recent_data.empty:
                    recent_data = recent_data[recent_data.index >
                                              last_historical_timestamp]
                    df = pd.concat([df, recent_data])

                    self.logger.info(
                        f"Saving updated data for {currency_pair}")
                    df.to_parquet(file_path)
            # Resample if needed
            if timeframe != "1min":
                df = self.aggregate_timeframe(df, "1min", timeframe)
            # Calculate indicators
            print(
                f"no fetching, move to calculate indicators for: {currency_pair}")
            df = self.indicator_manager.calculate_indicators(df)

            # Normalize features
            indicator_columns = [col for col in df.columns
                                 if col not in ['open', 'high', 'low', 'close', 'volume']]
            print(f"Move to normalize for: {currency_pair}")

            if normalize:
                indicator_columns = [col for col in df.columns
                                     if col not in ['open', 'high', 'low', 'close', 'volume']]
                print(f"Normalizing features for: {currency_pair}")
                df = self.preprocessor.normalize_features(
                    df, indicator_columns)

            return df

        except Exception as e:
            self.logger.error(
                f"Error loading/updating data for {currency_pair}: {str(e)}")
            raise

    def update_live_data(self) -> Dict[str, pd.DataFrame]:
        """
        Update all datasets with latest data from OANDA.

        Returns:
            Dict mapping currency pairs to their updated DataFrames
        """
        updated_data = {}
        end_time = datetime.now()

        # Update data for all currency pairs
        for pair in self.get_currency_pairs():
            try:
                df = self.load_and_update_dataset(
                    currency_pair=pair,
                    timeframe="1min",  # Always fetch 1-min data
                    end_time=end_time,
                    use_cache=False
                )

                # Store different timeframe aggregations
                updated_data[pair] = {
                    "1min": df
                }

                # Create aggregated timeframes
                for timeframe in ["5min", "15min", "1h"]:
                    updated_data[pair][timeframe] = self.aggregate_timeframe(
                        df, "1min", timeframe
                    )

            except Exception as e:
                self.logger.error(
                    f"Error updating {pair}: {str(e)}"
                )

        return updated_data

    def aggregate_timeframe(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Aggregate data from a smaller to a larger timeframe.

        Args:
            df: Input DataFrame with OHLCV data
            source_timeframe: Original timeframe
            target_timeframe: Desired timeframe
        """
        if source_timeframe == target_timeframe:
            return df

        # Define resampling rules
        timeframe_rules = {
            "5min": "5T",
            "15min": "15T",
            "1h": "1h"
        }

        # Resample OHLCV data
        resampled = df.resample(timeframe_rules[target_timeframe]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        # Forward fill any missing values
        resampled.fillna(method='ffill', inplace=True)

        return resampled

    def get_currency_pairs(self) -> List[str]:
        """Get list of supported currency pairs."""
        return [
            'GBP_CHF',
            'GBP_JPY',
            'EUR_CHF',
            'EUR_JPY',
            'USD_CHF',
            'EUR_CAD',
            'EUR_USD',
            'GBP_USD',
            'EUR_GBP',
            'USD_JPY',
            'USD_CAD',
            'AUD_USD',
            'CHF_JPY',
            'AUD_JPY',
            'NZD_USD',
            'NZD_JPY',
            'XAU_USD',
            'XAG_USD']
