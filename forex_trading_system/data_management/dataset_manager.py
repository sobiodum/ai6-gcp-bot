# File: data_management/dataset_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pytz
from dataclasses import dataclass

from .data_fetcher import OandaDataFetcher
from .preprocessor import DataPreprocessor
from .indicator_manager import IndicatorManager


@dataclass
class TimeframeConfig:
    """Configuration for timeframe handling."""
    code: str  # Internal code (e.g., "1min", "1h")
    resample_rule: str  # Pandas resample rule (e.g., "1T", "1H")
    description: str  # Human readable description
    minutes: int  # Number of minutes in timeframe


class DatasetManager:
    """Manages loading, preprocessing, and splitting of forex datasets."""

    def __init__(
        self,
        base_path: str = "/Volumes/ssd_fat2/ai6_trading_bot/datasets/1min",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ):
        """
        Initialize the dataset manager.

        Args:
            base_path: Path to historical parquet files
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation
            test_ratio: Ratio of data for testing
        """
        self.logger = logging.getLogger(__name__)
        self.base_path = Path(base_path)
        self.data_dir = Path("data")
        self.cache_dir = self.data_dir / "cache"

        # Initialize timeframe configurations
        self._init_timeframes()

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
          
        )

    def _init_timeframes(self):
        """Initialize timeframe configurations."""
        # Define all available timeframe configurations
        self.timeframe_configs = {
            "1min": TimeframeConfig("1min", "1T", "1 Minute", 1),
            "5min": TimeframeConfig("5min", "5T", "5 Minutes", 5),
            "15min": TimeframeConfig("15min", "15T", "15 Minutes", 15),
            "30min": TimeframeConfig("30min", "30T", "30 Minutes", 30),
            "1h": TimeframeConfig("1h", "1h", "1 Hour", 60),
            "4h": TimeframeConfig("4h", "4h", "4 Hours", 240),
            "1d": TimeframeConfig("1d", "1D", "1 Day", 1440)
        }
    
    def split_dataset(
        self,
        df: pd.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train, validation and test sets.
        
        Args:
            df: Input DataFrame
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15)
            test_ratio: Proportion for testing (default: 0.15)
            shuffle: Whether to shuffle before splitting (default: False for time series)
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
        
        n = len(df)
        indices = np.arange(n)
        
        if shuffle:
            np.random.shuffle(indices)
        
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[indices[:train_idx]]
        val_df = df.iloc[indices[train_idx:val_idx]]
        test_df = df.iloc[indices[val_idx:]]
        
        # Sort by index again if shuffled
        if shuffle:
            train_df = train_df.sort_index()
            val_df = val_df.sort_index()
            test_df = test_df.sort_index()
        
        print(f"Dataset split sizes:")
        print(f"Training: {len(train_df)} samples ({len(train_df)/n:.1%})")
        print(f"Validation: {len(val_df)} samples ({len(val_df)/n:.1%})")
        print(f"Test: {len(test_df)} samples ({len(test_df)/n:.1%})")
        
        return train_df, val_df, test_df

    def _detect_timeframe(self, df: pd.DataFrame) -> Optional[str]:
        """
        Attempt to detect the timeframe of a DataFrame.

        Args:
            df: Input DataFrame with datetime index

        Returns:
            Detected timeframe code or None if cannot determine
        """
        if len(df) < 2:
            return None

        # Calculate median time difference between rows
        median_diff = pd.Series(df.index).diff().median()
        minutes_diff = median_diff.total_seconds() / 60

        # Find closest matching timeframe
        for code, config in self.timeframe_configs.items():
            if abs(config.minutes - minutes_diff) < 1:  # 1 minute tolerance
                return code

        return None

    def prepare_dataset(
        self,
        data_source: Union[str, pd.DataFrame],
        target_timeframe: str = "1h",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        normalize: bool = True,
        add_indicators: bool = True,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Prepare dataset for training/testing."""
        # Load or use provided data
        if isinstance(data_source, str):
            df = self.load_and_update_dataset(
                currency_pair=data_source,
                timeframe="1min",
                start_time=start_time,
                end_time=end_time,
                use_cache=use_cache
            )
            source_timeframe = "1min"
        
        elif isinstance(data_source, pd.DataFrame):
            df = data_source.copy()
            source_timeframe = "1h"  # Assume hourly data for pre-loaded DataFrame
            
            # Ensure DataFrame index is timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # Apply time filters if provided
            if start_time is not None:
                # Convert start_time to UTC if needed
                if not isinstance(start_time, pd.Timestamp):
                    start_time = pd.Timestamp(start_time)
                if start_time.tz is None:
                    start_time = start_time.tz_localize('UTC')
                df = df[df.index >= start_time]
                self.logger.info(f"Applied start time filter: {start_time}")
                
            if end_time is not None:
                # Convert end_time to UTC if needed
                if not isinstance(end_time, pd.Timestamp):
                    end_time = pd.Timestamp(end_time)
                if end_time.tz is None:
                    end_time = end_time.tz_localize('UTC')
                df = df[df.index <= end_time]
                self.logger.info(f"Applied end time filter: {end_time}")
            
            self.logger.info(f"After time filtering: {len(df)} rows")
        else:
            raise ValueError("data_source must be either a currency pair string or a DataFrame")

        # Early check for empty DataFrame
        if df.empty:
            raise ValueError("No data available after applying time filters")

        # Store original length for logging
        original_length = len(df)
        
        # Add indicators if requested (before aggregation)
        if add_indicators:
            self.logger.info("Adding technical indicators...")
            df = self.indicator_manager.calculate_indicators(df)
            self.logger.info(f"After adding indicators: {len(df)} rows")
            
           
            
        # Aggregate to target timeframe if needed
        if source_timeframe != target_timeframe:
            self.logger.info(f"Aggregating data from {source_timeframe} to {target_timeframe}")
            df = self.aggregate_timeframe(df, source_timeframe, target_timeframe)
            self.logger.info(f"After aggregation: {len(df)} rows")

        # Remove any remaining NaN rows
        df_length_before = len(df)
        df = df.dropna()
        rows_dropped = df_length_before - len(df)
        
        if rows_dropped > 0:
            self.logger.info(
                f"Dropped {rows_dropped} rows containing NaN values "
                f"({rows_dropped/df_length_before:.1%} of data)")

        # Verify we have enough data left
        if len(df) < 100:  # Arbitrary minimum, adjust as needed
            raise ValueError(
                f"Insufficient data remaining after processing: {len(df)} rows.\n"
                f"Original rows: {original_length}\n"
                f"Date range: {df.index[0]} to {df.index[-1]}\n"
                f"Consider adjusting the date range or indicator parameters.")

        # Normalize if requested (after NaN removal)
        if normalize:
            feature_columns = [col for col in df.columns 
                            if col not in ['open', 'high', 'low', 'close', 'volume']]
            df = self.preprocessor.normalize_features(df, feature_columns)

        # Final verification
        if df.isnull().any().any():
            problematic_columns = df.columns[df.isnull().any()].tolist()
            raise ValueError(
                f"Unexpected NaN values found in columns {problematic_columns} "
                "after preprocessing")

        # Log data preparation summary
        self.logger.info(
            f"Data preparation complete:\n"
            f"- Original rows: {original_length}\n"
            f"- Final rows: {len(df)}\n"
            f"- Timeframe: {target_timeframe}\n"
            f"- Features: {len(df.columns)}\n"
            f"- Date range: {df.index[0]} to {df.index[-1]}"
        )

        return df

    def aggregate_timeframe(
        self,
        df: pd.DataFrame,
        source_timeframe: str,
        target_timeframe: str
    ) -> pd.DataFrame:
        """Aggregate data from smaller to larger timeframe."""
        if source_timeframe == target_timeframe:
            return df.copy()

        # Validate timeframes
        if (source_timeframe not in self.timeframe_configs or 
            target_timeframe not in self.timeframe_configs):
            raise ValueError(
                f"Unsupported timeframe. Supported: {list(self.timeframe_configs.keys())}")

        source_config = self.timeframe_configs[source_timeframe]
        target_config = self.timeframe_configs[target_timeframe]

        # Validate aggregation direction
        if source_config.minutes > target_config.minutes:
            raise ValueError("Cannot aggregate to smaller timeframe")

        try:
            # Basic OHLCV aggregation
            resampled = df.resample(target_config.resample_rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })

            # Aggregate other columns (indicators, if present)
            for column in df.columns:
                if column in ['open', 'high', 'low', 'close', 'volume']:
                    continue

                # Determine aggregation method based on column name/type
                if any(indicator in column.lower() for indicator in ['sma', 'ema', 'mean']):
                    resampled[column] = df[column].resample(
                        target_config.resample_rule).mean()
                elif any(indicator in column.lower() for indicator in ['std', 'vol']):
                    resampled[column] = df[column].resample(
                        target_config.resample_rule).last()
                elif any(indicator in column.lower() for indicator in ['rsi', 'adx', 'macd']):
                    resampled[column] = df[column].resample(
                        target_config.resample_rule).last()
                else:
                    resampled[column] = df[column].resample(
                        target_config.resample_rule).last()

            return resampled

        except Exception as e:
            self.logger.error(
                f"Error during timeframe aggregation: {str(e)}")
            raise

    def load_parquet_dataset(
        self,
        pair: str,
        timeframe: str = "1h"
    ) -> pd.DataFrame:
        """Load dataset from parquet file."""
        file_path = self.base_path.parent / timeframe / f"{pair}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No dataset found at {file_path}")
            
        df = pd.read_parquet(file_path)
        
        # Ensure index is datetime with UTC timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
            
        return df

    def get_currency_pairs(self) -> List[str]:
        """Get list of supported currency pairs."""
        return [
            'GBP_CHF', 'GBP_JPY', 'EUR_CHF', 'EUR_JPY', 'USD_CHF',
            'EUR_CAD', 'EUR_USD', 'GBP_USD', 'EUR_GBP', 'USD_JPY',
            'USD_CAD', 'AUD_USD', 'CHF_JPY', 'AUD_JPY', 'NZD_USD',
            'NZD_JPY', 'XAU_USD', 'XAG_USD'
        ]

    def load_and_update_dataset(
        self,
        currency_pair: str,
        timeframe: str = "1min",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        use_cache: bool = True,
        normalize: bool = False
    ) -> pd.DataFrame:
        """Load and optionally update dataset from OANDA."""
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
            print(f"should_update is {should_update}")

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
                print('WARNING: timeframe is loading new Oanda data is not 1min')
                df = self.aggregate_timeframe(df, "1min", timeframe)

            return df

        except Exception as e:
            self.logger.error(
                f"Error loading data for {currency_pair}: {str(e)}")
            raise