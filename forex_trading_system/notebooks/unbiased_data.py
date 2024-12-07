from tqdm import tqdm
import logging
from data_management.preprocessor import DataPreprocessor
from data_management.indicator_manager import IndicatorManager

import os
import sys
import pandas as pd
import numpy as np
import pytz
from typing import List, Optional
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


processor = DataPreprocessor()
indicator_manager = IndicatorManager()


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_prep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('dataset_prep')


def get_max_indicator_periods(indicator_params):
    """Calculate the maximum number of periods required by all indicators."""
    max_period = 0
    for indicator, params in indicator_params.items():
        periods = []
        if indicator == 'sma':
            periods.extend(params.get('periods', []))
        elif indicator in ['rsi', 'atr', 'adx', 'dmi']:
            periods.append(params.get('period', 0))
        elif indicator == 'macd':
            periods.append(params.get('slowperiod', 0))
        elif indicator == 'bollinger':
            periods.append(params.get('timeperiod', 0))
        elif indicator == 'ichimoku':
            # Ichimoku uses periods of 9, 26, 52
            periods.extend([9, 26, 52])
        if periods:
            max_period = max(max_period, max(periods))
    return max_period


def prepare_unbiased_dataset_row_by_row(
    df: pd.DataFrame,
    indicator_manager,
    indicator_timeframe: str = '1h',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Prepare dataset with technical indicators calculated without look-ahead bias,
    processing data row by row without skipping any rows.

    At each step:
    - Resample to 5-minute candles
    - Take all available historical data up to the current row
    - Calculate indicators (which may return NaNs if insufficient data)
    - Append the combined row (price + indicators) to results

    This ensures the final result_df has the same number of rows as the 5-minute DataFrame,
    and that early rows will contain NaNs until enough history is accumulated.
    """

    if verbose:
        logger.info("Starting data preparation using row-by-row method...")

    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    elif df.index.tz != pytz.UTC:
        df.index = df.index.tz_convert('UTC')

    data_cache_size = 1_000
    if indicator_timeframe == '1h':
        data_cache_size = 1_000

    if indicator_timeframe == 'D':
        data_cache_size = 20_000
    # Create 5-minute OHLC data
    df_5min = df.resample('5min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    if verbose:
        logger.info(f"Resampled to 5-minute candles. Shape: {df_5min.shape}")

    # Initialize results
    results_dict = {}  # Using a dictionary instead of a list

    # Initialize data cache
    data_cache = pd.DataFrame(columns=['open', 'high', 'low', 'close'])

    if verbose:
        iterator = tqdm(df_5min.iterrows(), total=len(
            df_5min), desc='Processing rows')
    else:
        iterator = df_5min.iterrows()

    for idx, row in iterator:
        try:
            # Append current candle to cache
            data_cache.loc[idx] = row
            if len(data_cache) > data_cache_size:
                data_cache = data_cache.iloc[-data_cache_size:]

            # Get data up to current time
            data_up_to_now = data_cache.loc[:idx]

            # Resample to indicator timeframe
            period_data = data_up_to_now.resample(
                indicator_timeframe,
                closed='right',
                label='right'
            ).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            })

            # Remove any rows with NaN values before calculating indicators
            period_data = period_data.dropna()

            if len(period_data) > 0:
                # Calculate indicators
                indicators_df = indicator_manager.calculate_indicators_unbiased(
                    period_data)

                # Get the last row of indicators (most recent)
                last_indicator_row = indicators_df.iloc[-1]

                # Combine current candle with indicators
                combined_row = pd.concat([row, last_indicator_row])
                # logger.info(f"Combined row: {combined_row}")
            else:
                # If no valid period data, use just the price data
                combined_row = row.copy()

            # Store in dictionary using timestamp as key
            results_dict[idx] = combined_row.to_dict()
            # print(results_dict)

            # if verbose and idx.minute == 0:  # Log only for hourly candles
            # logger.info(f"\nProcessing timestamp: {idx}")
            # logger.info(f"Period data shape: {period_data.shape}")

        except Exception as e:
            logger.error(f"Error processing row at {idx}: {str(e)}")
            # On error, store just the price data
            results_dict[idx] = row.to_dict()
            continue

    # Create final DataFrame using the dictionary
    result_df = pd.DataFrame.from_dict(results_dict, orient='index')

    if verbose:
        logger.info(f"\nFinal dataset prepared. Shape: {result_df.shape}")
        logger.info(
            f"Date range: {result_df.index[0]} to {result_df.index[-1]}")

        # Add data quality check
        indicator_cols = [col for col in result_df.columns
                          if col not in ['open', 'high', 'low', 'close']]
        if indicator_cols:
            nan_pcts = result_df[indicator_cols].isna().mean() * 100
            logger.info("\nPercentage of NaN values in indicator columns:")
            for col, pct in nan_pcts.items():
                logger.info(f"{col}: {pct:.2f}%")

    return result_df


def process_currency_pairs(
    currencies: List[str],
    base_path: str = './',
    indicator_timeframe: str = '1h'
) -> pd.DataFrame:
    """
    Process multiple currency pairs with unbiased indicator calculation using row-by-row method.

    Args:
        currencies: List of currency pairs to process
        base_path: Base path for data storage
        indicator_timeframe: Timeframe for indicator calculation

    Returns:
        Processed DataFrame for inspection
    """
    for ccy in currencies:
        logger.info(f"\nProcessing {ccy}...")
        source = f'/Volumes/ssd_fat2/ai6_trading_bot/datasets/1min/{ccy}.parquet'

        try:
            # Read source data
            df = pd.read_parquet(source)

            # Prepare dataset with unbiased indicators
            df_with_indicators = prepare_unbiased_dataset_row_by_row(
                df=df,
                indicator_manager=indicator_manager,
                indicator_timeframe=indicator_timeframe,
                verbose=True
            )

            if df_with_indicators.empty:
                logger.info(f"No data processed for {ccy}. Skipping.")
                continue

            output_path_not_norm = f'{base_path}/{ccy}_5min_indics_{indicator_timeframe}_not_norm_unbiased.parquet'
            df_with_indicators.to_parquet(output_path_not_norm)

            # Normalize the data if desired
            logger.info("\nNormalizing data...")
            df_norm = processor.normalize_simple(df=df_with_indicators)

            # Save results
            output_path = f'{base_path}/{ccy}_5min_indics_{indicator_timeframe}_norm_unbiased.parquet'
            df_norm.to_parquet(output_path)

            logger.info(f"Completed processing {ccy}")
            return df_norm  # Return for inspection

        except Exception as e:
            logger.info(f"Error processing {ccy}: {str(e)}")
            continue
