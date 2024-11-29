# File: data_management/indicator_manager.py
# Path: forex_trading_system/data_management/indicator_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib
from dataclasses import dataclass
import json
import os
from pathlib import Path
import yaml
import pandas_ta as ta


@dataclass
class IndicatorConfig:
    """Configuration for technical indicators."""
    enabled: bool
    params: Dict
    visualize: bool


class IndicatorManager:
    """Manages calculation and caching of technical indicators."""

    def __init__(self):
        """
        Initialize indicator manager with configuration.
        """
        self.indicator_params = {
            'sma': {'periods': [20, 50]},
            'rsi': {'period': 14},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'bollinger': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'atr': {'period': 14},
            'adx': {'period': 14},
            'dmi': {'period': 14},
        }

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        selected_indicators: list = None,
        indicator_timeframe: Optional[str] = 'D'
    ) -> pd.DataFrame:
        """
        Calculate technical indicators on specified timeframe and merge back.

        Args:
            df: DataFrame with OHLCV data at original timeframe
            selected_indicators: List of indicators to calculate (defaults to all)
            indicator_timeframe: Timeframe to calculate indicators on (e.g., 'D', 'H', '5T'). If None, use original timeframe.
        """
        #! leave volume for now
        if 'volume' in df.columns:
            df = df.drop('volume', axis=1)

        if selected_indicators is None:
            selected_indicators = ['sma', 'rsi', 'macd',
                                   'bollinger', 'atr', 'adx', 'dmi', 'ichimoku']

        try:
            # Store original DataFrame
            original_df = df.copy()

            if indicator_timeframe is not None:
                # Resample to specified timeframe
                resampled_df = df.resample(indicator_timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                }).dropna()

            else:
                # Use original dataframe
                resampled_df = df.copy()

            # Calculate indicators on resampled data
            indicator_df = resampled_df.copy()

            # SMA calculations
            if 'sma' in selected_indicators:
                for period in self.indicator_params['sma']['periods']:
                    indicator_df[f'sma_{period}'] = talib.SMA(
                        indicator_df['close'],
                        timeperiod=period
                    )

            # RSI
            if 'rsi' in selected_indicators:
                indicator_df['rsi'] = talib.RSI(
                    indicator_df['close'],
                    timeperiod=self.indicator_params['rsi']['period']
                )

            # MACD
            if 'macd' in selected_indicators:
                macd, signal, hist = talib.MACD(
                    indicator_df['close'],
                    **self.indicator_params['macd']
                )
                indicator_df['macd'] = macd
                indicator_df['macd_signal'] = signal
                indicator_df['macd_hist'] = hist

            # Bollinger Bands
            if 'bollinger' in selected_indicators:
                upper, middle, lower = talib.BBANDS(
                    indicator_df['close'],
                    **self.indicator_params['bollinger']
                )
                indicator_df['bb_upper'] = upper
                indicator_df['bb_middle'] = middle
                indicator_df['bb_lower'] = lower
                # Calculate Bollinger Bandwidth
                indicator_df['bb_bandwidth'] = ((upper - lower) / middle) * 100

                # Calculate Bollinger %B
                band_width = upper - lower
                indicator_df['bb_percent'] = np.where(
                    band_width == 0,
                    0.0,
                    ((indicator_df['close'] - lower) / band_width) * 100
                )

            # ATR
            if 'atr' in selected_indicators:
                indicator_df['atr'] = talib.ATR(
                    indicator_df['high'],
                    indicator_df['low'],
                    indicator_df['close'],
                    timeperiod=self.indicator_params['atr']['period']
                )

            # DMI
            if 'dmi' in selected_indicators:
                indicator_df['plus_di'] = talib.PLUS_DI(
                    indicator_df['high'],
                    indicator_df['low'],
                    indicator_df['close'],
                    timeperiod=self.indicator_params['dmi']['period']
                )
                indicator_df['minus_di'] = talib.MINUS_DI(
                    indicator_df['high'],
                    indicator_df['low'],
                    indicator_df['close'],
                    timeperiod=self.indicator_params['dmi']['period']
                )

            # ADX
            if 'adx' in selected_indicators:
                indicator_df['adx'] = talib.ADX(
                    indicator_df['high'],
                    indicator_df['low'],
                    indicator_df['close'],
                    timeperiod=self.indicator_params['adx']['period']
                )

            # Ichimoku
            if 'ichimoku' in selected_indicators:
                self._add_ichimoku(indicator_df)

            # Get indicator columns (exclude OHLCV)
            indicator_columns = [col for col in indicator_df.columns
                                 if col not in ['open', 'high', 'low', 'close', 'volume']]

            # Drop any rows with NaN values in indicators
            indicator_df = indicator_df.dropna(subset=indicator_columns)

            # Create final DataFrame with original data
            result_df = original_df.copy()

            # Add each indicator back to original timeframe
            for col in indicator_columns:
                if indicator_timeframe is not None:
                    # Reindex to original timeframe and forward fill
                    result_df[col] = indicator_df[col].reindex(
                        original_df.index, method='ffill'
                    )
                else:
                    # Indicators calculated on original timeframe, just assign
                    result_df[col] = indicator_df[col]

            print(f"Added indicators: {indicator_columns}")
            print(
                f"Original shape: {original_df.shape}, Final shape: {result_df.shape}")
            result_df.dropna(inplace=True)
            return result_df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            print(f"Shape of input DataFrame: {df.shape}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            raise

    def _add_ichimoku(self, df: pd.DataFrame) -> None:
        """Calculate Ichimoku Cloud indicators using pandas_ta."""

        # Calculate ichimoku with append=True to add columns directly to dataframe
        df.ta.ichimoku(
            high='high',
            low='low',
            close='close',
            append=True,
            lookahead=False
        )

        # Rename columns to match our convention
        df.rename(columns={
            'ISA_9': 'senkou_span_a',
            'ISB_26': 'senkou_span_b',
            'ITS_9': 'tenkan_sen',
            'IKS_26': 'kijun_sen'
        }, inplace=True)


# class IndicatorManager:
#     """Manages calculation and caching of technical indicators."""

#     def __init__(self):
#         """
#         Initialize indicator manager with configuration.

#         Args:
#             config_path: Path to indicator configuration file
#             cache_dir: Directory for caching calculated indicators
#         """
#         self.indicator_params = {
#             'sma': {'periods': [20, 50]},
#             'rsi': {'period': 14},
#             'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
#             'bollinger': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
#             'atr': {'period': 14},
#             'adx': {'period': 14},
#             'dmi': {'period': 14},
#         }


#     def load_config(self):
#         """Load indicator configuration from YAML file."""
#         try:
#             with open(self.config_path, 'r') as f:
#                 config = yaml.safe_load(f)
#                 self.indicators = {
#                     name: IndicatorConfig(**params)
#                     for name, params in config['indicators'].items()
#                 }
#         except FileNotFoundError:
#             raise FileNotFoundError(
#                 f"Could not find indicators.yaml in {self.config_path}. Current working directory: {os.getcwd()}")

#     def calculate_indicators(
#         self,
#         df: pd.DataFrame,
#         selected_indicators: list = None
#     ) -> pd.DataFrame:
#         """
#         Calculate technical indicators on daily timeframe and resample back.

#         Args:
#             df: DataFrame with OHLCV data at original timeframe
#             selected_indicators: List of indicators to calculate (defaults to all)
#         """
#         if selected_indicators is None:
#             selected_indicators = ['sma', 'rsi', 'macd', 'bollinger', 'atr', 'adx', 'dmi', 'ichimoku']

#         try:
#             # Store original DataFrame
#             original_df = df.copy()

#             # Resample to daily timeframe
#             daily_df = df.resample('D').agg({
#                 'open': 'first',
#                 'high': 'max',
#                 'low': 'min',
#                 'close': 'last',
#                 'volume': 'sum'
#             }).dropna()

#             # Calculate indicators on daily data
#             daily_indicators = daily_df.copy()

#             # SMA calculations
#             if 'sma' in selected_indicators:
#                 for period in self.indicator_params['sma']['periods']:
#                     daily_indicators[f'sma_{period}'] = talib.SMA(
#                         daily_indicators['close'],
#                         timeperiod=period
#                     )

#             # RSI
#             if 'rsi' in selected_indicators:
#                 daily_indicators['rsi'] = talib.RSI(
#                     daily_indicators['close'],
#                     timeperiod=self.indicator_params['rsi']['period']
#                 )

#             # MACD
#             if 'macd' in selected_indicators:
#                 macd, signal, hist = talib.MACD(
#                     daily_indicators['close'],
#                     **self.indicator_params['macd']
#                 )
#                 daily_indicators['macd'] = macd
#                 daily_indicators['macd_signal'] = signal
#                 daily_indicators['macd_hist'] = hist

#             # Bollinger Bands
#             if 'bollinger' in selected_indicators:
#                 upper, middle, lower = talib.BBANDS(
#                     daily_indicators['close'],
#                     **self.indicator_params['bollinger']
#                 )
#                 daily_indicators['bb_upper'] = upper
#                 daily_indicators['bb_middle'] = middle
#                 daily_indicators['bb_lower'] = lower
#                 # Calculate Bollinger Bandwidth
#                 daily_indicators['bb_bandwidth'] = ((upper - lower) / middle) * 100

#                 # Calculate Bollinger %B
#                 daily_indicators['bb_percent'] = ((daily_indicators['close'] - lower) / (upper - lower)) * 100

#             # ATR
#             if 'atr' in selected_indicators:
#                 daily_indicators['atr'] = talib.ATR(
#                     daily_indicators['high'],
#                     daily_indicators['low'],
#                     daily_indicators['close'],
#                     timeperiod=self.indicator_params['atr']['period']
#                 )

#             # DMI
#             if 'dmi' in selected_indicators:
#                 daily_indicators['plus_di'] = talib.PLUS_DI(
#                     daily_indicators['high'],
#                     daily_indicators['low'],
#                     daily_indicators['close'],
#                     timeperiod=self.indicator_params['dmi']['period']
#                 )
#                 daily_indicators['minus_di'] = talib.MINUS_DI(
#                     daily_indicators['high'],
#                     daily_indicators['low'],
#                     daily_indicators['close'],
#                     timeperiod=self.indicator_params['dmi']['period']
#                 )

#             # ADX
#             if 'adx' in selected_indicators:
#                 daily_indicators['adx'] = talib.ADX(
#                     daily_indicators['high'],
#                     daily_indicators['low'],
#                     daily_indicators['close'],
#                     timeperiod=self.indicator_params['adx']['period']
#                 )
#             if 'ichimoku' in selected_indicators:
#                 self._add_ichimoku(daily_indicators)

#             # Get indicator columns (exclude OHLCV)
#             indicator_columns = [col for col in daily_indicators.columns
#                                if col not in ['open', 'high', 'low', 'close', 'volume']]

#             # Drop any rows with NaN values in indicators
#             daily_indicators = daily_indicators.dropna()

#             # Create final DataFrame with original data
#             result_df = original_df.copy()

#             # Add each indicator back to original timeframe
#             for col in indicator_columns:
#                 # Reindex to original timeframe and forward fill
#                 result_df[col] = daily_indicators[col].reindex(
#                     original_df.index, method='ffill'
#                 )

#             # # Final forward fill and backward fill for any remaining NaNs
#             # result_df[indicator_columns] = result_df[indicator_columns].ffill()

#             print(f"Added indicators: {indicator_columns}")
#             print(f"Original shape: {original_df.shape}, Final shape: {result_df.shape}")

#             return result_df

#         except Exception as e:
#             print(f"Error calculating indicators: {str(e)}")
#             print(f"Shape of input DataFrame: {df.shape}")
#             print(f"Available columns: {df.columns.tolist()}")
#             print(f"Date range: {df.index[0]} to {df.index[-1]}")
#             raise

#     def get_available_indicators(self) -> list:
#         """Get list of available indicators."""
#         return list(self.indicator_params.keys())

#     def update_params(self, new_params: dict) -> None:
#         """Update indicator parameters."""
#         self.indicator_params.update(new_params)

#     # def _add_ichimoku(self, df: pd.DataFrame, params: Dict) -> None:
#     #     """Calculate Ichimoku Cloud indicators."""
#     #     high = df['high']
#     #     low = df['low']

#     #     # Calculate Tenkan-sen (Conversion Line)
#     #     period9_high = high.rolling(
#     #         window=params['conversion_line_period']).max()
#     #     period9_low = low.rolling(
#     #         window=params['conversion_line_period']).min()
#     #     df['tenkan_sen'] = (period9_high + period9_low) / 2

#     #     # Calculate Kijun-sen (Base Line)
#     #     period26_high = high.rolling(window=params['base_line_period']).max()
#     #     period26_low = low.rolling(window=params['base_line_period']).min()
#     #     df['kijun_sen'] = (period26_high + period26_low) / 2

#     #     # Calculate Senkou Span A (Leading Span A)
#     #     df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(
#     #         params['displacement']
#     #     )

#     #     # Calculate Senkou Span B (Leading Span B)
#     #     period52_high = high.rolling(
#     #         window=params['lagging_span_period']).max()
#     #     period52_low = low.rolling(window=params['lagging_span_period']).min()
#     #     df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(
#     #         params['displacement']
#     #     )

#     #     # Calculate Chikou Span (Lagging Span)
#     #     df['chikou_span'] = df['close'].shift(-params['displacement'])

#     def _add_ichimoku(self, df: pd.DataFrame) -> None:
#         """Calculate Ichimoku Cloud indicators using pandas_ta."""

#         # Calculate ichimoku with append=True to add columns directly to dataframe
#         df.ta.ichimoku(
#             high='high',
#             low='low',
#             close='close',
#             append=True,
#             lookahead=False
#         )

#         # Rename columns to match our convention
#         df.rename(columns={
#             'ISA_9': 'senkou_span_a',
#             'ISB_26': 'senkou_span_b',
#             'ITS_9': 'tenkan_sen',
#             'IKS_26': 'kijun_sen'
#         }, inplace=True)
