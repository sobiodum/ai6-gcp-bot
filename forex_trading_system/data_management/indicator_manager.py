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
    timeframe: str  # Added to specify calculation timeframe for each indicator


class IndicatorManagerPandas:
    """Manages calculation and caching of technical indicators."""

    def __init__(self):
        """
        Initialize indicator manager with enhanced configuration.
        """
        self.indicator_params = {
            'sma': {
                'periods': [20, 50, 200],
                # Specify timeframe for each period
                'timeframes': ['native', 'native', 'D']
            },
            'rsi': {'period': 14},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'bollinger': {
                'timeperiod': 20,
                'nbdevup': 2,
                'nbdevdn': 2,
                'bandwidth_lookback': 52  # For bandwidth z-score calculation
            },
            'atr': {'period': 14},
            'adx': {'period': 14},
            'dmi': {'period': 14},
            'ichimoku': {
                'tenkan': 9,
                'kijun': 26,
                'senkou_b': 52
            },

        }

    def _calculate_moving_averages(self, df: pd.DataFrame, timeframe: str = None) -> pd.DataFrame:
        """
        Calculate moving averages with support for different timeframes.

        Args:
            df: DataFrame with price data
            timeframe: Target timeframe for calculation
        """
        result_df = df.copy()

        for period, tf in zip(self.indicator_params['sma']['periods'],
                              self.indicator_params['sma']['timeframes']):

            if tf == 'D' and timeframe != 'D':
                # Calculate periods needed for daily SMA based on intraday frequency
                candles_per_day = int(pd.Timedelta('1D') / df.index.freq)
                adjusted_period = period * candles_per_day

                # Resample to daily first
                daily_close = df['close'].resample('D').last()
                daily_sma = daily_close.rolling(window=period).mean()

                # Forward fill back to original frequency
                result_df[f'sma_{period}_{tf}'] = daily_sma.reindex(
                    df.index, method='ffill'
                )
            else:
                # Calculate on native timeframe
                result_df[f'sma_{period}'] = df['close'].rolling(
                    window=period
                ).mean()

        return result_df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced Bollinger Bands calculation with bandwidth z-score.
        """
        result_df = df.copy()

        # Calculate standard Bollinger Bands
        period = self.indicator_params['bollinger']['timeperiod']
        std_dev = self.indicator_params['bollinger']['nbdevup']

        middle = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        result_df['bb_upper'] = middle + (std * std_dev)
        result_df['bb_lower'] = middle - (std * std_dev)
        result_df['bb_middle'] = middle

        # Calculate Bandwidth
        bandwidth = (result_df['bb_upper'] -
                     result_df['bb_lower']) / result_df['bb_middle']
        result_df['bb_bandwidth'] = bandwidth * 100

        # Calculate Bandwidth Z-score
        lookback = self.indicator_params['bollinger']['bandwidth_lookback']
        result_df['bb_bandwidth_zscore'] = (
            (bandwidth - bandwidth.rolling(lookback).mean()) /
            bandwidth.rolling(lookback).std()
        )

        return result_df

    def _calculate_dmi_adx_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced DMI/ADX calculation with additional signals.
        """
        result_df = df.copy()
        period = self.indicator_params['dmi']['period']

        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()

        # Calculate DM
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']

        pdm = up_move.where(
            (up_move > down_move) & (up_move > 0),
            0
        )
        ndm = down_move.where(
            (down_move > up_move) & (down_move > 0),
            0
        )

        # Smooth DM
        smoothed_pdm = pdm.rolling(period).mean()
        smoothed_ndm = ndm.rolling(period).mean()

        # Calculate DI
        pdi = 100 * (smoothed_pdm / atr)
        ndi = 100 * (smoothed_ndm / atr)

        result_df['plus_di'] = pdi
        result_df['minus_di'] = ndi

        # Calculate ADX
        dx = 100 * abs(pdi - ndi) / (pdi + ndi)
        result_df['adx'] = dx.rolling(period).mean()

        # Add ADX threshold signals
        result_df['adx_below_15'] = result_df['adx'] < 15
        result_df['dmi_cross'] = (
            (result_df['plus_di'] > result_df['minus_di']) &
            (result_df['plus_di'].shift(1) <= result_df['minus_di'].shift(1))
        )

        return result_df

    def _calculate_golden_cross_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate golden/death cross signals with trend filters.
        """
        result_df = df.copy()
        fast_period = self.indicator_params['golden_cross']['fast_period']
        slow_period = self.indicator_params['golden_cross']['slow_period']

        # Calculate moving averages if not already present
        if f'sma_{fast_period}' not in result_df.columns:
            result_df[f'sma_{fast_period}'] = df['close'].rolling(
                fast_period).mean()
        if f'sma_{slow_period}' not in result_df.columns:
            result_df[f'sma_{slow_period}'] = df['close'].rolling(
                slow_period).mean()

        # Calculate cross signals
        result_df['golden_cross'] = (
            (result_df[f'sma_{fast_period}'] > result_df[f'sma_{slow_period}']) &
            (result_df[f'sma_{fast_period}'].shift(1) <=
             result_df[f'sma_{slow_period}'].shift(1))
        )
        result_df['death_cross'] = (
            (result_df[f'sma_{fast_period}'] < result_df[f'sma_{slow_period}']) &
            (result_df[f'sma_{fast_period}'].shift(1) >=
             result_df[f'sma_{slow_period}'].shift(1))
        )

        # Add trend filters
        result_df['price_above_200sma'] = df['close'] > result_df[f'sma_{slow_period}']
        result_df['sma200_rising'] = (
            result_df[f'sma_{slow_period}'] > result_df[f'sma_{slow_period}'].shift(
                1)
        )

        return result_df


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

    def calculate_indicators_unbiased(self, df: pd.DataFrame, selected_indicators: list = None) -> pd.DataFrame:
        """Calculate technical indicators with proper error handling and validation."""
        try:
            # Create a copy of the input DataFrame
            indicator_df = df.copy()

            if selected_indicators is None:
                selected_indicators = ['sma', 'rsi', 'macd',
                                       'bollinger', 'atr', 'adx', 'dmi', 'ichimoku']

            # SMA calculations
            if 'sma' in selected_indicators:
                for period in self.indicator_params['sma']['periods']:
                    sma = talib.SMA(df['close'].values, timeperiod=period)
                    indicator_df[f'sma_{period}'] = sma

            # RSI
            if 'rsi' in selected_indicators:
                rsi = talib.RSI(
                    df['close'].values, timeperiod=self.indicator_params['rsi']['period'])
                indicator_df['rsi'] = rsi

            # MACD
            if 'macd' in selected_indicators:
                macd_params = self.indicator_params['macd']
                macd, signal, hist = talib.MACD(
                    df['close'].values,
                    fastperiod=macd_params['fastperiod'],
                    slowperiod=macd_params['slowperiod'],
                    signalperiod=macd_params['signalperiod']
                )
                indicator_df['macd'] = macd
                indicator_df['macd_signal'] = signal
                indicator_df['macd_hist'] = hist

            # Bollinger Bands
            if 'bollinger' in selected_indicators:
                bb_params = self.indicator_params['bollinger']
                upper, middle, lower = talib.BBANDS(
                    df['close'].values,
                    timeperiod=bb_params['timeperiod'],
                    nbdevup=bb_params['nbdevup'],
                    nbdevdn=bb_params['nbdevdn']
                )
                indicator_df['bb_upper'] = upper
                indicator_df['bb_middle'] = middle
                indicator_df['bb_lower'] = lower

                # Calculate additional Bollinger Band metrics
                indicator_df['bb_bandwidth'] = (upper - lower) / middle * 100
                indicator_df['bb_percent'] = (
                    df['close'] - lower) / (upper - lower) * 100

            # ATR
            if 'atr' in selected_indicators:
                atr = talib.ATR(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=self.indicator_params['atr']['period']
                )
                indicator_df['atr'] = atr

            # DMI
            if 'dmi' in selected_indicators:
                plus_di = talib.PLUS_DI(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=self.indicator_params['dmi']['period']
                )
                minus_di = talib.MINUS_DI(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=self.indicator_params['dmi']['period']
                )
                indicator_df['plus_di'] = plus_di
                indicator_df['minus_di'] = minus_di

            # ADX
            if 'adx' in selected_indicators:
                adx = talib.ADX(
                    df['high'].values,
                    df['low'].values,
                    df['close'].values,
                    timeperiod=self.indicator_params['adx']['period']
                )
                indicator_df['adx'] = adx

            # Ichimoku Cloud
            if 'ichimoku' in selected_indicators:
                ichimoku_values = self._add_ichimoku_no_look_ahead(df)

                # Assign each calculated indicator to the output DataFrame
                for col_name, values in ichimoku_values.items():
                    indicator_df[col_name] = values
                # # Calculate Ichimoku components
                # high_values = df['high'].values
                # low_values = df['low'].values

                # # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
                # period9_high = pd.Series(high_values).rolling(window=9).max()
                # period9_low = pd.Series(low_values).rolling(window=9).min()
                # indicator_df['tenkan_sen'] = (period9_high + period9_low) / 2

                # # Kijun-sen (Base Line): (26-period high + 26-period low)/2
                # period26_high = pd.Series(high_values).rolling(window=26).max()
                # period26_low = pd.Series(low_values).rolling(window=26).min()
                # indicator_df['kijun_sen'] = (period26_high + period26_low) / 2

                # # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
                # indicator_df['senkou_span_a'] = (
                #     indicator_df['tenkan_sen'] + indicator_df['kijun_sen']) / 2

                # # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
                # period52_high = pd.Series(high_values).rolling(window=52).max()
                # period52_low = pd.Series(low_values).rolling(window=52).min()
                # indicator_df['senkou_span_b'] = (
                #     period52_high + period52_low) / 2

            return indicator_df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            print(f"Shape of input DataFrame: {df.shape}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            raise

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

            # print(f"Added indicators: {indicator_columns}")
            # print(
            #     f"Original shape: {original_df.shape}, Final shape: {result_df.shape}")
            # result_df.dropna(inplace=True)
            return result_df

        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            print(f"Shape of input DataFrame: {df.shape}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            raise

    def _add_ichimoku_no_look_ahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud indicators without lookahead bias.
        Returns DataFrame with calculated indicators.
        """
        temp_df = df.copy()

        # Calculate ichimoku with append=True to add columns directly to temp_df
        temp_df.ta.ichimoku(
            high='high',
            low='low',
            close='close',
            append=True,
            lookahead=False
        )

        # Rename the generated columns to your desired names
        temp_df.rename(columns={
            'ISA_9': 'senkou_span_a',
            'ISB_26': 'senkou_span_b',
            'ITS_9': 'tenkan_sen',
            'IKS_26': 'kijun_sen'
        }, inplace=True)

        # Return just the DataFrame of these columns (NaNs may be present if not enough data)
        return temp_df

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

        #! Rename columns to match our convention
        # df.rename(columns={
        #     'ISA_9': 'senkou_span_a',
        #     'ISB_26': 'senkou_span_b',
        #     'ITS_9': 'tenkan_sen',
        #     'IKS_26': 'kijun_sen'
        # }, inplace=True)


class DualTimeframeIndicators:
    def __init__(self, higher_timeframe='1D'):
        self.higher_timeframe = higher_timeframe

        self.indicator_params = {
            'sma': [20, 50, 200],
            'rsi': {'period': 14},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'bollinger': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'adx': {'period': 14},
        }

    def calculate_indicators(self, df: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """Calculate indicators with specified suffix."""
        df = df.copy()

        # SMAs
        for period in self.indicator_params['sma']:
            df[f'sma_{period}_{suffix}'] = talib.SMA(
                df['close'], timeperiod=period)

        # RSI
        df[f'rsi_{suffix}'] = talib.RSI(df['close'],
                                        timeperiod=self.indicator_params['rsi']['period'])

        # MACD
        macd, signal, hist = talib.MACD(
            df['close'], **self.indicator_params['macd'])
        df[f'macd_{suffix}'] = macd
        df[f'macd_signal_{suffix}'] = signal
        df[f'macd_hist_{suffix}'] = hist

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            df['close'], **self.indicator_params['bollinger'])
        df[f'bb_upper_{suffix}'] = upper
        df[f'bb_middle_{suffix}'] = middle
        df[f'bb_lower_{suffix}'] = lower

        # ADX
        df[f'adx_{suffix}'] = talib.ADX(df['high'], df['low'], df['close'],
                                        timeperiod=self.indicator_params['adx']['period'])

        return df

    def add_dual_timeframe_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators on both original and higher timeframes."""
        # Calculate indicators on original timeframe
        df = self.calculate_indicators(df, suffix='orig')

        # Resample to higher timeframe and calculate indicators
        resampled = df.resample(self.higher_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).dropna()

        higher_df = self.calculate_indicators(
            resampled, suffix=self.higher_timeframe.lower())

        # Select only the indicator columns from higher timeframe
        indicator_cols = [col for col in higher_df.columns if col not in [
            'open', 'high', 'low', 'close']]
        higher_indicators = higher_df[indicator_cols]

        # Forward fill higher timeframe indicators to original timeframe
        aligned_indicators = higher_indicators.reindex(
            df.index, method='ffill')

        # Combine original data with both sets of indicators
        result = pd.concat([df, aligned_indicators], axis=1)
        return result.dropna()
