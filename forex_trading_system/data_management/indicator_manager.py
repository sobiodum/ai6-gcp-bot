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

    def __init__(self, config_path: str, cache_dir: str):
        """
        Initialize indicator manager with configuration.

        Args:
            config_path: Path to indicator configuration file
            cache_dir: Directory for caching calculated indicators
        """
        self.config_path = config_path
        self.cache_dir = cache_dir
        # If config_path is a directory, append indicators.yaml
        if os.path.isdir(config_path):
            self.config_path = os.path.join(config_path, "indicators.yaml")
        else:
            self.config_path = config_path

        # Try relative to project root if not found
        if not os.path.exists(self.config_path):
            project_root = Path(__file__).parent.parent
            self.config_path = os.path.join(
                project_root, "config", "indicators.yaml")

        self.cache_dir = cache_dir
        self.load_config()

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

    def load_config(self):
        """Load indicator configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.indicators = {
                    name: IndicatorConfig(**params)
                    for name, params in config['indicators'].items()
                }
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find indicators.yaml in {self.config_path}. Current working directory: {os.getcwd()}")

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Calculate all enabled technical indicators.

        Args:
            df: Input DataFrame with OHLCV data
            use_cache: Whether to use cached indicators
        """
        df_with_indicators = df.copy()
        daily_df = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })

        daily_with_indicators = daily_df.copy()

        # Add SMA calculations
        if self.indicators['sma'].enabled:
            for period in self.indicators['sma'].params['periods']:
                daily_with_indicators[f'sma_{period}'] = talib.SMA(
                    daily_with_indicators['close'],
                    timeperiod=period
                )

        # Calculate each enabled indicator
        if self.indicators['ichimoku'].enabled:
            self._add_ichimoku(daily_with_indicators,
                               self.indicators['ichimoku'].params)

        if self.indicators['rsi'].enabled:
            daily_with_indicators['rsi'] = talib.RSI(
                daily_with_indicators['close'],
                timeperiod=self.indicators['rsi'].params['period']
            )

        if self.indicators['macd'].enabled:
            macd, signal, hist = talib.MACD(
                daily_with_indicators['close'],
                **self.indicators['macd'].params
            )
            daily_with_indicators['macd'] = macd
            daily_with_indicators['macd_signal'] = signal
            daily_with_indicators['macd_hist'] = hist

        if self.indicators['bollinger'].enabled:
            upper, middle, lower = talib.BBANDS(
                daily_with_indicators['close'],
                **self.indicators['bollinger'].params
            )
            daily_with_indicators['bb_upper'] = upper
            daily_with_indicators['bb_middle'] = middle
            daily_with_indicators['bb_lower'] = lower

        # Add other indicators (ATR, DMI, ADX)
        if self.indicators['atr'].enabled:
            daily_with_indicators['atr'] = talib.ATR(
                daily_with_indicators['high'],
                daily_with_indicators['low'],
                daily_with_indicators['close'],
                timeperiod=self.indicators['atr'].params['period']
            )

        if self.indicators['dmi'].enabled:
            daily_with_indicators['plus_di'] = talib.PLUS_DI(
                daily_with_indicators['high'],
                daily_with_indicators['low'],
                daily_with_indicators['close'],
                timeperiod=self.indicators['dmi'].params['period']
            )
            daily_with_indicators['minus_di'] = talib.MINUS_DI(
                daily_with_indicators['high'],
                daily_with_indicators['low'],
                daily_with_indicators['close'],
                timeperiod=self.indicators['dmi'].params['period']
            )

        if self.indicators['adx'].enabled:
            daily_with_indicators['adx'] = talib.ADX(
                daily_with_indicators['high'],
                daily_with_indicators['low'],
                daily_with_indicators['close'],
                timeperiod=self.indicators['adx'].params['period']
            )
        # Copy daily indicators back to original frequency
        indicator_columns = [col for col in daily_with_indicators.columns
                             if col not in ['open', 'high', 'low', 'close', 'volume']]
        # Forward fill daily values to original frequency
        df_with_indicators = df.copy()
        for col in indicator_columns:
            df_with_indicators[col] = daily_with_indicators[col].reindex(
                df_with_indicators.index, method='ffill'
            )

        return df_with_indicators

    # def _add_ichimoku(self, df: pd.DataFrame, params: Dict) -> None:
    #     """Calculate Ichimoku Cloud indicators."""
    #     high = df['high']
    #     low = df['low']

    #     # Calculate Tenkan-sen (Conversion Line)
    #     period9_high = high.rolling(
    #         window=params['conversion_line_period']).max()
    #     period9_low = low.rolling(
    #         window=params['conversion_line_period']).min()
    #     df['tenkan_sen'] = (period9_high + period9_low) / 2

    #     # Calculate Kijun-sen (Base Line)
    #     period26_high = high.rolling(window=params['base_line_period']).max()
    #     period26_low = low.rolling(window=params['base_line_period']).min()
    #     df['kijun_sen'] = (period26_high + period26_low) / 2

    #     # Calculate Senkou Span A (Leading Span A)
    #     df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(
    #         params['displacement']
    #     )

    #     # Calculate Senkou Span B (Leading Span B)
    #     period52_high = high.rolling(
    #         window=params['lagging_span_period']).max()
    #     period52_low = low.rolling(window=params['lagging_span_period']).min()
    #     df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(
    #         params['displacement']
    #     )

    #     # Calculate Chikou Span (Lagging Span)
    #     df['chikou_span'] = df['close'].shift(-params['displacement'])

    def _add_ichimoku(self, df: pd.DataFrame, params: Dict) -> None:
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
