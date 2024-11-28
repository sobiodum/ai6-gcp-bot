# File: data_management/preprocessor.py
# Path: forex_trading_system/data_management/preprocessor.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class DataPreprocessor:
    """Handles data preprocessing including normalization and cleaning."""

    def __init__(self, window_size: int = 252):
        """
        Initialize preprocessor with window size for rolling calculations.

        Args:
            window_size: Number of periods for rolling calculations (default: 252 for ~1 year)
        """
        self.window_size = window_size

    def normalize_features(
        self,
        df: pd.DataFrame,
        features: List[str],
        method: str = 'robust',
        batch_size: int = 100_000
    ) -> pd.DataFrame:
        """
        Normalize features using specified method.

        Args:
            df: Input DataFrame
            features: List of columns to normalize
            method: Normalization method ('robust', 'zscore', or 'minmax')
        """
        df_normalized = df.copy()
        print(f"Normalizing features: {features}")
        for feature in features:
            if feature not in df.columns:
                continue

            try:
                if method == 'flo':
                    pct_cols = ['RSI', 'bb_percent']
                    if feature in pct_cols:
                        pass
                    else:
                        pass
                    pass
                for start_idx in range(0, len(df), batch_size):
                    end_idx = min(start_idx + batch_size, len(df))
                    batch = df.iloc[start_idx:end_idx]
                    window = min(self.window_size, len(batch))

                    if method == 'robust':
                        rolling_median = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).median()

                        def calc_iqr(x):
                            x = x[~np.isnan(x)]
                            if len(x) == 0:
                                return 0
                            return np.percentile(x, 75) - np.percentile(x, 25)

                        rolling_iqr = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).apply(calc_iqr)

                        rolling_iqr = rolling_iqr.replace(0, np.nan)
                        normalized_values = (
                            batch[feature] - rolling_median) / rolling_iqr

                    elif method == 'zscore':
                        rolling_mean = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).mean()

                        rolling_std = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).std()

                        rolling_std = rolling_std.replace(0, np.nan)
                        normalized_values = (
                            batch[feature] - rolling_mean) / rolling_std

                    elif method == 'minmax':
                        rolling_min = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).min()

                        rolling_max = batch[feature].rolling(
                            window=window,
                            min_periods=1
                        ).max()

                        denominator = (rolling_max - rolling_min)
                        denominator = denominator.replace(0, np.nan)
                        normalized_values = (
                            batch[feature] - rolling_min) / denominator

                    # Apply normalized values to the dataframe
                    df_normalized.iloc[start_idx:end_idx, df_normalized.columns.get_loc(feature)] = \
                        normalized_values.fillna(0)

                    print(
                        f"Processed batch {start_idx//batch_size + 1} for feature {feature}")

            except Exception as e:
                print(f"Error normalizing feature {feature}: {str(e)}")
                df_normalized[feature] = df[feature]

        return df_normalized

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and removing outliers.

        Strategy:
        - Remove rows with missing values for OHLCV data
        - Forward fill missing indicator values
        - Remove extreme outliers (>5 IQR)
        """
        df_cleaned = df.copy()

        # Remove rows with missing OHLCV data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        df_cleaned.dropna(subset=required_columns, inplace=True)

        # Forward fill missing indicator values
        indicator_columns = [col for col in df_cleaned.columns
                             if col not in required_columns]
        df_cleaned[indicator_columns] = df_cleaned[indicator_columns].fillna(
            method='ffill'
        )

        # Remove extreme outliers
        for col in df_cleaned.columns:
            Q1 = df_cleaned[col].quantile(0.25)
            Q3 = df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            df_cleaned = df_cleaned[
                ~((df_cleaned[col] < (Q1 - 5 * IQR)) |
                  (df_cleaned[col] > (Q3 + 5 * IQR)))
            ]

        return df_cleaned

    def normalize_simple(
        self,
        df: pd.DataFrame,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Normalize features using simple price-relative or percentage-based approach.
        """
        df_normalized = df.copy()

        # Add normalized close price
        start_price = df['close'].iloc[0]
        df_normalized['price_norm'] = df['close'] / start_price

        # Define indicator groups
        price_relative_indicators = [
            'sma_20', 'sma_50',
            'bb_upper', 'bb_middle', 'bb_lower',
            'senkou_span_a', 'senkou_span_b',
            'tenkan_sen', 'kijun_sen',
            'macd', 'macd_signal', 'macd_hist',
            'atr',
            'bb_bandwidth'  # Moved from percentage to price-relative
        ]

        percentage_indicators = [
            'rsi', 'bb_percent',  # bb_bandwidth removed from here
            'plus_di', 'minus_di', 'adx'
        ]

        # Normalize price-relative indicators
        if features is None:
            features = [col for col in df.columns
                        if col not in ['open', 'high', 'low', 'close', 'volume']]
        for feature in features:
            if feature in price_relative_indicators:
                df_normalized[feature] = df[feature] / df['close']
                print(f"Normalized {feature} by price")

            elif feature in percentage_indicators:
                df_normalized[feature] = df[feature] / 100.0
                print(f"Normalized {feature} by percentage")

            # Skip volume and other unspecified features
            elif feature not in ['open', 'high', 'low', 'close', 'volume']:
                print(f"Warning: No normalization rule for {feature}")

        # Drop OHLC columns
        # columns_to_drop = ['open', 'high', 'low', 'volume']
        # df_normalized.drop(columns=columns_to_drop, inplace=True)

        return df_normalized
