from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


AVAILABLE_COLUMNS_IN_DF = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi', 'macd',
                           'macd_signal', 'macd_hist', 'roc_10', 'stoch_rsi', 'stoch_k', 'stoch_d',
                           'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent',
                           'atr', 'plus_di', 'minus_di', 'adx', 'senkou_span_a', 'senkou_span_b',
                           'tenkan_sen', 'kijun_sen']

PRIORITY_FEATURES = {
    'immediate_state': [
        'dist_sma_20_1D',    # Short-term trend
        'dist_sma_50_1D',     # Medium-term trend
        'rsi_5min',            # Short-term momentum
        'bb_position_5min',    # Volatility context
        'atr_pct_5min'         # Current volatility
    ],
    'sequential': [
        'close_norm>',       # Price divided by the first close
        'dist_sma_20_change',  # Trend changes
        'rsi_change',          # Momentum changes

    ]
}


logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature calculation settings."""
    normalize_windows: List[int] = field(default_factory=lambda: [20, 50, 100])
    momentum_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20])
    trend_windows: List[int] = field(default_factory=lambda: [20, 50])


class FeatureEngineer:
    """
    Enhanced feature engineering for forex RL agent.
    Focuses on creating meaningful representations of technical indicators
    while avoiding any forward-looking bias.
    """

    def __init__(
        self,
        config: Optional[FeatureConfig] = None,
        indicator_cols: Optional[List[str]] = None
    ):
        """
        Initialize feature engineer with configuration.

        Args:
            config: Feature calculation settings
            indicator_cols: List of indicator columns to process
        """
        self.config = config or FeatureConfig()
        self.indicator_cols = indicator_cols or AVAILABLE_COLUMNS_IN_DF

        # Verify indicators exist
        self._validate_indicators()

    def _validate_indicators(self):
        """Validate that required indicators are available."""
        required_base = ['open', 'high', 'low', 'close']
        if not all(col in self.indicator_cols for col in required_base):
            raise ValueError(f"Missing required base columns: {required_base}")

    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create enhanced features from existing indicators.
        Ensures no forward-looking bias is introduced.

        Args:
            df: DataFrame with OHLCV and technical indicators

        Returns:
            DataFrame with enhanced features
        """
        enhanced_df = df.copy()

        # 1. Price-based features (avoiding look-ahead bias)
        self._add_price_features(enhanced_df)

        # 2. Distance-based features
        self._add_distance_features(enhanced_df)

        # 3. Momentum enhancements
        self._enhance_momentum_features(enhanced_df)

        # 4. Volatility context
        self._add_volatility_context(enhanced_df)

        # 5. Trend strength indicators
        self._add_trend_features(enhanced_df)

        self._add_vol_adjusted_features(enhanced_df)

        #! Remove any NaN values that might have been introduced
        # enhanced_df.fillna(0, inplace=True)
        enhanced_df.dropna(inplace=True)

        return enhanced_df

    def _add_price_features(self, df: pd.DataFrame):
        """Add normalized price relationship features."""
        # GOOD Price change rate (no forward looking)
        df['price_change'] = df['close'].pct_change()

        # Normalize close price by first value
        df['close_norm'] = df['close'] / df['close'].iloc[0]

        # GOOG says Claude High-Low range relative to close
        df['hl_range_ratio'] = (df['high'] - df['low']) / df['close']

    def _add_distance_features(self, df: pd.DataFrame):
        """
        Calculate distances between indicators and current price.
        All calculations use only past data.
        """
        # Good but needs ATR normalization instead of price
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            # First normalize ATR by price level
            df['atr_normalized'] = df['atr'] / df['close']
            # Use normalized ATR for distance calculations
            df['dist_sma_20_atr_adj'] = (
                df['close'] - df['sma_20']) / df['atr']
            df['dist_sma_50_atr_adj'] = (
                df['close'] - df['sma_50']) / df['atr']

            # Not very useful in raw form
            # df['sma_cross'] = df['sma_20'] - df['sma_50']
            # df['sma_cross_normalized'] = df['sma_cross'] / df['close']
            # Better version:
            df['sma_cross_atr'] = (
                df['sma_20'] - df['sma_50']) / df['atr']

        # Distance to Bollinger Bands
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            df['bb_position'] = (df['close'] - df['bb_lower']) / \
                (df['bb_upper'] - df['bb_lower'])
            #! we already have bb_bandwidth which only needs to be normalized
            df['bb_width_normalized'] = (
                df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Ichimoku distances if available
        if 'tenkan_sen' in df.columns and 'kijun_sen' in df.columns:
            #! Same as with SMA - normalize by atr / close --> col might neet to be dropped
            df['atr_normalized'] = df['atr'] / df['close']
            df['tenkan_dist_atr_adj'] = (
                df['close'] - df['tenkan_sen']) / df['atr']
            df['kijun_dist_atr_adj'] = (
                df['close'] - df['kijun_sen']) / df['atr']

    def _enhance_momentum_features(self, df: pd.DataFrame):
        """Enhance momentum indicators with additional context."""
        if 'rsi' in df.columns:
            # RSI changes and extremes
            df['rsi_change'] = df['rsi'].diff()

        if all(col in df.columns for col in ['macd', 'macd_signal']):
            # Good but use ATR normalization
            df['macd_hist_norm'] = (
                df['macd'] - df['macd_signal']) / df['atr']

        if 'adx' in df.columns and 'plus_di' in df.columns and 'minus_di' in df.columns:
            # Enhanced directional movement
            df['di_spread'] = df['plus_di'] - df['minus_di']
            df['trend_strength'] = df['adx'] * np.sign(df['di_spread'])

    def _add_volatility_context(self, df: pd.DataFrame):
        """Add volatility context features."""
        if 'atr' in df.columns:
            # Normalize ATR by price level
            df['atr_normalized'] = df['atr'] / df['close']

            # ATR-based volatility regime
            for window in self.config.volatility_windows:
                atr_ma = df['atr'].rolling(window=window).mean()
                df[f'atr_regime_{window}'] = df['atr'] / atr_ma

    def _add_trend_features(self, df: pd.DataFrame):
        """Add trend strength and consistency features."""
        # Price momentum relative to volatility
        if 'atr' in df.columns:
            for window in self.config.trend_windows:
                price_change = df['close'].diff(window)
                df[f'trend_strength_{window}'] = price_change / \
                    (df['atr'] * np.sqrt(window))

    def _add_vol_adjusted_features(self, df: pd.DataFrame):
        # Short-term momentum
        df['momentum_1h'] = (
            df['close'].diff(12)  # 12 5-min bars = 1 hour
            / (df['atr'] * np.sqrt(12))
        )

        # Medium-term momentum
        df['momentum_4h'] = (
            df['close'].diff(48)  # 48 5-min bars = 4 hours
            / (df['atr'] * np.sqrt(48))
        )

    @staticmethod
    def remove_lookback_period(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Remove initial lookback period where features might be unreliable.

        Args:
            df: DataFrame with calculated features
            lookback: Number of periods to remove

        Returns:
            DataFrame with lookback period removed
        """
        return df.iloc[lookback:]


##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
##############################################################
