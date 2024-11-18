# File: data_management/data_quality.py
# Path: forex_trading_system/data_management/data_quality.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta


@dataclass
class QualityMetrics:
    """Container for data quality metrics."""
    missing_values: int
    gaps_detected: List[Tuple[datetime, datetime]]
    outliers_detected: Dict[str, int]
    stale_values: int
    data_freshness: timedelta
    statistics: Dict[str, Dict[str, float]]


class DataQualityMonitor:
    """Monitors and reports on data quality metrics."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the data quality monitor."""
        self.logger = logger or logging.getLogger(__name__)
        self.quality_thresholds = {
            'max_missing_pct': 0.01,  # Maximum 1% missing values
            'max_gap_hours': 24,      # Maximum gap size in hours
            # Max acceptable data age
            'staleness_threshold': timedelta(minutes=5),
            'outlier_std_threshold': 4.0  # Number of std devs for outlier detection
        }

    def check_data_quality(self, df: pd.DataFrame, timeframe: str) -> QualityMetrics:
        """
        Perform comprehensive data quality checks.

        Args:
            df: Input DataFrame with OHLCV and indicator data
            timeframe: Data timeframe (e.g., "1min", "5min", "15min", "1h")

        Returns:
            QualityMetrics object containing all quality measurements
        """
        metrics = QualityMetrics(
            missing_values=0,
            gaps_detected=[],
            outliers_detected={},
            stale_values=0,
            data_freshness=timedelta(0),
            statistics={}
        )

        # Check for missing values
        metrics.missing_values = df.isnull().sum().sum()
        missing_pct = metrics.missing_values / (df.shape[0] * df.shape[1])

        if missing_pct > self.quality_thresholds['max_missing_pct']:
            self.logger.warning(
                f"High number of missing values detected: {missing_pct:.2%}"
            )

        # Detect time series gaps
        metrics.gaps_detected = self._detect_gaps(df, timeframe)
        if metrics.gaps_detected:
            self.logger.warning(
                f"Found {len(metrics.gaps_detected)} gaps in time series"
            )

        # Check for outliers in each column
        metrics.outliers_detected = self._detect_outliers(df)
        for col, count in metrics.outliers_detected.items():
            if count > 0:
                self.logger.warning(
                    f"Found {count} outliers in column {col}"
                )

        # Check for stale values
        metrics.stale_values = self._detect_stale_values(df)
        if metrics.stale_values > 0:
            self.logger.warning(
                f"Found {metrics.stale_values} stale values"
            )

        # Calculate data freshness
        if not df.empty:
            metrics.data_freshness = datetime.now() - \
                df.index[-1].to_pydatetime()
            if metrics.data_freshness > self.quality_thresholds['staleness_threshold']:
                self.logger.warning(
                    f"Data is stale by {metrics.data_freshness}"
                )

        # Calculate basic statistics
        metrics.statistics = self._calculate_statistics(df)

        return metrics

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> List[Tuple[datetime, datetime]]:
        """Detect gaps in time series data."""
        if df.empty:
            return []

        # Define expected frequency based on timeframe
        freq_map = {
            "1min": "1T",
            "5min": "5T",
            "15min": "15T",
            "1h": "1H"
        }

        # Create continuous range of expected timestamps
        expected_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq_map[timeframe]
        )

        # Find missing timestamps
        missing_times = expected_range.difference(df.index)

        # Group consecutive missing timestamps into gaps
        gaps = []
        if len(missing_times) > 0:
            gap_start = missing_times[0]
            prev_time = gap_start

            for time in missing_times[1:]:
                if time - prev_time > pd.Timedelta(hours=self.quality_thresholds['max_gap_hours']):
                    gaps.append((gap_start, prev_time))
                    gap_start = time
                prev_time = time

            gaps.append((gap_start, prev_time))

        return gaps

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Detect outliers using rolling statistics.

        Uses a rolling window to calculate mean and std deviation,
        then identifies values outside threshold * std deviation.
        """
        outliers = {}

        for column in df.columns:
            if df[column].dtype.kind in 'iuf':  # Integer or float columns
                rolling_mean = df[column].rolling(window=20).mean()
                rolling_std = df[column].rolling(window=20).std()

                z_scores = np.abs((df[column] - rolling_mean) / rolling_std)
                outliers[column] = (
                    z_scores > self.quality_thresholds['outlier_std_threshold']
                ).sum()

        return outliers

    def _detect_stale_values(self, df: pd.DataFrame) -> int:
        """
        Detect stale values (unchanged over multiple periods).

        A value is considered stale if it remains exactly the same
        for more than 5 consecutive periods.
        """
        stale_count = 0

        for column in df.columns:
            if df[column].dtype.kind in 'iuf':
                # Count consecutive identical values
                consecutive_same = (
                    df[column] == df[column].shift(1)).astype(int)
                consecutive_same = consecutive_same.groupby(
                    (consecutive_same != consecutive_same.shift(1)).cumsum()
                ).cumsum()

                # Count values that remain same for more than 5 periods
                stale_count += (consecutive_same >= 5).sum()

        return stale_count

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate basic statistics for numeric columns."""
        stats = {}

        for column in df.columns:
            if df[column].dtype.kind in 'iuf':
                stats[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'kurtosis': df[column].kurtosis(),
                    'skewness': df[column].skew()
                }

        return stats

# Updating DatasetManager to include quality monitoring


def update_dataset_manager(self, df: pd.DataFrame, timeframe: str) -> Tuple[pd.DataFrame, QualityMetrics]:
    """
    Add this method to the DatasetManager class to integrate quality monitoring.
    """
    quality_monitor = DataQualityMonitor()
    quality_metrics = quality_monitor.check_data_quality(df, timeframe)

    if quality_metrics.missing_values > 0:
        self.logger.warning(
            f"Dataset contains {quality_metrics.missing_values} missing values"
        )

    if quality_metrics.gaps_detected:
        self.logger.warning(
            f"Found {len(quality_metrics.gaps_detected)} time gaps in data"
        )

    return df, quality_metrics
