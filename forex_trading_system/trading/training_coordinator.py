from typing import List, Optional, Dict
import multiprocessing
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from pathlib import Path


class TrainingCoordinator:
    """Coordinates and monitors training across multiple currency pairs."""

    def __init__(self,
                 max_concurrent: Optional[int] = None,
                 memory_warning_threshold: float = 0.85,
                 memory_critical_threshold: float = 0.95,
                 sequence_length: int = 10,
                 base_path: str = "/Volumes/ssd_fat2/ai6_trading_bot/datasets"):
        """
        Initialize training coordinator with resource monitoring.

        Args:
            max_concurrent: Maximum number of pairs to train simultaneously
            memory_warning_threshold: Log warning if memory usage exceeds this (85%)
            memory_critical_threshold: Log critical if memory usage exceeds this (95%)
            sequence_length: Number of time steps in each training sequence
            base_path: Path to dataset storage
        """
        self.max_concurrent = max_concurrent or max(
            1, multiprocessing.cpu_count() - 1)
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        self.sequence_length = sequence_length
        self.base_path = Path(base_path)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Track active training sessions
        self.active_trainers: Dict[str, Dict] = {}

    def prepare_training_data(self, pair: str, timeframe: str = "1min") -> pd.DataFrame:
        """
        Prepare data for training including sequence creation and spread calculation.

        Args:
            pair: Currency pair symbol
            timeframe: Data timeframe

        Returns:
            DataFrame with prepared sequences and features
        """
        # Load raw data
        file_path = self.base_path / timeframe / f"{pair}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for {pair} at {file_path}")

        df = pd.read_parquet(file_path)

        # Calculate spread (assuming typical spread values for major pairs)
        typical_spreads = {
            'EUR_USD': 0.00001,  # 0.1 pip
            'GBP_USD': 0.00002,  # 0.2 pip
            'USD_JPY': 0.009,    # 0.9 pip
            # Add more pairs as needed
        }

        base_spread = typical_spreads.get(pair, 0.00002)  # Default to 0.2 pip

        # Add spread as a feature (vary it slightly based on volatility)
        df['volatility'] = df['high'] - df['low']
        df['spread'] = base_spread * \
            (1 + (df['volatility'] / df['close']).rolling(20).std())

        # Create sequences
        sequences = []
        labels = []

        for i in range(len(df) - self.sequence_length):
            sequence = df.iloc[i:i + self.sequence_length]
            sequences.append(sequence)
            # Label will be determined by environment later
            labels.append(df.iloc[i + self.sequence_length])

        return pd.DataFrame({
            'sequence': sequences,
            'label': labels
        })

    def train_pairs(self, pairs: List[str], parallel: bool = True) -> None:
        """
        Train models for specified currency pairs.

        Args:
            pairs: List of currency pairs to train
            parallel: If True, train pairs in parallel up to max_concurrent
        """
        self.logger.info(
            f"Starting training for {len(pairs)} pairs. Parallel={parallel}")

        if parallel:
            self._train_parallel(pairs)
        else:
            self._train_sequential(pairs)

    def _train_parallel(self, pairs: List[str]) -> None:
        """Train multiple pairs in parallel with resource monitoring."""
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_pair = {
                executor.submit(self._train_single_pair, pair): pair
                for pair in pairs
            }

            for future in concurrent.futures.as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Training failed for {pair}: {str(e)}")

    def _train_sequential(self, pairs: List[str]) -> None:
        """Train pairs one at a time."""
        for pair in pairs:
            try:
                self._train_single_pair(pair)
            except Exception as e:
                self.logger.error(f"Training failed for {pair}: {str(e)}")
                continue

    def _check_resources(self) -> None:
        """Monitor system resources and log warnings if thresholds exceeded."""
        memory_usage = psutil.virtual_memory().percent / 100
        cpu_usage = psutil.cpu_percent() / 100

        if memory_usage > self.memory_critical_threshold:
            self.logger.critical(
                f"Critical memory usage: {memory_usage:.1%}. Performance may be severely impacted."
            )
        elif memory_usage > self.memory_warning_threshold:
            self.logger.warning(
                f"High memory usage: {memory_usage:.1%}"
            )

        if cpu_usage > 0.9:  # 90% CPU usage
            self.logger.warning(
                f"High CPU usage: {cpu_usage:.1%}"
            )

    def _train_single_pair(self, pair: str) -> None:
        """
        Train model for a single currency pair.

        Args:
            pair: Currency pair to train
        """
        self.logger.info(f"Starting training for {pair}")
        self._check_resources()

        try:
            # Prepare training data
            train_data = self.prepare_training_data(pair)

            # Track training session
            self.active_trainers[pair] = {
                'start_time': pd.Timestamp.now(),
                'status': 'training',
                'sequences': len(train_data)
            }

            # TODO: Actual training logic will be implemented when we add the model

            self.active_trainers[pair]['status'] = 'completed'
            self.logger.info(f"Completed training for {pair}")

        except Exception as e:
            self.active_trainers[pair] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
        finally:
            self._check_resources()

    def get_training_status(self) -> Dict[str, Dict]:
        """Get status of all training sessions."""
        return self.active_trainers
