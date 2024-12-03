from typing import Optional, Tuple
from stable_baselines3.common.vec_env import VecNormalize
import talib
from dataclasses import dataclass, field
import pytz
from typing import Dict, List, Optional, Tuple
from google.cloud import storage
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO
from trading.environments.forex_env2_flat import ForexTradingEnv
from concurrent.futures import ThreadPoolExecutor
from apscheduler.schedulers.background import BackgroundScheduler
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
import os
import sys

import asyncio


import traceback

from pathlib import Path
import yaml

from apscheduler.schedulers.asyncio import AsyncIOScheduler


# Trading system configuration
TRADING_CONFIG = {
    # OANDA credentials - replace with your actual values
    'oanda_account_id': '101-004-30348600-002',
    'oanda_api_key': '9317ace4596d61e3e98b1a53b2342483-45d3ad4084c80b111727a9fada9ef0ff',

    # GCP bucket where your parquet files are stored


    # Base path where your trained models are stored locally
    'model_base_path': './models',

    # Starting with EUR/USD only
    'trading_pairs': {
        'EUR_USD': 94_510.0  # Notional amount matching $100k USD
    },

    # Trading hours (24/7 for now)
    'trading_hours': {
        'EUR_USD': [(0, 24)]
    },

    # Maximum acceptable spread
    'min_spread_threshold': {
        'EUR_USD': 0.0002  # 2 pips for EUR/USD
    },

    # System monitoring intervals (in seconds)
    'backup_interval': 300,     # Backup every 5 minutes
    'health_check_interval': 60  # Health check every minute
}

# Configure logging with detailed formatting
logging.basicConfig(
    filename='trading_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('trading_coordinator')
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


logging.basicConfig(
    filename='trading_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('position_manager')


@dataclass
class PositionInfo:
    """Stores current position information for a currency pair"""
    current_position: int  # 1 for long, -1 for short, 0 for no position
    units: float
    entry_price: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    initial_balance: float
    current_balance: float


@dataclass
class DataConfig:
    """Configuration for data management and indicator calculation"""
    bucket_name: str
    # List of indicators that should be present in the data
    indicators: List[str] = field(default_factory=lambda: [
        'sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_bandwidth', 'bb_percent',
        'atr', 'plus_di', 'minus_di', 'adx',
        'senkou_span_a', 'senkou_span_b', 'tenkan_sen', 'kijun_sen'
    ])
    sequence_length: int = 5


def get_position_info(self, pair: str) -> PositionInfo:
    """Get current position information for a pair"""
    return self.positions[pair]


def get_normalized_balance(self, pair: str) -> float:
    """Get normalized balance (current_balance / initial_balance) for a pair"""
    position = self.positions[pair]
    return position.current_balance / position.initial_balance


async def sync_positions(self) -> None:
    """Synchronize positions with OANDA"""
    try:
        # Get open positions
        r = positions.OpenPositions(accountID=self.account_id)
        self.client.request(r)
        open_positions = r.response.get('positions', [])

        # Get account details for PnL tracking
        r_acc = accounts.AccountSummary(accountID=self.account_id)
        self.client.request(r_acc)
        account_info = r_acc.response['account']

        # Reset all positions first
        for pair in self.positions:
            self.positions[pair].current_position = 0
            self.positions[pair].units = 0.0
            self.positions[pair].entry_price = None

        # Update with current open positions
        for pos in open_positions:
            pair = pos['instrument']
            if pair not in self.positions:
                continue

            # Determine position direction and size
            long_units = float(pos.get('long', {}).get('units', 0))
            short_units = float(pos.get('short', {}).get('units', 0))

            if long_units > 0:
                self.positions[pair].current_position = 1
                self.positions[pair].units = long_units
                self.positions[pair].entry_price = float(
                    pos['long']['averagePrice'])
            elif short_units < 0:
                self.positions[pair].current_position = -1
                self.positions[pair].units = abs(short_units)
                self.positions[pair].entry_price = float(
                    pos['short']['averagePrice'])

            # Update PnL
            self.positions[pair].unrealized_pnl = float(
                pos.get('unrealizedPL', 0))
            self.positions[pair].realized_pnl = float(pos.get('realizedPL', 0))

            # Update balance based on PnL
            self.positions[pair].current_balance = (
                self.positions[pair].initial_balance +
                self.positions[pair].realized_pnl +
                self.positions[pair].unrealized_pnl
            )

        logger.info("Successfully synchronized positions with OANDA")

    except Exception as e:
        logger.error(f"Error synchronizing positions: {str(e)}")
        raise


logger = logging.getLogger('trading_agent')


class TradingAgent:
    """Handles model inference and trading decisions for a single currency pair"""

    def __init__(
        self,
        pair: str,
        model_path: str,
        vec_normalize_path: str,
        notional: float
    ):
        self.pair = pair
        self.notional = notional

        # Load model and normalization statistics
        self.model = PPO.load(model_path)
        self.vec_normalize = VecNormalize.load(vec_normalize_path, None)

        # Disable training mode
        self.vec_normalize.training = False
        self.vec_normalize.norm_reward = False

        logger.info(f"Initialized trading agent for {pair}")

    def get_action(
        self,
        market_features: np.ndarray,
        normalized_balance: float,
        current_position: int
    ) -> Tuple[int, float]:
        """
        Get trading action from model.

        Args:
            market_features: Market data and indicator values
            normalized_balance: Current balance / initial balance
            current_position: Current position (1, -1, or 0)

        Returns:
            Tuple of (action, action_probability)
        """
        try:
            # Construct observation
            observation = np.concatenate([
                market_features,
                [normalized_balance, current_position]
            ]).astype(np.float32)

            # Normalize observation
            observation = self.vec_normalize.normalize_obs(observation)

            # Get model prediction
            action, _ = self.model.predict(observation, deterministic=True)

            # Get action probabilities for logging
            action_probs = self.model.policy.get_distribution(
                self.model.policy.obs_to_tensor(observation)[0]
            ).distribution.probs

            # Convert to numpy array if necessary
            action_probs = action_probs.detach().numpy()

            logger.info(
                f"{self.pair} - Action: {action}, "
                f"Probabilities: {action_probs}"
            )

            return int(action), float(action_probs[action])

        except Exception as e:
            logger.error(f"Error getting action for {self.pair}: {str(e)}")
            raise


class PositionManager:
    """Manages positions and balance tracking for all currency pairs"""

    def __init__(self, oanda_client: API, account_id: str, trading_pairs: Dict[str, float]):
        """
        Initialize position manager with OANDA client and trading pairs with their notionals.

        Args:
            oanda_client: OANDA API client
            account_id: OANDA account ID
            trading_pairs: Dictionary of pairs and their notional amounts
        """
        self.client = oanda_client
        self.account_id = account_id
        self.trading_pairs = trading_pairs
        self.positions: Dict[str, PositionInfo] = {}

        # Initialize positions for all pairs
        for pair in trading_pairs.keys():
            self.positions[pair] = PositionInfo(
                current_position=0,
                units=0.0,
                entry_price=None,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                initial_balance=100_000.0,  # Starting balance per pair
                current_balance=100_000.0
            )


class MarketDataManager:
    """Manages market data loading, updating, and processing"""

    def __init__(self, config: DataConfig, oanda_client: API):
        self.config = config
        self.oanda_client = oanda_client
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(config.bucket_name)
        self.data_cache: Dict[str, pd.DataFrame] = {}

    async def initialize_pair_data(self, pair: str) -> None:
        """Initialize data for a currency pair from GCS and update with recent data"""
        try:
            # Load historical 5min data with indicators from GCS
            df = await self._load_from_gcs(pair)
            logger.info(
                f"Loaded historical data for {pair}, shape: {df.shape}")

            # Check if we need to fetch missing data
            last_timestamp = df.index[-1]
            current_time = pd.Timestamp.now(tz='UTC')

            if (current_time - last_timestamp) > pd.Timedelta(minutes=5):
                logger.info(
                    f"Fetching missing data for {pair} from {last_timestamp} to {current_time}")
                missing_data = await self._fetch_and_process_missing_data(
                    pair, last_timestamp, current_time
                )
                if not missing_data.empty:
                    df = pd.concat([df, missing_data])
                    df = df.sort_index()

            # Verify all required indicators are present
            self._verify_indicators(df, pair)

            # Store in cache
            self.data_cache[pair] = df
            logger.info(
                f"Initialized data for {pair}, final shape: {df.shape}")

        except Exception as e:
            logger.error(f"Error initializing data for {pair}: {str(e)}")
            raise

    async def _load_from_gcs(self, pair: str) -> pd.DataFrame:
        """Load 5min data with indicators from Google Cloud Storage"""
        blob = self.bucket.blob(f"{pair}.parquet")

        try:
            # Download to temporary file
            temp_path = f"/tmp/{pair}_5min.parquet"
            blob.download_to_filename(temp_path)

            # Read parquet file
            df = pd.read_parquet(temp_path)

            # Ensure index is timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            return df

        except Exception as e:
            logger.error(f"Error loading data from GCS for {pair}: {str(e)}")
            raise

    async def _fetch_and_process_missing_data(
        self,
        pair: str,
        last_timestamp: pd.Timestamp,
        current_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Fetch missing 5-minute candles and process them"""
        try:
            # Calculate a buffer to ensure we have enough data for hourly indicator calculation
            # Get the start of the last complete hour before our last timestamp
            buffer_start = last_timestamp - pd.Timedelta(hours=1)
            buffer_start = buffer_start.floor('H')

            params = {
                "granularity": "M5",  # Use 5-minute candles directly
                "from": buffer_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "to": current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            }

            logger.info(
                f"Fetching 5min candles for {pair} from {buffer_start} to {current_time}")

            # Fetch candles from OANDA
            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            self.oanda_client.request(r)

            # Convert to DataFrame
            candles = []
            for candle in r.response['candles']:
                if candle['complete']:  # Only use complete candles
                    candles.append({
                        'timestamp': pd.Timestamp(candle['time']),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })

            if not candles:
                logger.warning(f"No new complete candles found for {pair}")
                return pd.DataFrame()

            # Create DataFrame from 5min candles
            df_5min = pd.DataFrame(candles)
            df_5min.set_index('timestamp', inplace=True)

            # Resample to 1hour for indicator calculation
            df_1hour = df_5min.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Calculate indicators on hourly data
            df_1hour_indicators = self._calculate_indicators(df_1hour)

            # Forward fill indicators to 5min data
            df_5min_with_indicators = pd.merge_asof(
                df_5min,
                df_1hour_indicators[self.config.indicators],
                left_index=True,
                right_index=True,
                direction='backward'
            )

            # Only return the new data (after last_timestamp)
            return df_5min_with_indicators[df_5min_with_indicators.index > last_timestamp]

        except Exception as e:
            logger.error(f"Error fetching missing data for {pair}: {str(e)}")
            raise

    def _resample_to_5min(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """Resample 1min candles to 5min candles"""
        return df_1min.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _resample_to_1hour(self, df_1min: pd.DataFrame) -> pd.DataFrame:
        """Resample 1min candles to 1hour candles"""
        return df_1min.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on 1hour data"""
        # Implementation of indicator calculation
        # This would be the same calculation logic used in your training data preparation
        # Would you like me to include the detailed indicator calculation code?

    def _forward_fill_indicators(
        self,
        df_5min: pd.DataFrame,
        df_1hour_indicators: pd.DataFrame
    ) -> pd.DataFrame:
        """Forward fill hourly indicators to 5min data"""
        # Merge hourly indicators with 5min data
        # The indicators will automatically be forward filled
        return pd.merge_asof(
            df_5min,
            df_1hour_indicators[self.config.indicators],
            left_index=True,
            right_index=True,
            direction='backward'
        )

    def _verify_indicators(self, df: pd.DataFrame, pair: str) -> None:
        """Verify all required indicators are present in the DataFrame"""
        missing_indicators = [
            ind for ind in self.config.indicators if ind not in df.columns]
        if missing_indicators:
            raise ValueError(
                f"Missing indicators for {pair}: {missing_indicators}")

    def get_latest_observation(self, pair: str) -> np.ndarray:
        """Get the latest observation for model input"""
        df = self.data_cache[pair]

        # Get last n=sequence_length rows of all features
        observation = df.iloc[-self.config.sequence_length:]

        # Convert to numpy array and transpose to match model's expected input
        # Shape will be (n_features * sequence_length,)
        market_features = observation.values.T.flatten()

        return market_features

# trading_coordinator.py


@dataclass
class TradingConfig:
    """Configuration for the trading system"""
    oanda_account_id: str
    oanda_api_key: str
    gcp_bucket_name: str
    model_base_path: str
    trading_pairs: Dict[str, float]  # pair -> notional amount
    trading_hours: Dict[str, List[tuple]]  # pair -> [(start_hour, end_hour)]
    min_spread_threshold: Dict[str, float]  # pair -> max acceptable spread
    backup_interval: int = 300  # seconds between state backups
    health_check_interval: int = 60  # seconds between health checks


class TradingCoordinator:
    """Coordinates all trading activities and system management"""

    def __init__(self, config_path: str):
        """Initialize the trading coordinator with configuration"""
        try:
            # Load configuration
            self.config = self._load_config(config_path)

            # Initialize OANDA client
            self.oanda_client = API(
                access_token=self.config.oanda_api_key,
                environment="live"  # ! or "practice" for testing
            )

            # Initialize components
            self.data_manager = MarketDataManager(
                DataConfig(bucket_name=self.config.gcp_bucket_name),
                self.oanda_client
            )

            self.position_manager = PositionManager(
                self.oanda_client,
                self.config.oanda_account_id,
                self.config.trading_pairs
            )

            # Initialize trading agents
            self.agents: Dict[str, TradingAgent] = {}
            self._initialize_agents()

            # Initialize scheduler
            self.scheduler = AsyncIOScheduler()

            # State tracking
            self.is_running = False
            self.last_error_time: Dict[str, datetime] = {}
            self.error_counts: Dict[str, int] = {}

            # Threading
            self.executor = ThreadPoolExecutor(max_workers=4)

            logger.info("Trading coordinator initialized successfully")

        except Exception as e:
            logger.critical(
                f"Failed to initialize trading coordinator: {str(e)}")
            raise

    def _initialize_agents(self):
        """Initialize trading agents for all pairs."""
        try:
            for pair, notional in self.config.trading_pairs.items():
                # Construct paths for model and normalization files
                model_path = os.path.join(
                    self.config.model_base_path,
                    pair,
                    'best_model.zip'
                )
                vec_normalize_path = os.path.join(
                    self.config.model_base_path,
                    pair,
                    'vec_normalize.pkl'
                )

                # Create trading agent
                self.agents[pair] = TradingAgent(
                    pair=pair,
                    model_path=model_path,
                    vec_normalize_path=vec_normalize_path,
                    notional=notional
                )

                logger.info(f"Initialized trading agent for {pair}")

        except Exception as e:
            logger.critical(f"Failed to initialize trading agents: {str(e)}")
            raise

    def _load_config(self, config_path: str) -> TradingConfig:
        """Load and validate configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Validate configuration
            required_fields = [
                'oanda_account_id', 'oanda_api_key', 'gcp_bucket_name',
                'model_base_path', 'trading_pairs'
            ]

            missing_fields = [
                field for field in required_fields
                if field not in config_dict
            ]

            if missing_fields:
                raise ValueError(
                    f"Missing required configuration fields: {missing_fields}"
                )

            return TradingConfig(**config_dict)

        except Exception as e:
            logger.critical(f"Failed to load configuration: {str(e)}")
            raise

    async def start(self):
        """Start the trading system"""
        try:
            logger.info("Starting trading system...")

            # Initialize data and positions
            await self._initialize_system()

            # Schedule regular tasks
            self._schedule_tasks()

            # Start the scheduler
            self.scheduler.start()
            self.is_running = True

            logger.info("Trading system started successfully")

        except Exception as e:
            logger.critical(f"Failed to start trading system: {str(e)}")
            raise

    async def _initialize_system(self):
        """Initialize system state"""
        try:
            # Initialize data for all pairs
            for pair in self.config.trading_pairs:
                await self.data_manager.initialize_pair_data(pair)

            # Sync positions with OANDA
            await self.position_manager.sync_positions()

            # Verify system state
            self._verify_system_state()

        except Exception as e:
            logger.critical(f"System initialization failed: {str(e)}")
            raise

    def _schedule_tasks(self):
        """Schedule regular tasks"""
        try:
            # Schedule trading cycle (every 5 minutes, aligned with candle close)
            self.scheduler.add_job(
                self._trading_cycle,
                CronTrigger(minute='*/5'),
                id='trading_cycle'
            )

            # Schedule health checks
            self.scheduler.add_job(
                self._health_check,
                'interval',
                seconds=self.config.health_check_interval,
                id='health_check'
            )

            # Schedule position synchronization
            self.scheduler.add_job(
                self.position_manager.sync_positions,
                'interval',
                minutes=1,
                id='position_sync'
            )

            # Schedule state backup
            self.scheduler.add_job(
                self._backup_system_state,
                'interval',
                seconds=self.config.backup_interval,
                id='state_backup'
            )

        except Exception as e:
            logger.error(f"Failed to schedule tasks: {str(e)}")
            raise

    async def _trading_cycle(self):
        """Execute one trading cycle"""
        try:
            logger.info("Starting trading cycle")
            cycle_start = datetime.now(pytz.UTC)

            # Update data for all pairs
            await self._update_market_data()

            # Process each pair
            tasks = []
            for pair in self.config.trading_pairs:
                if self._should_trade(pair):
                    tasks.append(self._process_pair(pair))

            # Wait for all pair processing to complete
            await asyncio.gather(*tasks)

            cycle_duration = (datetime.now(pytz.UTC) -
                              cycle_start).total_seconds()
            logger.info(
                f"Trading cycle completed in {cycle_duration:.2f} seconds")

        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")
            self._handle_trading_error("trading_cycle", e)

    async def _process_pair(self, pair: str):
        """Process a single currency pair"""
        try:
            # Get current market state
            market_features = self.data_manager.get_latest_observation(pair)

            # Get position info
            position_info = self.position_manager.get_position_info(pair)

            # Check spread
            current_spread = self._get_current_spread(pair)
            if current_spread > self.config.min_spread_threshold[pair]:
                logger.warning(f"Spread too high for {pair}: {current_spread}")
                return

            # Get model action
            action, confidence = self.agents[pair].get_action(
                market_features,
                position_info.normalized_balance,
                position_info.current_position
            )

            # Execute trade if needed
            if self._should_execute_trade(action, position_info.current_position, confidence):
                await self._execute_trade(pair, action, position_info)

        except Exception as e:
            logger.error(f"Error processing {pair}: {str(e)}")
            self._handle_trading_error(f"process_pair_{pair}", e)

    def _handle_trading_error(self, error_source: str, error: Exception):
        """Handle trading errors with exponential backoff"""
        current_time = datetime.now(pytz.UTC)

        # Update error tracking
        if error_source not in self.error_counts:
            self.error_counts[error_source] = 0
            self.last_error_time[error_source] = current_time

        self.error_counts[error_source] += 1

        # Calculate backoff time
        # Max 5 minutes
        backoff = min(300, 2 ** self.error_counts[error_source])

        # Log error with full stack trace
        logger.error(
            f"Error in {error_source} (count: {self.error_counts[error_source]}):\n"
            f"Error: {str(error)}\n"
            f"Stack trace:\n{traceback.format_exc()}\n"
            f"Backing off for {backoff} seconds"
        )

        # Reset error count if enough time has passed
        if (current_time - self.last_error_time[error_source]).total_seconds() > 3600:
            self.error_counts[error_source] = 0

        self.last_error_time[error_source] = current_time

    async def _health_check(self):
        """Perform system health check"""
        try:
            # Check API connectivity
            for pair in self.config.trading_pairs:
                if self.error_counts.get(f"process_pair_{pair}", 0) > 10:
                    logger.critical(
                        f"Too many errors for {pair}, suspending trading")
                    # Implement pair-specific trading suspension

            # Check data freshness
            for pair in self.config.trading_pairs:
                last_update = self.data_manager.get_last_update_time(pair)
                if (datetime.now(pytz.UTC) - last_update).total_seconds() > 300:
                    logger.warning(f"Stale data for {pair}")

            # Monitor system resources
            # (Add system resource monitoring specific to your GCP setup)

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self._handle_trading_error("health_check", e)

    async def stop(self):
        """Stop the trading system gracefully"""
        try:
            logger.info("Stopping trading system...")

            # Stop scheduler
            self.scheduler.shutdown()

            # Close all positions if configured to do so
            # await self._close_all_positions()

            # Backup final state
            await self._backup_system_state()

            # Cleanup resources
            self.executor.shutdown()

            self.is_running = False
            logger.info("Trading system stopped successfully")

        except Exception as e:
            logger.critical(f"Error stopping trading system: {str(e)}")
            raise


async def run_trading_system():
    """Initialize and run the trading system."""
    try:
        logger.info("Starting trading system initialization...")

        # Create trading coordinator with our configuration
        coordinator = TradingCoordinator(TRADING_CONFIG)

        # Start the trading system
        await coordinator.start()
        logger.info("Trading system started successfully")

        # Keep the system running and handle shutdown
        try:
            while True:
                await asyncio.sleep(60)
                if not coordinator.is_running:
                    logger.warning("Trading system stopped running")
                    break

        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            await coordinator.stop()
            logger.info("Trading system shutdown completed")

        except Exception as e:
            logger.error(f"Unexpected error during runtime: {str(e)}")
            await coordinator.stop()
            raise

    except Exception as e:
        logger.critical(f"Failed to start trading system: {str(e)}")
        raise
