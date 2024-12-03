from data_management.indicator_manager import IndicatorManager
from data_management.preprocessor import DataPreprocessor
from trading.environments.forex_env2_flat import ForexTradingEnv
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import threading
from queue import Queue
import time
import logging
from datetime import datetime, timedelta, timezone
import pytz
from dataclasses import dataclass, field
import json

# Trading components
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from apscheduler.schedulers.background import BackgroundScheduler

# OANDA components
from oandapyV20 import API
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.trades as trades


# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import local components

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('trading_system')

# OANDA Configuration
OANDA_API_KEY = '9317ace4596d61e3e98b1a53b2342483-45d3ad4084c80b111727a9fada9ef0ff'
OANDA_ACCOUNT_ID = '101-004-30348600-001'  # running account
# OANDA_ACCOUNT_ID = '101-004-30348600-002'
OANDA_ENV = 'practice'

# Initialize OANDA client
client = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)


currency_pairs = {
    'EUR_USD': 94_510.0,
    'GBP_USD': 78_500.0,
    'USD_JPY': 100_000.0,
    'USD_CHF': 100_000.0,
    'USD_CAD': 100_000.0,
    'AUD_USD': 153_000.0,
    'NZD_USD': 171_430.0,

    # Cross Pairs
    'EUR_GBP': 94_510,
    'EUR_CHF': 94_510,
    'EUR_JPY': 94_510,
    'EUR_CAD': 94_510,
    'GBP_CHF': 78_500.0,
    'GBP_JPY': 78_500.0,
    'CHF_JPY': 88_100.0,
    'AUD_JPY': 153_000.0,
    'NZD_JPY': 171_430.0,

    # Precious Metals
    'XAU_USD': 38.0,
    'XAG_USD': 3_266

}


@dataclass
class TradeRecord:
    """Detailed record of a single trade."""
    pair: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    position_type: str  # 'LONG' or 'SHORT'
    size: float
    pnl: float
    pnl_percentage: float
    trade_duration: timedelta
    spread_entry: float
    spread_exit: Optional[float]
    model_version: str
    market_session: str
    entry_indicators: Dict[str, float]  # Key indicator values at entry
    exit_indicators: Optional[Dict[str, float]]  # Key indicator values at exit


@dataclass
class PairPerformanceMetrics:
    """Performance metrics for a single currency pair."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    peak_balance: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: timedelta = timedelta(0)
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    model_version: str = ""
    last_retrain_date: Optional[datetime] = None
    performance_by_session: Dict[str, float] = field(default_factory=dict)


class PerformanceTracker:
    """Tracks and analyzes trading system performance."""

    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.trades_path = base_path / "trades"
        self.metrics_path = base_path / "metrics"
        self.trades_path.mkdir(parents=True, exist_ok=True)
        self.metrics_path.mkdir(parents=True, exist_ok=True)

        # Initialize storage
        self.trade_history: Dict[str, List[TradeRecord]] = {}
        self.pair_metrics: Dict[str, PairPerformanceMetrics] = {}
        self.error_log: List[Dict] = []
        self.model_versions: Dict[str, str] = {}

        # Performance thresholds for alerts
        self.thresholds = {
            'drawdown_alert': 0.10,  # 10% drawdown
            'win_rate_min': 0.45,    # 45% minimum win rate
            'trade_frequency_max': 50  # Max trades per day
        }

    def record_trade(self, trade: TradeRecord) -> None:
        """Record a completed trade and update metrics."""
        pair = trade.pair

        # Store trade record
        if pair not in self.trade_history:
            self.trade_history[pair] = []
        self.trade_history[pair].append(trade)

        # Update pair metrics
        if pair not in self.pair_metrics:
            self.pair_metrics[pair] = PairPerformanceMetrics()

        metrics = self.pair_metrics[pair]
        metrics.total_trades += 1
        metrics.total_pnl += trade.pnl

        if trade.pnl > 0:
            metrics.winning_trades += 1
        else:
            metrics.losing_trades += 1

        # Update win rate and other metrics
        self._update_pair_metrics(pair)

        # Check for performance alerts
        self._check_performance_alerts(pair)

    def _update_pair_metrics(self, pair: str) -> None:
        """Update detailed metrics for a currency pair."""
        metrics = self.pair_metrics[pair]
        trades = self.trade_history[pair]

        if not trades:
            return

        # Calculate basic metrics
        metrics.win_rate = metrics.winning_trades / metrics.total_trades

        # Calculate profit factor
        winning_pnl = sum(t.pnl for t in trades if t.pnl > 0)
        losing_pnl = abs(sum(t.pnl for t in trades if t.pnl < 0))
        metrics.profit_factor = winning_pnl / \
            losing_pnl if losing_pnl != 0 else float('inf')

        # Calculate drawdown
        cumulative_pnl = np.cumsum([t.pnl for t in trades])
        peak = np.maximum.accumulate(cumulative_pnl)
        drawdown = (peak - cumulative_pnl) / peak
        metrics.max_drawdown = np.max(drawdown)

        # Calculate session performance
        session_pnl = {}
        for trade in trades:
            session = trade.market_session
            session_pnl[session] = session_pnl.get(session, 0) + trade.pnl
        metrics.performance_by_session = session_pnl

        # Save updated metrics
        self._save_pair_metrics(pair)

    def _check_performance_alerts(self, pair: str) -> None:
        """Check for performance issues that require attention."""
        metrics = self.pair_metrics[pair]
        alerts = []

        # Check drawdown
        if metrics.max_drawdown >= self.thresholds['drawdown_alert']:
            alerts.append(f"High drawdown alert: {metrics.max_drawdown:.1%}")

        # Check win rate
        if metrics.total_trades >= 20 and metrics.win_rate < self.thresholds['win_rate_min']:
            alerts.append(f"Low win rate alert: {metrics.win_rate:.1%}")

        # Check trade frequency
        recent_trades = [t for t in self.trade_history[pair]
                         if t.entry_time > datetime.now() - timedelta(days=1)]
        if len(recent_trades) > self.thresholds['trade_frequency_max']:
            alerts.append("High trade frequency alert")

        if alerts:
            logging.warning(
                f"Performance alerts for {pair}:\n" + "\n".join(alerts))

    def analyze_model_performance(self, pair: str) -> pd.DataFrame:
        """Analyze performance metrics by model version."""
        if pair not in self.trade_history:
            return pd.DataFrame()

        trades = self.trade_history[pair]
        df = pd.DataFrame([{
            'model_version': t.model_version,
            'entry_time': t.entry_time,
            'pnl': t.pnl,
            'trade_duration': t.trade_duration,
            'market_session': t.market_session
        } for t in trades])

        return df.groupby('model_version').agg({
            'pnl': ['count', 'sum', 'mean', 'std'],
            'trade_duration': 'mean'
        })

    def get_pair_summary(self, pair: str, lookback_days: int = 30) -> Dict:
        """Get comprehensive performance summary for a pair."""
        if pair not in self.pair_metrics:
            return {}

        metrics = self.pair_metrics[pair]
        recent_trades = [t for t in self.trade_history[pair]
                         if t.entry_time > datetime.now() - timedelta(days=lookback_days)]

        return {
            'total_trades': metrics.total_trades,
            'win_rate': metrics.win_rate,
            'total_pnl': metrics.total_pnl,
            'max_drawdown': metrics.max_drawdown,
            'profit_factor': metrics.profit_factor,
            'performance_by_session': metrics.performance_by_session,
            'recent_trades_count': len(recent_trades),
            'model_version': metrics.model_version,
            'last_retrain': metrics.last_retrain_date
        }

    def _save_pair_metrics(self, pair: str) -> None:
        """Save pair metrics to disk."""
        metrics = self.pair_metrics[pair]

        # Convert to serializable format
        metrics_dict = {
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'total_pnl': metrics.total_pnl,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'profit_factor': metrics.profit_factor,
            'model_version': metrics.model_version,
            'last_retrain_date': metrics.last_retrain_date.isoformat()
            if metrics.last_retrain_date else None,
            'performance_by_session': metrics.performance_by_session
        }

        # Save to file
        metrics_file = self.metrics_path / f"{pair}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)

    def export_performance_report(self, lookback_days: Optional[int] = None) -> str:
        """Generate a comprehensive performance report."""
        report = ["Trading System Performance Report\n"]
        report.append(f"Generated at: {datetime.now()}\n")

        for pair in sorted(self.pair_metrics.keys()):
            metrics = self.pair_metrics[pair]
            trades = self.trade_history[pair]

            if lookback_days:
                trades = [t for t in trades
                          if t.entry_time > datetime.now() - timedelta(days=lookback_days)]

            report.append(f"\n{pair} Performance:")
            report.append(f"Total Trades: {metrics.total_trades}")
            report.append(f"Win Rate: {metrics.win_rate:.1%}")
            report.append(f"Total PnL: {metrics.total_pnl:,.2f}")
            report.append(f"Max Drawdown: {metrics.max_drawdown:.1%}")
            report.append(f"Profit Factor: {metrics.profit_factor:.2f}")
            report.append("\nPerformance by Session:")

            for session, pnl in metrics.performance_by_session.items():
                report.append(f"  {session}: {pnl:,.2f}")

            report.append(f"\nCurrent Model: {metrics.model_version}")
            if metrics.last_retrain_date:
                report.append(f"Last Retrain: {metrics.last_retrain_date}")

        return "\n".join(report)


class PositionManager:
    """Manages trading positions with safety features and position tracking."""

    def __init__(
        self,
        currency_pairs: Dict[str, float],
        logger: Optional[logging.Logger] = None,
        account_id: str = OANDA_ACCOUNT_ID,
        client: API = API(access_token=OANDA_API_KEY, environment=OANDA_ENV),
    ):
        """
        Initialize the position manager.

        Args:
            client: OANDA API client
            account_id: OANDA account ID
            currency_pairs: Dictionary of currency pairs and their position sizes
            logger: Optional logger instance
        """
        self.client = client
        self.account_id = account_id
        self.currency_pairs = currency_pairs
        self.logger = logger or logging.getLogger(__name__)
        self.positions = {}
        self.last_sync_time = None

    def close_all_positions(self, confirm: bool = True) -> bool:
        """
        Close all open positions with confirmation option.

        Args:
            confirm: If True, requires confirmation before closing positions

        Returns:
            bool: True if all positions closed successfully
        """
        try:
            # Get current positions
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.client.request(r)
            open_positions = response.get('positions', [])

            if not open_positions:
                self.logger.info("No open positions to close")
                return True

            # Show positions and get confirmation if required
            total_positions = len(open_positions)
            if confirm:
                print(f"\nFound {total_positions} open positions:")
                for pos in open_positions:
                    pair = pos['instrument']
                    long_units = float(pos.get('long', {}).get('units', 0))
                    short_units = float(pos.get('short', {}).get('units', 0))
                    print(f"- {pair}: Long: {long_units}, Short: {short_units}")

                confirm_input = input("\nClose all positions? (yes/no): ")
                if confirm_input.lower() != 'yes':
                    self.logger.info("Position closing cancelled by user")
                    return False

            # Close positions
            for pos in open_positions:
                pair = pos['instrument']

                try:
                    # Close long positions
                    if float(pos.get('long', {}).get('units', 0)) > 0:
                        data = {"longUnits": "ALL"}
                        r = positions.PositionClose(
                            accountID=self.account_id,
                            instrument=pair,
                            data=data
                        )
                        self.client.request(r)
                        self.logger.info(f"Closed long position for {pair}")

                    # Close short positions
                    if float(pos.get('short', {}).get('units', 0)) < 0:
                        data = {"shortUnits": "ALL"}
                        r = positions.PositionClose(
                            accountID=self.account_id,
                            instrument=pair,
                            data=data
                        )
                        self.client.request(r)
                        self.logger.info(f"Closed short position for {pair}")

                    # Small delay to prevent rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(
                        f"Error closing position for {pair}: {str(e)}")
                    return False

            # Verify all positions are closed
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.client.request(r)
            remaining_positions = response.get('positions', [])

            if not remaining_positions:
                self.logger.info("All positions successfully closed")
                return True
            else:
                self.logger.warning(
                    f"Some positions remain after closing attempt: {len(remaining_positions)} positions"
                )
                return False

        except Exception as e:
            self.logger.error(f"Error in close_all_positions: {str(e)}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all pending orders."""
        try:
            # Get all pending orders
            r = orders.OrderList(accountID=self.account_id)
            response = self.client.request(r)
            pending_orders = response.get('orders', [])

            if not pending_orders:
                self.logger.info("No pending orders to cancel")
                return True

            # Cancel each order
            for order in pending_orders:
                try:
                    r = orders.OrderCancel(
                        accountID=self.account_id,
                        orderID=order['id']
                    )
                    self.client.request(r)
                    self.logger.info(f"Cancelled order {order['id']}")
                    time.sleep(0.1)  # Rate limiting prevention

                except Exception as e:
                    self.logger.error(
                        f"Error cancelling order {order['id']}: {str(e)}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in cancel_all_orders: {str(e)}")
            return False

    def emergency_shutdown(self) -> None:
        """
        Emergency shutdown - closes all positions and cancels all orders.
        Returns only after confirming all positions are closed.
        """
        self.logger.warning("Initiating emergency shutdown...")

        # First attempt
        success = self.close_all_positions(confirm=False)
        self.cancel_all_orders()

        # Retry if necessary
        if not success:
            self.logger.warning("First closing attempt failed, retrying...")
            time.sleep(1)
            success = self.close_all_positions(confirm=False)

        # Final verification
        r = positions.OpenPositions(accountID=self.account_id)
        response = self.client.request(r)
        remaining_positions = response.get('positions', [])

        if remaining_positions:
            self.logger.error(
                "Emergency shutdown incomplete - some positions remain. "
                "Manual intervention may be required."
            )
        else:
            self.logger.info("Emergency shutdown completed successfully")

    def get_position_status(self) -> pd.DataFrame:
        """
        Get detailed status of all positions.
        Returns DataFrame with position information.
        """
        try:
            r = positions.OpenPositions(accountID=self.account_id)
            response = self.client.request(r)
            positions_data = []

            for pos in response.get('positions', []):
                pair = pos['instrument']
                long_units = float(pos.get('long', {}).get('units', 0))
                short_units = float(pos.get('short', {}).get('units', 0))

                positions_data.append({
                    'pair': pair,
                    'long_units': long_units,
                    'short_units': short_units,
                    'net_position': long_units + short_units,
                    'timestamp': pd.Timestamp.now(tz='UTC')
                })

            return pd.DataFrame(positions_data)

        except Exception as e:
            self.logger.error(f"Error getting position status: {str(e)}")
            return pd.DataFrame()


class SpreadTracker:
    """Tracks and analyzes spread costs by currency pair and trading session."""

    def __init__(self, save_path: str = "spread_history.parquet"):
        self.save_path = Path(save_path)
        self.spreads = pd.DataFrame(columns=[
            # Changed from spread_pips to spread
            'timestamp', 'pair', 'ask', 'bid', 'spread',
            'session', 'trade_type'
        ])
        self.load_history()

    def load_history(self):
        """Load existing spread history if available."""
        if self.save_path.exists():
            self.spreads = pd.read_parquet(self.save_path)

    def get_current_prices(self, pair: str) -> Tuple[float, float]:
        """Get current bid/ask prices from OANDA."""
        params = {
            "count": 1,
            "granularity": "S5",  # 5-second candles for most recent price
            "price": "AB"  # Ask and Bid prices
        }
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        response = client.request(r)

        if not response.get('candles'):
            raise ValueError(f"No price data available for {pair}")

        candle = response['candles'][0]

        ask = float(candle['ask']['c'])
        bid = float(candle['bid']['c'])
        return ask, bid

    def record_spread(self, pair: str, trade_type: str) -> float:
        """
        Record spread at time of trade execution.

        Args:
            pair: Currency pair
            trade_type: 'OPEN' or 'CLOSE'

        Returns:
            float: Raw spread (ask - bid)
        """
        try:
            # Get current prices
            ask, bid = self.get_current_prices(pair)

            # Calculate raw spread
            spread = ask - bid

            # Determine current trading session
            now = pd.Timestamp.now(tz='UTC')
            session = self._get_trading_session(now)

            # Record spread
            new_record = pd.DataFrame([{
                'timestamp': now,
                'pair': pair,
                'ask': ask,
                'bid': bid,
                'spread': spread,  # Raw spread value
                'session': session,
                'trade_type': trade_type
            }])

            self.spreads = pd.concat([self.spreads, new_record])

            # Save updated history
            self.spreads.to_parquet(self.save_path)

            logger.info(f"Recorded spread of {spread:.6f} for {pair} "
                        f"during {session} session ({trade_type})")

            return spread

        except Exception as e:
            logger.error(f"Error recording spread for {pair}: {str(e)}")
            return None

    def get_spread_statistics(self, pair: str = None, session: str = None) -> pd.DataFrame:
        """Get spread statistics by pair and/or session."""
        df = self.spreads

        if pair:
            df = df[df['pair'] == pair]
        if session:
            df = df[df['session'] == session]

        # Simpler aggregation that won't result in NaN
        stats = df.groupby(['pair', 'session']).agg({
            'spread': ['mean', 'std', 'min', 'max', 'count'],
            'timestamp': ['min', 'max']
        }).round(6)  # Round to 6 decimal places for spreads

        return stats

    def _get_trading_session(self, timestamp: pd.Timestamp) -> str:
        """Determine current trading session."""
        hour = timestamp.hour

        # Convert to major session times
        tokyo_hour = (hour + 9) % 24
        ny_hour = (hour - 4) % 24

        if 9 <= tokyo_hour < 15:
            return 'TOKYO'
        elif 8 <= hour < 16:
            return 'LONDON'
        elif 8 <= ny_hour < 17:
            return 'NEW_YORK'
        else:
            return 'OFF_HOURS'

# Usage in TradingSystem class:
# class TradingSystem:
#     def __init__(self):
#         # ... existing initialization ...
#         self.spread_tracker = SpreadTracker()

#     def execute_trade(self, pair: str, current_position: str, new_position: str):
#         """Execute trade with spread tracking."""
#         try:
#             # Record spread before closing position
#             if current_position != 'NO_POSITION':
#                 spread_close = self.spread_tracker.record_spread(pair, 'CLOSE')
#                 print(f'Closing spread for {pair}: {spread_close:.1f} pips')

#             # Record spread before opening position
#             if new_position != 'NO_POSITION':
#                 spread_open = self.spread_tracker.record_spread(pair, 'OPEN')
#                 print(f'Opening spread for {pair}: {spread_open:.1f} pips')

#             # Execute trade as before...

#         except Exception as e:
#             logger.error(f"Error executing trade for {pair}: {str(e)}")
#             raise

#     def analyze_trading_costs(self) -> None:
#         """Analyze current trading costs."""
#         stats = self.spread_tracker.get_spread_statistics()
#         print("\nSpread Statistics by Pair and Session:")
#         print(stats)

#         # Calculate cost impact
#         total_trades = len(self.spread_tracker.spreads)
#         avg_spread_cost = stats['spread_pips']['mean'].mean()
#         print(f"\nAverage spread cost across all pairs: {avg_spread_cost:.1f} pips")
#         print(f"Total trades analyzed: {total_trades}")


# Trading pairs configuration with position sizes


def get_current_time():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


class DataState:
    """
    Thread-safe container for storing and managing forex data.
    Separates raw OHLC data from calculated indicators and normalized data.
    """

    def __init__(self):
        # Raw OHLC data - only gets appended to, never recalculated
        self.raw_data = None

        # In-memory processed data with indicators - recalculated as needed
        self.processed_data = None

        # Normalized version of processed data for model input
        self.normalized_data = None

        self.last_update = None
        self._lock = threading.Lock()

    def update_raw_data(self, new_raw_data: pd.DataFrame) -> None:
        """
        Updates only the raw OHLC data. This should be used when new market data 
        is fetched and needs to be appended to existing data.
        """
        with self._lock:
            self.raw_data = new_raw_data
            self.last_update = pd.Timestamp.now(tz='UTC')

    def update_processed_data(self, processed_data: pd.DataFrame, normalized_data: pd.DataFrame) -> None:
        """
        Updates the processed (indicators) and normalized data. This is used after
        indicator calculations and normalization are performed on the raw data.
        """
        with self._lock:
            self.processed_data = processed_data
            self.normalized_data = normalized_data


class FastDataManager:
    """
    High-performance data manager optimized for low-latency trading.
    Maintains separation between raw OHLC data and calculated indicators.
    """

    def __init__(
        self,
        base_storage_path: str,
        max_history_size: int = 10000
    ):
        self.base_storage_path = Path(base_storage_path)
        self.max_history_size = max_history_size

        # Define features used in training for consistency
        self.training_features = [
            'close', 'sma_20', 'sma_50', 'rsi', 'macd',
            'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle',
            'bb_lower', 'bb_bandwidth', 'bb_percent', 'atr',
            'plus_di', 'minus_di', 'adx', 'senkou_span_a',
            'senkou_span_b', 'tenkan_sen', 'kijun_sen'
        ]

        # Storage for different data types
        self.pair_states = {}
        self.global_lock = threading.Lock()

        # Save operation queue and worker thread
        self.save_queue = Queue()

        # Initialize components
        self.indicator_manager = IndicatorManager()
        self.data_processor = DataPreprocessor()

        # Start save worker thread
        self.save_worker = threading.Thread(
            target=self._parquet_save_worker,
            daemon=True,
            name="ParquetSaveWorker"
        )
        self.save_worker.start()

    def fetch_missing_candles(self, pair: str, last_timestamp: pd.Timestamp) -> pd.DataFrame:
        """
        Fetches new candles from OANDA with improved error handling.
        Only fetches OHLC data without indicators.
        """
        logger.info(
            f'Fetching missing candles for {pair} - time {get_current_time()}')
        try:
            params = {
                "from": last_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "granularity": "M5",
                "price": "M"
            }

            r = instruments.InstrumentsCandles(instrument=pair, params=params)
            response = client.request(r)
            candles = response.get('candles', [])

            if not candles:
                return pd.DataFrame()

            # Extract only OHLC data
            df_list = [{
                'timestamp': pd.to_datetime(candle['time'], utc=True),
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
            } for candle in candles if candle['complete']]

            if not df_list:
                return pd.DataFrame()

            df = pd.DataFrame(df_list)
            df.set_index('timestamp', inplace=True)

            logger.info(
                f"Fetched {len(df)} candles for {pair} at time {get_current_time()}")
            logger.info(f'Last step for {pair} - {df.index[-1]}')

            return df

        except Exception as e:
            logger.error(f"Error fetching candles for {pair}: {str(e)}")
            return pd.DataFrame()

    def initialize_pair(self, pair: str) -> bool:
        """
        Initializes data for a currency pair by loading raw OHLC data and 
        calculating initial indicators. Ensures environment gets properly formatted data.
        """
        try:
            # Load raw OHLC data
            raw_parquet_path = self.base_storage_path / \
                f"{pair}_raw_5min.parquet"

            if not raw_parquet_path.exists():
                logger.error(f"Raw data file not found for {pair}")
                return False

            raw_df = pd.read_parquet(raw_parquet_path)

            if raw_df.empty:
                logger.error(f"Empty DataFrame loaded for {pair}")
                return False

            # Ensure timezone awareness
            if raw_df.index.tz is None:
                raw_df.index = raw_df.index.tz_localize('UTC')

            # Calculate indicators first
            processed_df = self.indicator_manager.calculate_indicators(raw_df)

            # Validate indicator calculation results
            if len(processed_df.columns) != 23:  # Expected number of columns after indicators
                raise ValueError(
                    f"Indicator calculation produced unexpected number of columns: {len(processed_df.columns)}")

            normalized_df = self.data_processor.normalize_simple(processed_df)

            # Initialize pair state
            with self.global_lock:
                if pair not in self.pair_states:
                    self.pair_states[pair] = DataState()

                current_state = self.pair_states[pair]
                # Store both raw and processed data
                current_state.update_raw_data(raw_df)
                current_state.update_processed_data(
                    processed_df, normalized_df)

            # Return the processed DataFrame for environment initialization
            logger.info(
                f"Initialized data for {pair}, loaded {len(raw_df)} candles")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize {pair}: {str(e)}")
            return False

    def update_pair_data(self, pair: str) -> bool:
        """
        Updates data for a pair by fetching new candles and recalculating indicators.
        Only appends new data to raw OHLC storage.
        """
        try:
            # Get current state
            with self.global_lock:
                if pair not in self.pair_states:
                    logger.error(f"Pair {pair} not initialized")
                    return False
                current_state = self.pair_states[pair]

            # Get current raw data
            with current_state._lock:
                if current_state.raw_data is None:
                    logger.error(f"No raw data found for {pair}")
                    return False

                current_raw_data = current_state.raw_data
                last_timestamp = current_raw_data.index[-1]

            # Fetch new data if needed
            current_time = pd.Timestamp.now(tz='UTC')
            if current_time - last_timestamp >= timedelta(minutes=5):
                new_data = self.fetch_missing_candles(pair, last_timestamp)

                if not new_data.empty:
                    # Combine with existing raw data
                    combined_raw = pd.concat([current_raw_data, new_data])
                    combined_raw = combined_raw[~combined_raw.index.duplicated(
                        keep='last')]
                    combined_raw.sort_index(inplace=True)

                    # Queue raw data save operation
                    self.save_queue.put((pair, combined_raw))

                    # Update state with new raw data
                    current_state.update_raw_data(combined_raw)

                    # Calculate indicators and normalize (in memory only)
                    processed_df = self.indicator_manager.calculate_indicators(
                        combined_raw)
                    normalized_df = self.data_processor.normalize_simple(
                        processed_df)

                    # Update processed data in memory
                    current_state.update_processed_data(
                        processed_df, normalized_df)
                    return True

            return False

        except Exception as e:
            logger.error(f"Error updating data for {pair}: {str(e)}")
            raise

    def _parquet_save_worker(self) -> None:
        """
        Background worker for saving raw OHLC data to parquet files.
        Includes backup protection and error handling.
        """
        while True:
            try:
                pair, raw_df = self.save_queue.get()
                if pair is None:  # Shutdown signal
                    break

                parquet_path = self.base_storage_path / \
                    f"{pair}_raw_5min.parquet"
                backup_path = parquet_path.with_suffix('.parquet.backup')

                # Create backup if original exists
                if parquet_path.exists():
                    parquet_path.rename(backup_path)

                try:
                    # Save new raw data
                    raw_df.to_parquet(parquet_path)

                    # Remove backup if save successful
                    if backup_path.exists():
                        backup_path.unlink()

                    logger.info(f"Successfully saved raw data for {pair}")

                except Exception as e:
                    # Restore from backup if save fails
                    if backup_path.exists():
                        backup_path.rename(parquet_path)
                    logger.error(
                        f"Error saving raw data for {pair}, restored from backup: {str(e)}")

            except Exception as e:
                logger.error(f"Error in save worker: {str(e)}")
            finally:
                self.save_queue.task_done()

    def get_prediction_data(self, pair: str, sequence_length: int, current_position: float) -> np.ndarray:
        """
        Gets normalized data sequence for prediction using in-memory processed data.
        """
        try:
            with self.global_lock:
                if pair not in self.pair_states:
                    raise KeyError(f"No data available for {pair}")
                current_state = self.pair_states[pair]

            with current_state._lock:
                if current_state.normalized_data is None:
                    raise ValueError(
                        f"Normalized data not initialized for {pair}")

                df = current_state.normalized_data
                # Get the last timestamp before processing the data
                last_timestamp = df.index[-1]

                # Select only the features used in training
                df_features = df[self.training_features]

                # Get last sequence_length rows
                sequence = df_features.iloc[-sequence_length:].values
                sequence_transposed = sequence.T
                market_features = sequence_transposed.flatten()

                # Add position information
                position_info = np.array([current_position])
                observation = np.concatenate([market_features, position_info])

                return observation.astype(np.float32), last_timestamp

        except Exception as e:
            logger.error(
                f"Error constructing prediction data for {pair}: {str(e)}")
            raise

    def update_all_raw_data(self) -> Dict[str, bool]:
        """
        Updates raw OHLC data for all currency pairs without running the trading system.
        This function is useful for maintenance and initialization of historical data.

        Returns:
            Dict[str, bool]: Dictionary mapping pair names to update success status
        """
        update_results = {}

        for pair in currency_pairs:  # Using the global currency_pairs dictionary
            logger.info(f"Starting update for {pair}")
            try:
                # Initialize state if needed
                with self.global_lock:
                    if pair not in self.pair_states:
                        self.pair_states[pair] = DataState()
                    current_state = self.pair_states[pair]

                # Load existing raw data or create new DataFrame
                raw_parquet_path = self.base_storage_path / \
                    f"{pair}_raw_5min.parquet"

                if raw_parquet_path.exists():
                    current_raw_data = pd.read_parquet(raw_parquet_path)
                    if current_raw_data.index.tz is None:
                        current_raw_data.index = current_raw_data.index.tz_localize(
                            'UTC')
                    last_timestamp = current_raw_data.index[-1]
                else:
                    # If no existing data, start from a reasonable point in the past
                    last_timestamp = pd.Timestamp.now(
                        tz='UTC') - pd.Timedelta(days=30)
                    current_raw_data = pd.DataFrame()

                logger.info(
                    f"Fetching new data for {pair} since {last_timestamp}")

                # Fetch new data
                new_data = self.fetch_missing_candles(pair, last_timestamp)

                if not new_data.empty:
                    # Combine with existing data if any
                    if not current_raw_data.empty:
                        combined_raw = pd.concat([current_raw_data, new_data])
                        combined_raw = combined_raw[~combined_raw.index.duplicated(
                            keep='last')]
                        combined_raw.sort_index(inplace=True)
                    else:
                        combined_raw = new_data

                    # Update state
                    current_state.update_raw_data(combined_raw)

                    # Save to parquet
                    self.save_queue.put((pair, combined_raw))

                    logger.info(
                        f"Successfully updated {pair} with {len(new_data)} new candles")
                    logger.info(
                        f"Total data points for {pair}: {len(combined_raw)}")
                    update_results[pair] = True
                else:
                    logger.info(f"No new data available for {pair}")
                    update_results[pair] = False

            except Exception as e:
                logger.error(f"Error updating raw data for {pair}: {str(e)}")
                logger.info(f"Failed to update {pair}: {str(e)}")
                update_results[pair] = False

        # Wait for all saves to complete
        self.save_queue.join()

        # Print summary
        print("\nUpdate Summary:")
        for pair, success in update_results.items():
            status = "✓ Success" if success else "✗ Failed"
            print(f"{pair}: {status}")

        return update_results

    def shutdown(self):
        """
        Cleanly shuts down the data manager.
        """
        # Signal save worker to stop
        self.save_queue.put((None, None))
        self.save_worker.join()


class TradingSystem:
    """
    Main trading system coordinator with improved thread safety and error handling.
    Manages the interaction between data, models, and trading execution.
    """

    def __init__(self):
        # Core components
        self.data_manager = None
        self.models = {}
        self.spread_tracker = SpreadTracker()

        # Position tracking with thread safety
        self.positions = {}
        self.positions_lock = threading.Lock()

        # Trade metadata tracking
        self.position_metadata = {
            'entry_prices': {},
            'entry_times': {},
            'entry_indicators': {},
            'entry_spreads': {}
        }
        self.metadata_lock = threading.Lock()

        # System state
        self.start_time = datetime.now(timezone.utc)
        self.is_running = False
        self.scheduler = None

    def initialize(self):
        """Initialize the trading system with proper data handling."""
        logger.info("Initializing trading system...")

        try:
            self.data_manager = FastDataManager(
                base_storage_path="./raw_data"
            )

            for pair in currency_pairs:
                try:
                    # Initialize data first
                    if not self.data_manager.initialize_pair(pair):
                        continue

                    # Get the processed data with indicators for environment creation
                    current_state = self.data_manager.pair_states[pair]
                    with current_state._lock:
                        processed_data = current_state.processed_data

                    # Create environment with processed data
                    vec_env = DummyVecEnv([lambda: ForexTradingEnv(
                        processed_data, pair  # Using processed data with indicators
                    )])

                    model_path = f'./models_and_vecs/{pair}_best_model'
                    env_path = f'./models_and_vecs/{pair}_vec_normalize.pkl'

                    env = VecNormalize.load(env_path, vec_env)
                    env.training = False
                    env.norm_reward = False

                    model = PPO.load(model_path, env=env)
                    self.models[pair] = model

                except Exception as e:
                    logger.error(f"Error initializing {pair}: {str(e)}")
                    continue

            self.sync_positions()
            return True

        except Exception as e:
            logger.error(f"Fatal error during initialization: {str(e)}")
            return False

    def position_to_float(self, position_type: str) -> float:
        """Convert position type to float representation for model input."""
        position_map = {
            'LONG': 1.0,
            'SHORT': -1.0,
            'NO_POSITION': 0.0
        }
        return position_map.get(position_type, 0.0)

    def _make_prediction(self, pair: str, observation: np.ndarray) -> str:
        """
        Make a prediction using the loaded model with improved error handling.
        """
        try:
            if pair not in self.models:
                raise KeyError(f"No model loaded for {pair}")

            # Validate observation shape
            expected_shape = self.models[pair].policy.observation_space.shape[0]
            if observation.shape[0] != expected_shape:
                raise ValueError(
                    f"Observation shape mismatch for {pair}: "
                    f"expected {expected_shape}, got {observation.shape[0]}"
                )

            # Get prediction
            action, _ = self.models[pair].predict(
                observation.reshape(1, -1), deterministic=True)

            # Map action to position type
            action_map = {0: 'NO_POSITION', 1: 'LONG', 2: 'SHORT'}
            return action_map[action[0]]

        except Exception as e:
            logger.error(f"Error making prediction for {pair}: {str(e)}")
            # Return current position on error to avoid unwanted changes
            with self.positions_lock:
                return self.positions.get(pair, 'NO_POSITION')

    def sync_positions(self):
        """Synchronize positions with broker with thread safety."""
        try:
            r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
            response = client.request(r)

            with self.positions_lock:
                self.positions.clear()
                for pos in response.get('positions', []):
                    pair = pos['instrument']
                    if pair in currency_pairs:
                        if float(pos.get('long', {}).get('units', 0)) > 0:
                            self.positions[pair] = 'LONG'
                        elif float(pos.get('short', {}).get('units', 0)) < 0:
                            self.positions[pair] = 'SHORT'
                        else:
                            self.positions[pair] = 'NO_POSITION'

            logger.info("Positions synchronized successfully")
            return True

        except Exception as e:
            logger.error(f"Error syncing positions: {str(e)}")
            return False

    def execute_trade(self, pair: str, current_position: str, new_position: str):
        """Execute a trade with improved position tracking and error handling."""
        try:
            logger.info(
                f"Executing trade for {pair}: {current_position} -> {new_position}")

            # Close existing position if any
            if current_position != 'NO_POSITION':
                self.close_position(pair, current_position)
                # spread_close = self.spread_tracker.record_spread(pair, 'CLOSE')
                # print(f'Closing spread for {pair}: {spread_close:.1f} pips')

            # Open new position if not moving to neutral
            if new_position != 'NO_POSITION':
                self.open_position(pair, new_position)
                # spread_open = self.spread_tracker.record_spread(pair, 'OPEN')
                # print(f'Opening spread for {pair}: {spread_open:.1f} pips')

            # Update position tracking
            with self.positions_lock:
                self.positions[pair] = new_position

            logger.info(f"Successfully executed trade for {pair}")
            return True

        except Exception as e:
            logger.error(f"Error executing trade for {pair}: {str(e)}")
            return False

    def open_position(self, pair: str, position_type: str):
        """Open a new position with improved error handling."""
        try:
            units = currency_pairs[pair]
            if position_type == 'SHORT':
                units = -units

            data = {
                "order": {
                    "instrument": pair,
                    "units": str(units),
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }

            r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
            response = client.request(r)
            logger.info(f'Open position response: {response}')

            # Record entry metadata
            with self.metadata_lock:
                self.position_metadata['entry_times'][pair] = datetime.now(
                    timezone.utc)
                # Add other metadata as needed

            return True

        except Exception as e:
            logger.error(f"Error opening position for {pair}: {str(e)}")
            return False

    def close_position(self, pair: str, position_type: str):
        """Close an existing position with improved error handling."""
        try:
            data = {
                "longUnits": "ALL"
            } if position_type == 'LONG' else {
                "shortUnits": "ALL"
            }

            r = positions.PositionClose(
                accountID=OANDA_ACCOUNT_ID,
                instrument=pair,
                data=data
            )
            response = client.request(r)
            logger.info(f'Close position response: {response}')

            # Clear position metadata
            with self.metadata_lock:
                for metadata_dict in self.position_metadata.values():
                    metadata_dict.pop(pair, None)

            return True

        except Exception as e:
            logger.error(f"Error closing position for {pair}: {str(e)}")
            return False

    def trading_cycle(self):
        """Execute one trading cycle with improved error handling and logging."""
        logger.info("Starting trading cycle")
        current_time = datetime.now(timezone.utc)
        # print("Available models:", list(self.models.keys()))

        for pair in currency_pairs:
            try:
                if pair not in self.models:
                    continue

                # Update market data
                if self.data_manager.update_pair_data(pair):
                    # Get current position with proper locking
                    with self.positions_lock:
                        current_position_type = self.positions.get(
                            pair, 'NO_POSITION')
                        logger.info(
                            f"Current position for {pair}: {current_position_type}")

                    # Convert position to float for observation
                    current_position_float = self.position_to_float(
                        current_position_type)

                    # Get prediction data
                    try:
                        observation, last_timestamp = self.data_manager.get_prediction_data(
                            pair=pair,
                            sequence_length=5,
                            current_position=current_position_float
                        )
                        time_difference = current_time - last_timestamp
                        logger.info(
                            f"Latest data timestamp for {pair}: {last_timestamp}")
                        logger.info(f"Data age: {time_difference}")
                        # Warning if data is too old (e.g., more than 10 minutes)
                        if time_difference > timedelta(minutes=7):
                            logger.info(
                                f"WARNING: Data for {pair} is {time_difference} old")
                    except Exception as e:
                        logger.error(
                            f"Error getting prediction data for {pair}: {str(e)}")
                        continue

                    # Get model prediction
                    action_name = self._make_prediction(pair, observation)
                    logger.info(
                        f"Decision for {pair}: {current_position_type} -> {action_name}")

                    # Execute trade if position change needed
                    if current_position_type != action_name:
                        if not self.execute_trade(pair, current_position_type, action_name):
                            logger.error(f"Failed to execute trade for {pair}")

            except Exception as e:
                logger.error(f"Error in trading cycle for {pair}: {str(e)}")
                continue

    def run(self):
        """Run the trading system with improved shutdown handling."""
        try:
            if not self.initialize():
                raise RuntimeError("Failed to initialize trading system")

            self.is_running = True
            self.scheduler = BackgroundScheduler()
            self.scheduler.add_job(
                self.trading_cycle,
                'cron',
                minute='*/5',
                second=0
            )
            self.scheduler.start()

            logger.info("Trading system started successfully")

            while self.is_running:
                time.sleep(60)

        except (KeyboardInterrupt, SystemExit):
            self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error in trading system: {str(e)}")
            self.shutdown()
            raise

    def shutdown(self):
        """Clean shutdown of all system components."""
        logger.info("Initiating trading system shutdown...")

        self.is_running = False

        if self.scheduler:
            self.scheduler.shutdown()

        if self.data_manager:
            self.data_manager.shutdown()

        logger.info("Trading system shutdown complete")
