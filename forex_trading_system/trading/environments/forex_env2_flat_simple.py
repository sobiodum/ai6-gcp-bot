from typing import Dict, Optional
from utils.logging_utils import setup_logging, get_logger
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from numba import jit  # Add numba for performance-critical calculations
# Add the project root to the Python path
import os
import sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

setup_logging()
logger = get_logger('ForexEnv2_flat')


class Actions(Enum):
    NO_POSITION = 0
    LONG = 1
    SHORT = 2


class MarketSession(Enum):
    TOKYO = 0
    LONDON = 1
    NEW_YORK = 2
    OFF_HOURS = 3


class TradingPairCosts:
    """Manages trading costs for different currency pairs."""

    base_spreads = {
        # Major Pairs
        'EUR_USD': 0,
        'GBP_USD': 0.0,
        'USD_JPY': 0.0,
        'AUD_USD': 0.0,
        'USD_CAD': 0.0,
        'USD_CHF': 0.0,

        # Cross Pairs
        'EUR_GBP': 0.0,
        'EUR_JPY': 0.0,
        'GBP_JPY': 0.0,
        'EUR_CHF': 0.0,
        'EUR_CAD': 0.0,
        'GBP_CHF': 0.0,
        'CHF_JPY': 0.0,
        'AUD_JPY': 0.0,
        'NZD_USD': 0.0,
        'NZD_JPY': 0.0,

        # Precious Metals
        'XAU_USD': 0.0,
        'XAG_USD': 0.0
    }

    session_multipliers = {
        'ASIAN': 1.1,
        'LONDON': 1.0,
        'NEW_YORK': 1.0,
        'OFF_HOURS': 1.2
    }

    def get_cost(self, pair: str, session: str = 'LONDON') -> float:
        """Get trading cost for a pair during specific session."""
        base_cost = self.base_spreads.get(pair, 0.0001)
        # multiplier = self.session_multipliers.get(session, 1.2)
        return base_cost


@dataclass
class TradingPairNotional:
    default_notional: float = 100_000.0
    pair_notional: Dict[str, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        """
        Initialize the notional dictionary after the dataclass is instantiated.
        This method is automatically called after __init__.
        """
        default_pairs = {
            # Major Pairs
            'EUR_USD': 94_510.0,
            'GBP_USD': 78_500.0,
            'USD_JPY': self.default_notional,
            'USD_CHF': self.default_notional,
            'USD_CAD': self.default_notional,
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
            'XAU_USD': 37.68,  # Gold
            'XAG_USD': 3_266  # Silver
        }
        self.pair_notional.update(default_pairs)


@dataclass
class Position:
    """Represents an open trading position."""
    type: str
    entry_price: float
    size: float
    entry_time: int  # Changed to integer index for faster access
    base_currency: str
    quote_currency: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


@dataclass
class RewardParams:
    """Parameters controlling the reward function behavior."""
    realized_pnl_weight: float = 1.0
    unrealized_pnl_weight: float = 1.0
    holding_time_threshold: int = 7*12
    holding_penalty_factor: float = -0.00001
    max_trades_per_day: int = 6
    overtrading_penalty_factor: float = -0.0001
    win_rate_threshold: float = 0.4
    win_rate_bonus_factor: float = 0.0005
    drawdown_penalty_factor: float = -0.0001


class ForexTradingEnv(gym.Env):
    current_position: Actions = None

    def __init__(
        self,
        # df: pd.DataFrame,
        # pair: str,
        # Can be list of paths or dict of pair:path
        df_paths: Union[List[str], Dict[str, str]],

        eval_mode: bool = False,  # New parameter to control evaluation mode
        eval_path: Optional[str] = None,  # Specific path for evaluation
        pair: str = None,  # Optional - if None, will be extracted from path
        initial_balance: float = 50_000.0,
        trading_notional: Optional[TradingPairNotional] = None,

        max_position_size: float = 1.0,

        trading_costs: Optional[TradingPairCosts] = None,  # New parameter

        reward_scaling: float = 1e-4,
        sequence_length: int = 5,
        random_start: bool = False,
        margin_rate_pct: float = 0.01,
        trading_history_size: int = 50,
        reward_params: Optional[RewardParams] = None,
        excluded_features: List[str] = [
            'timestamp', 'volume', 'open', 'high', 'low'],
        included_features: List[str] = None,
    ):
        super(ForexTradingEnv, self).__init__()
        # Store paths and configuration
        self.df_paths = df_paths if isinstance(
            df_paths, list) else list(df_paths.values())
        self.eval_mode = eval_mode
        self.eval_path = eval_path
        self.current_path = None

        self.excluded_features = excluded_features or [
            'timestamp', 'open', 'high', 'low', 'close']
        self.included_features = included_features
        # self.df = df
        # Basic configuration
        self.pair = pair

        self.initial_balance = initial_balance

        # Initialize trading notional handler
        # Get the correct notional for this pair
        self.max_position_size = max_position_size

        self.reward_scaling = reward_scaling
        self.sequence_length = sequence_length
        self.random_start = random_start
        self.margin_rate_pct = margin_rate_pct
        self.trading_history_size = trading_history_size
        self.reward_params = reward_params or RewardParams()

        # Convert DataFrame to structured arrays for faster access
        # self._preprocess_data(df)
        # Pre-compute time-based features
        #! This excluded for now as likely not needed
        # self._precompute_time_features()

        # These will be set after loading the first dataset
        self.trading_costs = trading_costs or TradingPairCosts()
        self.trading_notional = trading_notional or TradingPairNotional()
        self.transaction_cost = None
        self.trade_size = None
        self.base_currency = None
        self.quote_currency = None
        self.balance = initial_balance
        self.current_step = 0
        self.prev_price = None
        self.total_pnl = 0.0
        self.total_trades = 0
        self.transaction_costs_paid = 0.0

        # Initialize spaces
        self._load_random_dataset()
        self._setup_spaces()

        # Initialize state variables
        self.reset()

    def _load_random_dataset(self):
        """Load a random dataset from available paths and setup pair-specific parameters."""
        try:
            # Select appropriate dataset path
            if self.eval_mode and self.eval_path:
                self.current_path = self.eval_path
            else:
                self.current_path = np.random.choice(self.df_paths)

            # Load the dataset
            self.df = pd.read_parquet(self.current_path)

            # Handle pair determination
            if self.pair is None:
                # Extract pair from filename (assuming format like "EUR_USD_train.parquet")
                filename = os.path.basename(self.current_path)
                # Split on either _train or _validate to handle both cases
                pair_part = filename.split('_train')[0].split('_validate')[0]
                if '_' in pair_part:
                    self.pair = pair_part
                else:
                    raise ValueError(
                        f"Could not extract pair from filename: {filename}")

            # Now that we have the pair, setup pair-specific parameters
            if not hasattr(self, 'base_currency') or self.base_currency is None:
                self.base_currency, self.quote_currency = self.pair.split('_')

            # Update trading parameters
            self.transaction_cost = self.trading_costs.get_cost(self.pair)
            self.trade_size = self.trading_notional.pair_notional.get(
                self.pair, 100_000.0)

            # Process the dataset
            self._preprocess_data(self.df)

        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            logger.error(f"Current path: {self.current_path}")
            logger.error(f"Pair: {self.pair}")
            raise

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment state."""
        super().reset(seed=seed)

        # Load new dataset if not in eval mode
        if not self.eval_mode:
            self._load_random_dataset()

        # Set initial position
        self.current_step = self.sequence_length if not self.random_start else \
            np.random.randint(self.sequence_length, len(
                self.market_data['close']) - 100)

        # Reset account state
        self.balance = self.initial_balance
        self._current_position_size = 0.0  # Start with no position
        self.prev_price = self.current_price

        # Reset metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.transaction_costs_paid = 0.0
        self.winning_trades = 0

        # Zero out pre-allocated arrays
        self.market_obs.fill(0)
        self.account_obs.fill(0)

        return self._get_observation_hstack(), self._get_info()

    def _print_after_episode(self):
        """Print episode summary with corrected metrics."""
        total_return = (
            (self.balance / self.initial_balance) - 1) * 100
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100

        logger.info("\nEpisode Summary:")
        logger.info(f"Pair: {self.pair}")
        logger.info(f"Pair costs: {self.transaction_cost:.5f}")
        logger.info(f"Final Return: {total_return:.2f}%")
        logger.info(f"Total PnL: {self.total_pnl:.2f}")
        logger.info(f"Total Costs: {self.transaction_costs_paid:.2f}")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Initial Balance: {self.initial_balance:.2f}")
        logger.info(f"Final Balance: {self.balance:.2f}")
        logger.info(f"Trade_size: {self.trade_size:.2f}")
        logger.info("-" * 50)
        pass

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment with continuous position sizing.

        Args:
            action: Float between -1 and 1 representing desired position size
                   (-1 = maximum short, 0 = no position, 1 = maximum long)
        """
        # Ensure action is in correct range
        action = float(np.clip(action, -1.0, 1.0))

        # Store previous state
        prev_position_size = self._current_position_size
        prev_balance = self.balance
        prev_price = self.current_price

        # Update position and get current price
        self._current_position_size = action
        self.current_step += 1
        current_price = self.current_price

        # Calculate P&L based on the position size
        pnl = 0.0
        if self._current_position_size != 0:
            self.total_trades += 1
            pnl = self._calculate_pnl(
                self._current_position_size,
                prev_price,
                current_price
            )
            # Track winning trade if PnL is positive for this step
            if pnl > 0:
                self.winning_trades += 1

        # Calculate transaction costs based on absolute position change
        position_change = abs(self._current_position_size - prev_position_size)
        # Scale position change to actual size (e.g., -1 to 1 becomes 0 to 2)
        actual_position_change = position_change * self.trade_size
        transaction_cost = actual_position_change * self.transaction_cost

        # Update balance
        self.balance -= transaction_cost
        self.balance += pnl

        # Update metrics
        self.transaction_costs_paid += transaction_cost
        self.total_pnl += pnl
        # if position_change > 0:
        #     self.total_trades += 1

        # Calculate reward (change in balance)
        reward = (self.balance - prev_balance) / self.initial_balance

        # Update price history
        self.prev_price = current_price
        # Enhanced reward calculation
        reward_components = {
            'pnl': pnl / self.initial_balance,  # Normalized PnL
            'transaction_cost': -transaction_cost / self.initial_balance,  # Cost penalty
            # Discourage excessive holding
            'holding_penalty': -0.0001 if self._current_position_size == prev_position_size else 0,
            # Encourage profitable trades
            'win_rate_bonus': 0.001 if pnl > 0 and position_change > 0 else 0,
            # Risk management
            'risk_penalty': -0.0002 * abs(self._current_position_size),
        }
        # Combine reward components
        # reward = sum(reward_components.values())

        # Prepare metrics for logging
        self.metrics = {
            'net_pnl': self.total_pnl - self.transaction_costs_paid,
            'total_pnl': self.total_pnl,
            'balance': self.balance,
            'transaction_costs': self.transaction_costs_paid,
            'win_rate': (self.winning_trades / max(1, self.total_trades)) * 100,
            'total_trades': self.total_trades,
            'position_size': abs(self._current_position_size),
            'reward': reward,
            **reward_components
        }

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.balance <= 0
        if terminated or truncated:
            self._print_after_episode()

        return self._get_observation_hstack(), reward, terminated, truncated, self._get_info()

    def _calculate_transaction_cost(self, current_price: float) -> float:
        """
        Calculate transaction cost in the same currency as PnL is calculated.
        For USD-quoted pairs, keep costs in USD to match PnL calculations.
        For other pairs, convert to base currency.
        """
        # Calculate cost in quote currency
        quote_currency_cost = self.trade_size * self.transaction_cost

        if self.quote_currency == 'USD':
            # For XAU/USD, XAG/USD, keep cost in USD to match PnL
            return quote_currency_cost
        else:
            # For other pairs like USD/JPY, convert to base currency
            return quote_currency_cost / current_price

    def _preprocess_data(self, df: pd.DataFrame):
        """Convert DataFrame to structured arrays for faster access."""
        try:
            # Store timestamps as integers for faster indexing
            self.timestamps = np.array(df.index.astype(np.int64))

            # Ensure included_features is a flat list of strings
            if self.included_features is not None:
                # Flatten any nested lists and convert to strings
                flat_features = []
                features_to_process = self.included_features

                # Handle potential nesting
                while features_to_process:
                    item = features_to_process.pop(0)
                    if isinstance(item, (list, tuple)):
                        features_to_process.extend(item)
                    else:
                        flat_features.append(str(item))

                # Convert to set after flattening
                included_set = set(flat_features)

                # Find which features are actually present in the DataFrame
                available_features = set(df.columns)
                self.feature_columns = list(
                    included_set.intersection(available_features))

                # Log if any requested features are missing
                missing_features = included_set - available_features
                if missing_features:
                    logger.warning(
                        f"Some requested features are missing from the dataset: {missing_features}")
            else:
                # Use all features except excluded ones
                excluded_set = set(self.excluded_features or [])
                self.feature_columns = [
                    col for col in df.columns if col not in excluded_set]

            # Verify we have features to work with
            if not self.feature_columns:
                raise ValueError("No valid features found in the dataset!")

            # Log selected features
            logger.info(
                f"Selected features ({len(self.feature_columns)}): {self.feature_columns}")

            # Set market_features before using it in setup_spaces
            self.market_features = len(self.feature_columns)

            # Convert market data to numpy arrays
            self.market_data = {
                'close': df['price_norm'].values if 'price_norm' in df else df['close'].values,
            }

            # Store feature data
            self.feature_data = df[self.feature_columns].values

            # Validate feature data
            if np.any(np.isnan(self.feature_data)) or np.any(np.isinf(self.feature_data)):
                nan_columns = [col for col, has_nan in zip(
                    self.feature_columns, np.any(np.isnan(self.feature_data), axis=0)) if has_nan]
                inf_columns = [col for col, has_inf in zip(
                    self.feature_columns, np.any(np.isinf(self.feature_data), axis=0)) if has_inf]

                error_msg = "Feature data contains invalid values:\n"
                if nan_columns:
                    error_msg += f"Columns with NaN: {nan_columns}\n"
                if inf_columns:
                    error_msg += f"Columns with Inf: {inf_columns}\n"
                raise ValueError(error_msg)

            # Pre-allocate observation arrays
            self.market_obs = np.zeros(
                (self.sequence_length, len(self.feature_columns)))
            # Simplified to balance and position only
            self.account_obs = np.zeros(2)

        except Exception as e:
            logger.error(f"Error in _preprocess_data: {str(e)}")
            logger.error(f"DataFrame columns: {df.columns.tolist()}")
            logger.error(f"Included features: {self.included_features}")
            raise

    def _precompute_time_features(self):
        """Pre-compute time-based features for all timestamps."""
        timestamps = pd.to_datetime(self.timestamps)
        hours = timestamps.hour + timestamps.minute / 60.0
        days = timestamps.dayofweek

        # Store the raw hours and days for _get_market_context
        self.hours = hours
        self.days = days

        # Pre-compute market sessions
        self.market_sessions = np.zeros((len(timestamps), 3))
        for i, ts in enumerate(timestamps):
            session = self._get_market_session(ts)
            if session != MarketSession.OFF_HOURS:
                self.market_sessions[i, session.value] = 1.0

    def _setup_spaces(self):
        """Initialize flattened observation space."""
        # Calculate total observation size
        market_size = self.sequence_length * self.market_features
        position_info_size = 1  # Balance and position direction

        total_size = market_size + position_info_size

        # Define action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Single flat observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

    def _calculate_pnl(
        self,
        position_size: float,  # Normalized position size (-1 to 1)
        entry_price: float,
        exit_price: float,
    ) -> float:
        """
        Calculate PnL based on position size and price changes.

        Args:
            position_size: Normalized position size (-1 = max short, 1 = max long)
            entry_price: Position entry price
            exit_price: Position exit/current price
        """
        # Calculate raw price difference
        price_diff = exit_price - entry_price

        # Scale the PnL by position size and trade size
        actual_position_size = position_size * self.trade_size

        # If USD is quote currency (e.g., EUR/USD), PnL is already in USD
        if self.quote_currency == 'USD':
            return price_diff * actual_position_size
        # If USD is base currency (e.g., USD/JPY), convert PnL to USD
        elif self.base_currency == 'USD':
            return (price_diff * actual_position_size) / exit_price
        # If neither currency is USD (e.g., EUR/GBP), PnL is in base currency
        else:
            return (price_diff * actual_position_size) / exit_price

    def _get_market_sequence(self) -> np.ndarray:
        """Get the market data sequence with proper padding."""
        if self.current_step < self.sequence_length:
            # Need padding at the start
            pad_length = self.sequence_length - self.current_step
            available_data = self.feature_data[:self.current_step]

            # Create padding with zeros
            self.market_obs[:pad_length] = 0
            if len(available_data) > 0:
                self.market_obs[pad_length:] = available_data
        else:
            # No padding needed
            start_idx = self.current_step - self.sequence_length
            self.market_obs[:] = self.feature_data[start_idx:self.current_step]

        return self.market_obs.astype(np.float32)

    def _validate_observation_shapes(self, observation: np.ndarray) -> None:
        """Validate observation shape and values."""
        expected_size = (self.sequence_length * self.market_features) + 2

        if observation.shape != (expected_size,):
            raise ValueError(
                f"Invalid observation shape: {observation.shape}, "
                f"expected ({expected_size},)"
            )

    @property
    def current_position(self) -> float:
        """Get current position as a single normalized value."""
        if self.position is None:
            return 0.0
        return 1.0 if self.position.type == 'long' else -1.0

    def _get_observation(self) -> np.ndarray:
        """Construct flattened observation."""
        try:
            # Get all components
            # Should be (sequence_length * features,)
            market_features = self._get_market_sequence(
            ).flatten()  # Basic price and indicators
            # Position info
            position_info = np.array([
                self.balance / self.initial_balance,  # Normalized balance
                self.current_position,                # Position direction
            ])

            # Concatenate all features
            observation = np.concatenate([
                market_features,
                position_info,           #
            ]).astype(np.float32)

            # Handle any NaN or infinite values
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                logger.info(
                    f"Warning: Invalid values in observation at step {self.current_step}")
                observation = np.nan_to_num(
                    observation, nan=0.0, posinf=1e6, neginf=-1e6)

            return observation

        except Exception as e:
            logger.info(f"Error constructing observation: {e}")
            raise

    def _get_observation_hstack(self) -> np.ndarray:
        """Construct flattened observation with improved temporal structure."""
        try:
            # Get market data sequence (shape: sequence_length x feature_dim)
            market_data = self._get_market_sequence()

            # Transpose the market data to shape (feature_dim x sequence_length)
            market_data_transposed = market_data.T

            # Flatten the transposed market data to group each feature's values over time
            market_features = market_data_transposed.flatten()

            # Position info
            position_info = np.array([
                # self.balance / self.initial_balance,  # Normalized balance
                self._current_position_size,                # Position direction
            ])

            # Concatenate all features into a single flat array
            observation = np.concatenate([
                market_features,
                position_info,
            ]).astype(np.float32)

            if np.any(np.isnan(market_features)) or np.any(np.isinf(market_features)):
                logger.info(
                    f"Invalid market_features at step {self.current_step}")
                logger.info(f"market_features: {market_features}")
                raise ValueError("market_features contain NaN or Inf values")

            if np.any(np.isnan(position_info)) or np.any(np.isinf(position_info)):
                logger.info(
                    f"Invalid position_info at step {self.current_step}")
                logger.info(f"position_info: {position_info}")
                raise ValueError("position_info contains NaN or Inf values")

            return observation

        except Exception as e:
            logger.info(f"Error constructing observation: {e}")
            raise

        except Exception as e:
            logger.info(f"Error constructing observation: {e}")
            raise

    @property
    def current_price(self) -> float:
        """Get current market price."""
        return self.market_data['close'][self.current_step]

    @property
    def unrealized_pnl(self) -> float:
        """Calculate current unrealized PnL if position exists."""
        if self.position is None:
            return 0.0
        return self._calculate_pnl(
            self.position.type,
            self.position.entry_price,
            self.current_price,
            self.position.size
        )

    def _get_info(self) -> Dict:
        """Get current state information for logging."""
        info = {
            # Account metrics
            'balance': self.balance,
            'net_profit': self.total_pnl - self.transaction_costs_paid,
            'initial_balance': self.initial_balance,
            'total_return_pct': ((self.balance / self.initial_balance) - 1) * 100,

            # Trade metrics
            'total_pnl': self.total_pnl,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades) * 100,

            # Cost metrics
            'transaction_costs': self.transaction_costs_paid,
            'transaction_costs_pct': (self.transaction_costs_paid / self.initial_balance) * 100,

            # Position metrics
            'current_position': self._current_position_size,
            'position_size_pct': abs(self._current_position_size) * 100,

            # Market metrics
            'current_step': self.current_step,
            'current_price': self.current_price,
        }
        return info

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.winning_trades / max(1, self.total_trades)

    @property
    def avg_trade_duration(self) -> float:
        """Calculate average trade duration."""
        if not self.trade_history:
            return 0.0
        return sum(t['duration'] for t in self.trade_history) / len(self.trade_history)

    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if self.peak_balance <= 0:
            return 0.0
        return (self.peak_balance - self.balance) / self.peak_balance

    @property
    def position_ratios(self) -> Dict[str, float]:
        """Calculate position type ratios."""
        if not self.trade_history:
            return {'long': 0.0, 'short': 0.0, 'none': 1.0}

        total = len(self.trade_history)
        longs = sum(1 for t in self.trade_history if t.get('type') == 'long')
        shorts = sum(1 for t in self.trade_history if t.get('type') == 'short')

        return {
            'long': longs / total,
            'short': shorts / total,
            'none': (total - longs - shorts) / total
        }
