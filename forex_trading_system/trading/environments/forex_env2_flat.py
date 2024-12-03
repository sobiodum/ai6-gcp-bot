from typing import Dict, Optional
from utils.logging_utils import setup_logging, get_logger
import numpy as np
from typing import Dict, List, Tuple, Optional
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


@dataclass
class TradingPairCosts:
    default_cost: float = 0.0001
    pair_costs: Dict[str, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        """
        Initialize the pair_costs dictionary after the dataclass is instantiated.
        This method is automatically called after __init__.
        """
        default_pairs = {
            # Major Pairs
            'EUR_USD': self.default_cost,
            'GBP_USD': self.default_cost,
            'USD_JPY': self.default_cost,
            'USD_CHF': self.default_cost,
            'USD_CAD': self.default_cost,
            'AUD_USD': self.default_cost,
            'NZD_USD': self.default_cost,

            # Cross Pairs
            'EUR_GBP': self.default_cost,
            'EUR_CHF': self.default_cost,
            'EUR_JPY': self.default_cost,
            'EUR_CAD': self.default_cost,
            'GBP_CHF': self.default_cost,
            'GBP_JPY': self.default_cost,
            'CHF_JPY': self.default_cost,
            'AUD_JPY': self.default_cost,
            'NZD_JPY': self.default_cost,

            # Precious Metals
            'XAU_USD': self.default_cost,  # Gold
            'XAG_USD': self.default_cost,  # Silver
        }
        self.pair_costs.update(default_pairs)

    def get_cost(self, pair: str) -> float:
        """
        Get the trading cost for a specific currency pair.

        Args:
            pair: Currency pair symbol (e.g., 'EUR_USD')

        Returns:
            float: Trading cost for the pair, or default cost if pair not found
        """
        return self.pair_costs.get(pair, self.default_cost)


@dataclass
class TradingPairCosts1:
    """Manages trading costs for different currency pairs."""

    base_spreads = {
        # Major Pairs
        'EUR_USD': 0.000157,
        'GBP_USD': 0.00015,
        'USD_JPY': 0.011,
        'AUD_USD': 0.00012,
        'USD_CAD': 0.00014,
        'USD_CHF': 0.00016,

        # Cross Pairs
        'EUR_GBP': 0.0002,
        'EUR_JPY': 0.016,
        'GBP_JPY': 0.022,
        'EUR_CHF': 0.00022,
        'EUR_CAD': 0.00025,
        'GBP_CHF': 0.00028,
        'CHF_JPY': 0.025,
        'AUD_JPY': 0.022,
        'NZD_USD': 0.00018,
        'NZD_JPY': 0.025,

        # Precious Metals
        'XAU_USD': 0.45,
        'XAG_USD': 0.021
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

    def get_notional(self, pair: str) -> float:
        """
        Get the trading notional for a specific currency pair.

        Args:
            pair: Currency pair symbol (e.g., 'EUR_USD')

        Returns:
            float: Trading notional for the pair, or default notional if pair not found
        """
        return self.pair_costs.get(pair, self.default_cost)


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
    def __init__(
        self,
        df: pd.DataFrame,
        pair: str,
        initial_balance: float = 1_000_000.0,
        trading_notional: Optional[TradingPairNotional] = None,
        trade_size: float = 100_000.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.0001,
        trading_costs: Optional[TradingPairCosts] = None,  # New parameter
        trading_costs1: Optional[TradingPairCosts1] = None,
        reward_scaling: float = 1e-4,
        sequence_length: int = 5,
        random_start: bool = True,
        margin_rate_pct: float = 0.01,
        trading_history_size: int = 50,
        reward_params: Optional[RewardParams] = None,
        excluded_features: List[str] = [
            'timestamp', 'volume', 'open', 'high', 'low'],
        included_features: List[str] = None,
    ):
        super(ForexTradingEnv, self).__init__()

        self.excluded_features = excluded_features or [
            'timestamp', 'open', 'high', 'low']
        self.included_features = included_features
        self.df = df
        # Basic configuration
        self.pair = pair
        # self.trading_costs = trading_costs or TradingPairCosts()
        # self.transaction_cost = self.trading_costs.get_cost(pair)
        self.trading_costs = trading_costs or TradingPairCosts1()
        self.transaction_cost = self.trading_costs.get_cost(pair)
        self.base_currency, self.quote_currency = pair.split('_')
        self.initial_balance = initial_balance
        # Initialize trading notional handler
        self.trading_notional = trading_notional or TradingPairNotional()
        # Get the correct notional for this pair
        self.trade_size = self.trading_notional.pair_notional.get(
            pair, 100_000.0)
        self.max_position_size = max_position_size
        # self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.sequence_length = sequence_length
        self.random_start = random_start
        self.margin_rate_pct = margin_rate_pct
        self.trading_history_size = trading_history_size
        self.reward_params = reward_params or RewardParams()
        self.net_worth_chg = 0

        # Convert DataFrame to structured arrays for faster access
        self._preprocess_data(df)
        # Pre-compute time-based features
        #! This excluded for now as likely not needed
        # self._precompute_time_features()

        # Initialize spaces
        self._setup_spaces()

        # Initialize state variables
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Set initial step - use np_random instead of rng
        if self.random_start and len(self.market_data['close']) > self.sequence_length + 100:
            self.current_step = np.random.randint(
                self.sequence_length,
                len(self.market_data['close']) - 100
            )
        else:
            self.current_step = self.sequence_length
        # Initialize/seed the random number generator
        self.np_random = np.random.RandomState(seed)

        # Reset account state
        self.balance = self.initial_balance
        self._prev_net_worth = self.balance
        self.position = None
        self.peak_balance = self.initial_balance
        self.session_start_balance = self.initial_balance
        self.net_worth_chg = 0

        # Reset trading metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self._last_trade_info = None

        # Reset trading history
        self.trade_history = []
        self.session_trades = {session: [] for session in MarketSession}

        # Zero out pre-allocated arrays
        self.market_obs.fill(0)
        self.account_obs.fill(0)
        self.risk_obs.fill(0)
        self.context_obs.fill(0)
        self.history_obs.fill(0)

        return self._get_observation_hstack(), self._get_info()

    def _print_after_episode(self):
        """Print episode summary with corrected metrics."""
        total_return = ((self.balance / self.initial_balance) - 1) * 100
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100

        logger.info("\nEpisode Summary:")
        logger.info(f"Final Return: {total_return:.2f}%")
        logger.info(f"Total PnL: {self.total_pnl:.2f}")
        logger.info(f"Total Trades: {self.total_trades}")
        logger.info(f"Winning Trades: {self.winning_trades}")
        logger.info(f"Win Rate: {win_rate:.2f}%")
        logger.info(f"Initial Balance: {self.initial_balance:.2f}")
        logger.info(f"Final Balance: {self.balance:.2f}")
        logger.info(f"Trade_size: {self.trade_size:.2f}")
        logger.info("-" * 50)
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action = Actions(action)
        reward = 0.0
        # Store previous net worth (balance + unrealized PnL)
        prev_net_worth = self.balance + self.unrealized_pnl

        # Store price before stepping
        pre_step_price = self.current_price
        self.current_step += 1
        post_step_price = self.current_price

        if self.balance == 0 or self.initial_balance == 0:
            logger.info(
                f"0 Value balance: {self.balance} self.initial_balance: {self.initial_balance} at step: {self.current_step}")

        # Handle position transitions
        if action == Actions.NO_POSITION and self.position is not None:
            # Close current position
            reward = self._calculate_reward(
                self._close_position(pre_step_price))

        elif action == Actions.LONG:
            if self.position is None:
                # Open long position
                self._open_position('long', pre_step_price)

            elif self.position.type == 'short':
                # Close short and open long
                self._close_position(pre_step_price)
                self._open_position('long', pre_step_price)

            elif self.position.type == 'long':
                pass
                # No action needed

        elif action == Actions.SHORT:
            if self.position is None:
                # Open short position
                self._open_position('short', pre_step_price)

            elif self.position.type == 'long':
                # Close long and open short
                self._close_position(pre_step_price)
                self._open_position('short', pre_step_price)

            elif self.position.type == 'short':
                pass
                # No action needed

        # Now, compute current net worth
        current_net_worth = self.balance + self.unrealized_pnl

        # Reward is change in net worth
        self.net_worth_chg = current_net_worth - prev_net_worth
        reward = self.net_worth_chg / self.initial_balance

        # Update self._prev_net_worth
        self._prev_net_worth = current_net_worth

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0
        truncated = False
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
        # Store timestamps as integers for faster indexing
        self.timestamps = np.array(df.index.astype(np.int64))
        actual_excluded = [
            col for col in self.excluded_features if col in df.columns]

        # Flexible feature selection
        if self.included_features is not None:
            # Only use specifically included features
            self.feature_columns = [
                col for col in self.included_features if col in df.columns]
        else:
            # Use all features except excluded ones that exist in df
            self.feature_columns = [
                col for col in df.columns if col not in actual_excluded]

        # Log selected features
        # logger.info(
        #     f"Selected features for observation space: {self.feature_columns}")

        # Set market_features before using it in setup_spaces
        self.market_features = len(self.feature_columns)

        # Convert market data to numpy arrays
        self.market_data = {
            'close': df['close'].values,
            # 'open': df['open'].values,
            # 'high': df['high'].values,
            # 'low': df['low'].values,
            # 'atr': df['atr'].values if 'atr' in df else np.zeros(len(df))
        }

        self.feature_data = df[self.feature_columns].values

        # Pre-allocate arrays for faster observation construction
        self.market_obs = np.zeros(
            (self.sequence_length, len(self.feature_columns)))
        self.account_obs = np.zeros(7)
        self.risk_obs = np.zeros(5)
        # 4 for time encoding + 3 for session
        self.context_obs = np.zeros(7, dtype=np.float32)
        self.history_obs = np.zeros(5)

        if np.any(np.isnan(self.feature_data)) or np.any(np.isinf(self.feature_data)):
            # Identify columns with NaN or Inf values
            nan_columns = df[self.feature_columns].columns[df[self.feature_columns].isnull(
            ).any()].tolist()
            inf_columns = df[self.feature_columns].columns[np.isinf(
                df[self.feature_columns]).any()].tolist()

            logger.info(f"Feature data contains NaN or Inf values.")
            if nan_columns:
                logger.info(f"Columns with NaN values: {nan_columns}")
            if inf_columns:
                logger.info(f"Columns with infinite values: {inf_columns}")
            raise ValueError("Feature data contains NaN or Inf values")

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
        self.action_space = spaces.Discrete(len(Actions))

        # Single flat observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_size,),
            dtype=np.float32
        )

    def _calculate_step_pnl(self, current_price: float) -> Dict[str, float]:
        """
        Calculate P&L for the current step, regardless of realized/unrealized status.
        Returns both the step P&L and its categorization.
        """
        # Initialize step P&L components
        step_pnl = {
            'realized': 0.0,
            'unrealized': 0.0
        }

        # Skip if we don't have previous price (first step)
        if self._prev_price is None:
            self._prev_price = current_price
            self._prev_position_type = 'none' if self.position is None else self.position.type
            self._prev_position_size = 0.0 if self.position is None else self.position.size
            return step_pnl

        # Calculate price change
        price_change = current_price - self._prev_price

        # Case 1: Position held through the step
        if self.position is not None and self._prev_position_type == self.position.type:
            # Calculate unrealized P&L for the step
            step_pnl['unrealized'] = price_change * self.position.size * \
                (1 if self.position.type == 'long' else -1)

        # Case 2: Position closed or reversed
        elif self._prev_position_type != 'none':
            # Calculate realized P&L for the closed position
            step_pnl['realized'] = price_change * self._prev_position_size * \
                (1 if self._prev_position_type == 'long' else -1)

            # If new position opened, calculate its unrealized P&L
            if self.position is not None:
                step_pnl['unrealized'] = price_change * self.position.size * \
                    (1 if self.position.type == 'long' else -1)

        # Case 3: New position opened from flat
        elif self.position is not None and self._prev_position_type == 'none':
            step_pnl['unrealized'] = price_change * self.position.size * \
                (1 if self.position.type == 'long' else -1)

        # Update previous state
        self._prev_price = current_price
        self._prev_position_type = 'none' if self.position is None else self.position.type
        self._prev_position_size = 0.0 if self.position is None else self.position.size

        return step_pnl

    def _calculate_pnl(
        self,
        position_type: str,
        entry_price: float,
        exit_price: float,
        position_size: float
    ) -> float:
        """
        Calculate PnL in USD if USD is base/quote currency, otherwise in base currency.

        For pairs with USD (e.g., EUR/USD, USD/JPY), PnL will be in USD.
        For other pairs (e.g., EUR/GBP), PnL will be in base currency (EUR in this case).

        Args:
            position_type: 'long' or 'short'
            entry_price: Position entry price
            exit_price: Position exit/current price
            position_size: Size of position in base currency

        Returns:
            float: PnL in appropriate currency (USD if USD pair, base currency otherwise)
        """

        # Calculate raw price difference based on position type
        if position_type == 'long':
            price_diff = exit_price - entry_price

        else:  # short
            price_diff = entry_price - exit_price

        # If USD is quote currency (e.g., EUR/USD), PnL is already in USD
        if self.quote_currency == 'USD':

            return price_diff * position_size

        # If USD is base currency (e.g., USD/JPY), convert PnL to USD
        elif self.base_currency == 'USD':
            return (price_diff * position_size) / exit_price

        # If neither currency is USD (e.g., EUR/GBP), PnL is in base currency
        else:
            return (price_diff * position_size) / exit_price

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

    def _get_account_state(self) -> np.ndarray:
        """Optimized account state calculation."""
        self.account_obs[0] = self.balance / self.initial_balance

        if self.position is not None:
            self.account_obs[1] = 1.0 if self.position.type == 'long' else -1.0
        else:
            self.account_obs[1] = 0.0
        #! Below taken out to simplify observation space
        # self.account_obs[4] = self.total_pnl / self.initial_balance
        # self.account_obs[5] = self.total_trades / 1000.0
        # self.account_obs[6] = self.winning_trades / max(1, self.total_trades)

        return self.account_obs

    def _validate_observation_shapes(self, observation: np.ndarray) -> None:
        """Validate observation shape and values."""
        expected_size = (self.sequence_length * self.market_features) + 2

        if observation.shape != (expected_size,):
            raise ValueError(
                f"Invalid observation shape: {observation.shape}, "
                f"expected ({expected_size},)"
            )

    # def _get_observation_old(self) -> np.ndarray:
    #     """Construct flattened observation."""
    #     try:
    #         # Get all components
    #         # Should be (sequence_length * features,)
    #         market_obs = self._get_market_sequence().flatten()
    #         account_obs = self._get_account_state()            # Should be (7,)
    #         risk_obs = self._get_risk_metrics(
    #             self.market_data['close'][self.current_step])  # Should be (5,)
    #         context_obs = self._get_market_context(
    #             self.df.index[self.current_step])  # Should be (7,)
    #         history_obs = self._get_trading_history()          # Should be (5,)

    #         # Concatenate all features
    #         observation = np.concatenate([
    #             # Market data (sequence_length * features)
    #             market_obs,
    #             account_obs,           # Account state (7 features)
    #             risk_obs,              # Risk metrics (5 features)
    #             context_obs,           # Market context (7 features)
    #             history_obs            # Trading history (5 features)
    #         ]).astype(np.float32)

    #         # Validate observation shape
    #         expected_size = (
    #             self.sequence_length * self.market_features +  # Market data
    #             self.account_obs.shape[0] +                   # Account state
    #             self.risk_obs.shape[0] +                      # Risk metrics
    #             self.context_obs.shape[0] +                   # Market context
    #             self.history_obs.shape[0]                     # Trading history
    #         )

    #         if observation.shape[0] != expected_size:
    #             raise ValueError(
    #                 f"Observation shape mismatch. Expected {expected_size}, got {observation.shape[0]}"
    #             )

    #         # Handle any NaN or infinite values
    #         if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
    #             print(
    #                 f"Warning: Invalid values in observation at step {self.current_step}")
    #             observation = np.nan_to_num(
    #                 observation, nan=0.0, posinf=1e6, neginf=-1e6)

    #         return observation

    #     except Exception as e:
    #         print(f"Error constructing observation: {e}")
    #         print(f"Shapes - market: {market_obs.shape}, account: {account_obs.shape}, "
    #               f"risk: {risk_obs.shape}, context: {context_obs.shape}, "
    #               f"history: {history_obs.shape}")
    #         raise
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
                self.current_position,                # Position direction
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

    def _get_market_session(self, timestamp: pd.Timestamp) -> MarketSession:
        """Determine current market session."""
        hour = timestamp.hour

        # Convert to UTC+9 for Tokyo
        tokyo_hour = (hour + 9) % 24
        if 9 <= tokyo_hour < 15:
            return MarketSession.TOKYO

        # London session (UTC+0)
        if 8 <= hour < 16:
            return MarketSession.LONDON

        # New York session (UTC-4)
        ny_hour = (hour - 4) % 24
        if 8 <= ny_hour < 17:
            return MarketSession.NEW_YORK

        return MarketSession.OFF_HOURS

    def _on_trade_closed(self, pnl: float, exit_price: float) -> None:
        """Update trade history when a position is closed."""
        if self.position is None:
            return

        current_time = self.df.index[self.current_step]

        trade_info = {
            'pnl': pnl,
            'type': self.position.type,
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'trade_closed': True,
            'size': self.position.size,
            'entry_time': self.position.entry_time,
            'exit_time': current_time,
            'duration': (current_time - self.position.entry_time).total_seconds() / 3600,
            'session': self._get_market_session(current_time),

        }

        self.trade_history.append(trade_info)
        if len(self.trade_history) > self.trading_history_size:
            self.trade_history.pop(0)

        current_session = self._get_market_session(current_time)
        self.session_trades[current_session].append(trade_info)

        # Update peak balance
        self.peak_balance = max(self.peak_balance, self.balance)

    # def _calculate_pnl(
    #     self,
    #     position_type: str,
    #     entry_price: float,
    #     exit_price: float,
    #     position_size: float
    # ) -> float:
    #     """
    #     Calculate PnL in base currency terms.

    #     For example:
    #     - EUR/USD: PnL in EUR
    #     - USD/JPY: PnL in USD
    #     """
    #     if position_type == 'long':
    #         # Convert PnL to base currency
    #         if self.quote_currency == 'USD':
    #             # For pairs like EUR/USD, convert USD PnL to base currency (EUR)
    #             pnl = (exit_price - entry_price) * position_size / exit_price
    #         else:
    #             # For pairs like USD/JPY, PnL is already in base currency (USD)
    #             pnl = (exit_price - entry_price) * position_size
    #     else:  # short
    #         if self.quote_currency == 'USD':
    #             pnl = (entry_price - exit_price) * position_size / exit_price
    #         else:
    #             pnl = (entry_price - exit_price) * position_size

    #     return pnl

    def _open_position(self, position_type: str, current_price: float) -> None:
        """Open a new position."""

        entry_price = current_price
        # transaction_cost = self.transaction_cost * self.trade_size / current_price
        transaction_cost = self._calculate_transaction_cost(current_price)
        self.balance -= transaction_cost

        # Add transaction costs
        # if position_type == 'long':
        #     entry_price += self.transaction_cost
        # else:
        #     entry_price -= self.transaction_cost

        self.position = Position(
            type=position_type,
            entry_price=entry_price,
            size=self.trade_size,
            entry_time=self.df.index[self.current_step],
            base_currency=self.base_currency,
            quote_currency=self.quote_currency
        )

        required_margin = self.trade_size * self.margin_rate_pct  # 1% margin requirement

    def _close_position(self, current_price: float) -> float:
        """Close current position and return reward."""
        if not self.position:
            return 0.0
        # transaction_cost = self.transaction_cost * self.trade_size / current_price
        transaction_cost = self._calculate_transaction_cost(current_price)
        # Deduct transaction cost from balance
        self.balance -= transaction_cost
        # Calculate PnL with transaction costs
        exit_price = current_price
        # if self.position.type == 'long':
        #     exit_price -= self.transaction_cost
        # else:
        #     exit_price += self.transaction_cost

        pnl = self._calculate_pnl(
            self.position.type,
            self.position.entry_price,
            exit_price,
            self.position.size
        )

        self._last_trade_info = {
            'trade_closed': True,  # Must be True to trigger trade recording
            'trade_pnl': pnl,
            'entry_time': self.position.entry_time,
            'exit_time': self.df.index[self.current_step],
            'entry_price': self.position.entry_price,
            'exit_price': exit_price,
            'position_type': self.position.type,
            'position_size': self.position.size,
            'market_state': {
                'session': self._get_market_session(self.df.index[self.current_step]).name,
                'balance': self.balance,
                'total_trades': self.total_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades)
            }
        }

        # Update metrics
        self.total_pnl += pnl
        self.balance += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Call _on_trade_closed before clearing position
        self._on_trade_closed(pnl, exit_price)
        # Clear position
        self.position = None

        return pnl

    def _calculate_reward(self, realized_pnl: float = 0.0) -> float:
        """
        Calculate reward based on multiple factors:
        1. Realized PnL from closed trades
        2. Unrealized PnL from open positions
        3. Risk-adjusted returns (Sharpe-like ratio)
        4. Position holding costs
        5. Trade efficiency metrics

        Returns:
            float: Calculated reward
        """
        reward = 0.0
        current_price = self.current_price

        # 1. Realized PnL component
        if realized_pnl != 0:
            normalized_pnl = realized_pnl / self.trade_size
            reward += normalized_pnl * self.reward_params.realized_pnl_weight

            # Calculate win rate bonus
            # if self.total_trades > 0:
            #     win_rate = self.winning_trades / self.total_trades
            #     reward += win_rate * 0.1  # Small bonus for maintaining good win rate

        # 2. Unrealized PnL component for open positions
        if self.position is not None:
            unrealized_pnl = self.unrealized_pnl
            normalized_unrealized = unrealized_pnl / self.trade_size
            reward += normalized_unrealized * self.reward_params.unrealized_pnl_weight

            #! Stronger penalty for very long holds
            # holding_hours = (self.df.index[self.current_step] -
            #             self.position.entry_time).total_seconds() / 3600  # in hours

            # if holding_hours > self.reward_params.holding_time_threshold:  # Penalize holds over x hours
            #     holding_penalty = self.reward_params.holding_penalty_factor * (holding_hours - self.reward_params.holding_time_threshold)
            #     reward += holding_penalty

            #! 3. Anti-overtrading penalty
            # if self.total_trades > 0:
            #     # Calculate trades per day
            #     total_days = (self.df.index[self.current_step] -
            #                 self.df.index[0]).total_seconds() / (24 * 3600)
            #     trades_per_day = self.total_trades / max(1, total_days)

            #     # Penalty for excessive trading (more than 6 trades per day)
            #     if trades_per_day > self.reward_params.max_trades_per_day:
            #         overtrading_penalty = self.reward_params.overtrading_penalty_factor * (trades_per_day - self.reward_params.max_trades_per_day)
            #         reward += overtrading_penalty

            #! 4. Win rate Linear increase in bonus above 40% win rate
            # min_trades_required = 10
            # if self.total_trades >= min_trades_required:
            #     win_rate = self.winning_trades / self.total_trades
            #     # Linear scaling between 40% and 60% win rate
            #     win_rate_bonus = max(0, (win_rate - self.reward_params.win_rate_threshold) * self.reward_params.win_rate_bonus_factor)
            #     reward += win_rate_bonus

            #! 5. Risk management penalty (progressive with drawdown)
            # if self.balance < self.initial_balance:
            #     drawdown_pct = (self.initial_balance - self.balance) / self.initial_balance
            #     # Linear penalty that increases with drawdown
            #     risk_penalty = self.reward_params.drawdown_penalty_factor * (drawdown_pct * 100) ** 2
            #     reward += risk_penalty

        return float(reward)

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
        """Get current state information and performance metrics."""
        current_price = self.current_price

        # Calculate unrealized PnL if position exists
        unrealized_pnl = 0.0
        position_duration = 0
        position_type = 'none'

        if self.position is not None:
            position_type = self.position.type
            unrealized_pnl = self.unrealized_pnl
            position_duration = (self.df.index[self.current_step] -
                                 self.position.entry_time).total_seconds() / 3600  # Convert to hours

        # Calculate drawdown
        peak_balance = max(self.peak_balance, self.balance + unrealized_pnl)
        current_balance = self.balance + unrealized_pnl
        drawdown = (peak_balance - current_balance) / \
            peak_balance if peak_balance > 0 else 0.0
        info = {
            # Account metrics
            'balance': self.balance,
            'net_worth_chg': self.net_worth_chg,
            'total_pnl': self.total_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_trades': self.total_trades,
            'trade_count': self.total_trades,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'drawdown': drawdown,

            # Position info
            'position_type': position_type,
            'position_size': self.position.size if self.position else 0.0,
            'position_duration': position_duration,

            # Trading costs and metrics
            'trading_costs': self._calculate_transaction_cost(current_price) if self.position else 0.0,
            'avg_trade_pnl': self.total_pnl / max(1, self.total_trades),

            # Episode progress
            'current_step': self.current_step,
            'total_steps': len(self.df),
            'timestamp': self.df.index[self.current_step],

            # Market info
            'current_price': current_price,
            'spread': self.df.iloc[self.current_step].get('spread', self.transaction_cost)
        }
        if self._last_trade_info is not None:
            info.update(self._last_trade_info)
            self._last_trade_info = None
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
