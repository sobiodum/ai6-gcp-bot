import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from dataclasses import dataclass
from enum import Enum


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
class Position:
    """Represents an open trading position."""
    type: str  # 'long' or 'short'
    entry_price: float
    size: float
    entry_time: pd.Timestamp
    base_currency: str
    quote_currency: str
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None


class ForexTradingEnv(gym.Env):
    def __init__(
        self,
        df: pd.DataFrame,
        pair: str,
        initial_balance: float = 10000.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.0001,
        reward_scaling: float = 1e-4,
        sequence_length: int = 10,
        random_start: bool = True,
        trading_history_size: int = 50  # Keep track of last 50 trades
    ):
        super(ForexTradingEnv, self).__init__()

        self.df = df
        self.pair = pair
        self.base_currency = pair.split('_')[0]
        self.quote_currency = pair.split('_')[1]
        self.initial_balance = initial_balance
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.sequence_length = sequence_length
        self.random_start = random_start

        # Additional tracking for enhanced observations
        self.trading_history_size = trading_history_size
        self.trade_history = []  # List of past trade results
        self.session_trades = {session: [] for session in MarketSession}
        self.peak_balance = initial_balance
        self.session_start_balance = initial_balance

        # Calculate observation space size including account state
        self.feature_columns = [col for col in df.columns
                                if col not in ['timestamp', 'volume']]
        # Enhanced observation space
        self.market_features = len(self.feature_columns)
        # Basic account features (balance, position type, size)
        self.account_features = 3
        # Time in pos, drawdown, dist to SL/TP, ATR ratio, unrealized PnL
        self.risk_features = 5
        # Hour sin/cos, day sin/cos, session one-hot (3)
        self.context_features = 7
        # Win ratio, avg PnL, drawdown, trade count, session success
        self.history_features = 5

        # Define action space (NO_POSITION, LONG, SHORT)
        self.action_space = spaces.Discrete(len(Actions))

        self.observation_space = spaces.Dict({
            'market': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(sequence_length, self.market_features),
                dtype=np.float32
            ),
            'account': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.account_features,),
                dtype=np.float32
            ),
            'risk': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.risk_features,),
                dtype=np.float32
            ),
            'context': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.context_features,),
                dtype=np.float32
            ),
            'history': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.history_features,),
                dtype=np.float32
            )
        })

        # Initialize state
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.balance = self.initial_balance
        self.position: Optional[Position] = None
        self.current_step = self.sequence_length

        if self.random_start and len(self.df) > self.sequence_length + 100:
            self.current_step = np.random.randint(
                self.sequence_length,
                len(self.df) - 100
            )

        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action = Actions(action)
        current_price = self.df.iloc[self.current_step]['close']
        reward = 0.0

        # Handle position transitions
        if action == Actions.NO_POSITION and self.position is not None:
            # Close current position
            reward = self._close_position(current_price)

        elif action == Actions.LONG:
            if self.position is None:
                # Open long position
                self._open_position('long', current_price)
            elif self.position.type == 'short':
                # Close short and open long
                reward = self._close_position(current_price)
                self._open_position('long', current_price)

        elif action == Actions.SHORT:
            if self.position is None:
                # Open short position
                self._open_position('short', current_price)
            elif self.position.type == 'long':
                # Close long and open short
                reward = self._close_position(current_price)
                self._open_position('short', current_price)

        # Calculate unrealized PnL if position is open
        if self.position is not None:
            unrealized_pnl = self._calculate_pnl(
                self.position.type,
                self.position.entry_price,
                current_price,
                self.position.size
            )
            reward = unrealized_pnl * self.reward_scaling

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.df) - 1 or self.balance <= 0

        return self._get_observation(), reward, done, False, self._get_info()

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Construct enhanced observation with additional features."""
        current_time = self.df.index[self.current_step]
        current_price = self.df.iloc[self.current_step]['close']

        # 1. Market data sequence (with padding if needed)
        market_obs = self._get_market_sequence()

        # 2. Account state
        account_obs = self._get_account_state()

        # 3. Risk metrics
        risk_obs = self._get_risk_metrics(current_price)

        # 4. Market context
        context_obs = self._get_market_context(current_time)

        # 5. Trading history
        history_obs = self._get_trading_history()

        return {
            'market': market_obs.astype(np.float32),
            'account': account_obs.astype(np.float32),
            'risk': risk_obs.astype(np.float32),
            'context': context_obs.astype(np.float32),
            'history': history_obs.astype(np.float32)
        }

    def _get_risk_metrics(self, current_price: float) -> np.ndarray:
        """Calculate risk-related metrics."""
        if self.position is None:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        # Time in position (normalized by typical holding period, e.g., 24 hours)
        time_in_pos = (self.df.index[self.current_step] -
                       self.position.entry_time).total_seconds() / (24 * 3600)

        # Current drawdown from peak balance
        drawdown = (self.peak_balance - self.balance) / self.peak_balance

        # Distance to stop loss/take profit (if set)
        if self.position.stop_loss:
            dist_to_sl = abs(
                current_price - self.position.stop_loss) / current_price
        else:
            dist_to_sl = 1.0  # No stop loss set

        # ATR ratio to position size
        atr = self.df.iloc[self.current_step]['atr']
        atr_ratio = atr * self.position.size / self.balance

        # Unrealized PnL (normalized by position size)
        unrealized_pnl = self._calculate_pnl(
            self.position.type,
            self.position.entry_price,
            current_price,
            self.position.size
        ) / self.position.size

        return np.array([
            time_in_pos,
            drawdown,
            dist_to_sl,
            atr_ratio,
            unrealized_pnl
        ])

    def _get_market_context(self, current_time: pd.Timestamp) -> np.ndarray:
        """Calculate market context features."""
        # Hour encoding (sin/cos for cyclical nature)
        hour = current_time.hour + current_time.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)

        # Day of week encoding
        day = current_time.weekday()
        day_sin = np.sin(2 * np.pi * day / 7.0)
        day_cos = np.cos(2 * np.pi * day / 7.0)

        # Market session one-hot encoding
        session = self._get_market_session(current_time)
        session_encoding = np.zeros(3)  # Tokyo, London, NY
        if session != MarketSession.OFF_HOURS:
            session_encoding[session.value] = 1.0

        return np.concatenate([
            [hour_sin, hour_cos, day_sin, day_cos],
            session_encoding
        ])

    def _get_trading_history(self) -> np.ndarray:
        """Calculate trading history metrics."""
        if not self.trade_history:
            return np.zeros(5)

        recent_trades = self.trade_history[-self.trading_history_size:]

        # Overall win ratio
        win_ratio = sum(1 for t in recent_trades if t > 0) / len(recent_trades)

        # Average PnL
        avg_pnl = np.mean(recent_trades) / self.initial_balance

        # Maximum drawdown in current session
        session_drawdown = (self.session_start_balance -
                            self.balance) / self.session_start_balance

        # Number of trades in current session (normalized)
        current_session = self._get_market_session(
            self.df.index[self.current_step])
        # Normalize by expected max
        session_trade_count = len(self.session_trades[current_session]) / 20

        # Success rate in current session type
        session_trades = self.session_trades[current_session]
        if session_trades:
            session_success = sum(
                1 for t in session_trades if t > 0) / len(session_trades)
        else:
            session_success = 0.0

        return np.array([
            win_ratio,
            avg_pnl,
            session_drawdown,
            session_trade_count,
            session_success
        ])

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

    def _on_trade_closed(self, pnl: float) -> None:
        """Update trade history when a position is closed."""
        self.trade_history.append(pnl)
        if len(self.trade_history) > self.trading_history_size:
            self.trade_history.pop(0)

        current_session = self._get_market_session(
            self.df.index[self.current_step])
        self.session_trades[current_session].append(pnl)

        # Update peak balance
        self.balance = max(self.balance, self.peak_balance)

    def _calculate_pnl(
        self,
        position_type: str,
        entry_price: float,
        exit_price: float,
        position_size: float
    ) -> float:
        """
        Calculate PnL in base currency terms.

        For example:
        - EUR/USD: PnL in EUR
        - USD/JPY: PnL in USD
        """
        if position_type == 'long':
            # Convert PnL to base currency
            if self.quote_currency == 'USD':
                # For pairs like EUR/USD, convert USD PnL to base currency (EUR)
                pnl = (exit_price - entry_price) * position_size / exit_price
            else:
                # For pairs like USD/JPY, PnL is already in base currency (USD)
                pnl = (exit_price - entry_price) * position_size
        else:  # short
            if self.quote_currency == 'USD':
                pnl = (entry_price - exit_price) * position_size / exit_price
            else:
                pnl = (entry_price - exit_price) * position_size

        return pnl

    def _open_position(self, position_type: str, current_price: float) -> None:
        """Open a new position."""
        position_size = self.balance * self.max_position_size
        entry_price = current_price

        # Add transaction costs
        if position_type == 'long':
            entry_price += self.transaction_cost
        else:
            entry_price -= self.transaction_cost

        self.position = Position(
            type=position_type,
            entry_price=entry_price,
            size=position_size,
            entry_time=self.df.index[self.current_step],
            base_currency=self.base_currency,
            quote_currency=self.quote_currency
        )

        self.balance -= position_size

    def _close_position(self, current_price: float) -> float:
        """Close current position and return reward."""
        if not self.position:
            return 0.0

        # Calculate PnL with transaction costs
        exit_price = current_price
        if self.position.type == 'long':
            exit_price -= self.transaction_cost
        else:
            exit_price += self.transaction_cost

        pnl = self._calculate_pnl(
            self.position.type,
            self.position.entry_price,
            exit_price,
            self.position.size
        )

        # Update metrics
        self.total_pnl += pnl
        self.balance += self.position.size + pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        # Clear position
        self.position = None

        return pnl * self.reward_scaling
