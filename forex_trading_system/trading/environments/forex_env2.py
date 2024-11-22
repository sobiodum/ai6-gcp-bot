import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from numba import jit  # Add numba for performance-critical calculations

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
    realized_pnl_weight: float = 0.0
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
        trade_size: float = 100_000.0,
        max_position_size: float = 1.0,
        transaction_cost: float = 0.0001,
        reward_scaling: float = 1e-4,
        sequence_length: int = 10,
        random_start: bool = True,
        margin_rate_pct: float = 0.01,
        trading_history_size: int = 50,
        reward_params: Optional[RewardParams] = None,
    ):
        super(ForexTradingEnv, self).__init__()
        
        self.df = df
        # Basic configuration
        self.pair = pair
        self.base_currency, self.quote_currency = pair.split('_')
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.reward_scaling = reward_scaling
        self.sequence_length = sequence_length
        self.random_start = random_start
        self.margin_rate_pct = margin_rate_pct
        self.trading_history_size = trading_history_size
        self.reward_params = reward_params or RewardParams()
        
        # Convert DataFrame to structured arrays for faster access
        self._preprocess_data(df)
        # Pre-compute time-based features
        self._precompute_time_features()
        
        # Initialize spaces
        self._setup_spaces()
        
        # Initialize state variables
        self.reset()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize/seed the random number generator
        self.np_random = np.random.RandomState(seed)
        
        # Reset account state
        self.balance = self.initial_balance
        self.position = None
        self.peak_balance = self.initial_balance
        self.session_start_balance = self.initial_balance
        
        # Reset trading metrics
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self._last_trade_info = None
        
        # Reset trading history
        self.trade_history = []
        self.session_trades = {session: [] for session in MarketSession}
        
        # Set initial step - use np_random instead of rng
        if self.random_start and len(self.market_data['close']) > self.sequence_length + 100:
            self.current_step = self.np_random.randint(
                self.sequence_length,
                len(self.market_data['close']) - 100
            )
        else:
            self.current_step = self.sequence_length
        
        # Zero out pre-allocated arrays
        self.market_obs.fill(0)
        self.account_obs.fill(0)
        self.risk_obs.fill(0)
        self.context_obs.fill(0)
        self.history_obs.fill(0)
        
        return self._get_observation(), self._get_info()
    
    def _print_after_episode(self):
        """Print episode summary with corrected metrics."""
        total_return = ((self.balance / self.initial_balance) - 1) * 100
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        
        print("\nEpisode Summary:")
        print(f"Final Return: {total_return:.2f}%")
        print(f"Total PnL: {self.total_pnl:.2f}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Initial Balance: {self.initial_balance:.2f}")
        print(f"Final Balance: {self.balance:.2f}")
        print("-" * 50)
        pass 

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """Execute one step in the environment."""
        action = Actions(action)
        reward = 0.0
        # Move to next step / get next price
        current_price = self.df.iloc[self.current_step]['close']
        self.current_step += 1
        prev_price = self.df.iloc[self.current_step-1]['close']
        if self.balance == 0 or self.initial_balance == 0:
            print(f"0 Value balance: {self.balance} self.initial_balance: {self.initial_balance} at step: {self.current_step}")

        # Handle position transitions
        if action == Actions.NO_POSITION and self.position is not None:
            # Close current position
            reward = self._calculate_reward(self._close_position(current_price))

        elif action == Actions.LONG:
            if self.position is None:
                # Open long position
                self._open_position('long', current_price)
                reward = self._calculate_reward()
            
            elif self.position.type == 'short':
                # Close short and open long
                reward = self._calculate_reward(self._close_position(current_price))
                self._open_position('long', current_price)

            elif self.position.type == 'long':
                # Maintain long position, calculate reward based on holding
                reward = self._calculate_reward()

        elif action == Actions.SHORT:
            if self.position is None:
                # Open short position
                self._open_position('short', current_price)
                reward = self._calculate_reward()
            
            elif self.position.type == 'long':
                # Close long and open short
                reward = self._calculate_reward(self._close_position(current_price))
                self._open_position('short', current_price)
            
            elif self.position.type == 'short':
                # Maintain short position, calculate reward based on holding
                reward = self._calculate_reward()

       


        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1 or self.balance <= 0
        truncated = False
        if terminated or truncated:
            self._print_after_episode()

        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _preprocess_data(self, df: pd.DataFrame):
        """Convert DataFrame to structured arrays for faster access."""
        # Store timestamps as integers for faster indexing
        self.timestamps = np.array(df.index.astype(np.int64))
        
        # Convert market data to numpy arrays
        self.market_data = {
            'close': df['close'].values,
            'open': df['open'].values,
            'high': df['high'].values,
            'low': df['low'].values,
            'atr': df['atr'].values if 'atr' in df else np.zeros(len(df))
        }
        
        # Store feature column names and data
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'volume']]
        self.feature_data = df[self.feature_columns].values
        
        # Pre-allocate arrays for faster observation construction
        self.market_obs = np.zeros((self.sequence_length, len(self.feature_columns)))
        self.account_obs = np.zeros(7)
        self.risk_obs = np.zeros(5)
        self.context_obs = np.zeros(7, dtype=np.float32)  # 4 for time encoding + 3 for session
        self.history_obs = np.zeros(5)

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
        """Dynamically initialize observation and action spaces."""
        # Dynamic calculation of market features
        self.market_features = len(self.feature_columns)
        self.account_features = self.account_obs.shape[0]
        self.risk_features = self.risk_obs.shape[0]
        self.history_features = self.history_obs.shape[0]
        self.context_features = 7  # 4 for time encoding (sin/cos hour, sin/cos day) + 3 for session


        # Define action space
        self.action_space = spaces.Discrete(len(Actions))

        # Define observation space dynamically
        self.observation_space = spaces.Dict({
            'market': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.sequence_length, self.market_features),
                dtype=np.float32,
            ),
            'account': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.account_features,),
                dtype=np.float32,
            ),
            'risk': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.risk_features,),
                dtype=np.float32,
            ),
            'context': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.context_features,),
                dtype=np.float32,
            ),
            'history': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.history_features,),
                dtype=np.float32,
            ),
        })

   
    def _calculate_pnl(self, position_type: str, entry_price: float, exit_price: float, position_size: float) -> float:
        """Optimized PnL calculation."""
        if position_type == 'long':
            if self.quote_currency == 'USD':
                return (exit_price - entry_price) * position_size / exit_price
            return (exit_price - entry_price) * position_size
        else:
            if self.quote_currency == 'USD':
                return (entry_price - exit_price) * position_size / exit_price
            return (entry_price - exit_price) * position_size

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
            self.account_obs[2] = self.position.size / self.initial_balance
            current_price = self.market_data['close'][self.current_step]
            self.account_obs[3] = self._calculate_pnl(
                self.position.type,
                self.position.entry_price,
                current_price,
                self.position.size
            ) / self.initial_balance
        else:
            self.account_obs[1:4] = 0.0
            
        self.account_obs[4] = self.total_pnl / self.initial_balance
        self.account_obs[5] = self.total_trades / 1000.0
        self.account_obs[6] = self.winning_trades / max(1, self.total_trades)
        
        return self.account_obs

    def _validate_observation_shapes(self, obs: Dict[str, np.ndarray]) -> None:
        """Validate observation shapes match the defined spaces."""
        expected_shapes = {
            'market': (self.sequence_length, self.market_features),
            'account': (self.account_features,),
            'risk': (self.risk_features,),
            'context': (self.context_features,),
            'history': (self.history_features,)
        }
        
        for key, expected_shape in expected_shapes.items():
            actual_shape = obs[key].shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for {key}: "
                    f"expected {expected_shape}, got {actual_shape}"
                )
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Construct observation with shape validation."""
        obs = {
            'market': self._get_market_sequence(),
            'account': self._get_account_state(),
            'risk': self._get_risk_metrics(self.market_data['close'][self.current_step]),
            'context': self._get_market_context(self.df.index[self.current_step]),
            'history': self._get_trading_history()
        }
        
        try:
            self._validate_observation_shapes(obs)
        except ValueError as e:
            print(f"Observation shape validation failed: {e}")
            print("Current shapes:")
            for k, v in obs.items():
                print(f"{k}: {v.shape}")
            raise

        return obs

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
        if self.balance > 0:
            atr_ratio = atr * self.position.size / self.balance
        else:
            atr_ratio = 0.0

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

    def _get_market_context(self, timestamp: pd.Timestamp) -> np.ndarray:
        """Calculate market context features with fixed shape."""
        current_idx = self.current_step
        
        # Hour encoding
        hour = self.hours[current_idx]
        self.context_obs[0] = np.sin(2 * np.pi * hour / 24.0)
        self.context_obs[1] = np.cos(2 * np.pi * hour / 24.0)
        
        # Day of week encoding
        day = self.days[current_idx]
        self.context_obs[2] = np.sin(2 * np.pi * day / 7.0)
        self.context_obs[3] = np.cos(2 * np.pi * day / 7.0)
        
        # Market session encoding
        self.context_obs[4:7] = self.market_sessions[current_idx]
                
        return self.context_obs


    def _get_trading_history(self) -> np.ndarray:
        """Calculate trading history metrics."""
        if not self.trade_history:
            return np.zeros(5)

        recent_trades = self.trade_history[-self.trading_history_size:]

        # Overall win ratio - check PnL field in trade dictionaries
        win_ratio = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)

        # Average PnL
        avg_pnl = np.mean([t['pnl'] for t in recent_trades]) / self.initial_balance

        # Maximum drawdown in current session
        session_drawdown = (self.session_start_balance - 
                            self.balance) / self.session_start_balance

        # Number of trades in current session (normalized)
        current_session = self._get_market_session(
            self.df.index[self.current_step])
        # Normalize by expected max trades per session
        session_trade_count = len(self.session_trades[current_session]) / 20.0

        # Success rate in current session type
        session_trades = self.session_trades[current_session]
        if session_trades:
            session_success = sum(1 for t in session_trades 
                                if t['pnl'] > 0) / len(session_trades)
        else:
            session_success = 0.0

        return np.array([
            win_ratio,
            avg_pnl,
            session_drawdown,
            session_trade_count,
            session_success
        ], dtype=np.float32)

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
        if self.position is None:
            return
            
        current_time = self.df.index[self.current_step]
        current_price = self.market_data['close'][self.current_step]
        
        trade_info = {
            'pnl': pnl,
            'type': self.position.type,
            'entry_price': self.position.entry_price,
            'exit_price': current_price,
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
        self._on_trade_closed(pnl)  
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
        current_price = self.market_data['close'][self.current_step]
        
        # 1. Realized PnL component
        if realized_pnl != 0:
            normalized_pnl = realized_pnl / self.trade_size
            reward += normalized_pnl * (1 + (self.reward_params.realized_pnl_weight if realized_pnl > 0 else 0))
            
      
            
            # Calculate win rate bonus
            # if self.total_trades > 0:
            #     win_rate = self.winning_trades / self.total_trades
            #     reward += win_rate * 0.1  # Small bonus for maintaining good win rate
        
        # 2. Unrealized PnL component for open positions
        if self.position is not None:
            unrealized_pnl = self._calculate_pnl(
                self.position.type,
                self.position.entry_price,
                current_price,
                self.position.size
            )
            
            normalized_unrealized = unrealized_pnl / self.trade_size
        
            # Add scaled unrealized PnL (smaller weight than realized)
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
    
    def _get_info(self) -> Dict:
        """Get current state information and performance metrics."""
        current_price = self.market_data['close'][self.current_step]
        
        # Calculate unrealized PnL if position exists
        unrealized_pnl = 0.0
        position_duration = 0
        position_type = 'none'
        
        if self.position is not None:
            position_type = self.position.type
            unrealized_pnl = self._calculate_pnl(
                self.position.type,
                self.position.entry_price,
                current_price,
                self.position.size
            )
            position_duration = (self.df.index[self.current_step] - 
                            self.position.entry_time).total_seconds() / 3600  # Convert to hours
        
        # Calculate drawdown
        peak_balance = max(self.peak_balance, self.balance + unrealized_pnl)
        current_balance = self.balance + unrealized_pnl
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
        info= {
            # Account metrics
            'balance': self.balance,
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
            'trading_costs': self.transaction_cost * (self.position.size if self.position else 0.0),
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
