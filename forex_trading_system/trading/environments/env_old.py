# from process_data import LoadDataFrame
from dataclasses import dataclass, asdict
from typing import Optional
# from process_data import LoadDataFrame, AddTechnicalIndicators
import matplotlib.pyplot as plt
from copy import deepcopy, copy
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random
import sys
import os
from scipy.special import expit
import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Now set up the new configuration
logging.basicConfig(level=logging.DEBUG,
                    format='%(message)s')
# logging.basicConfig(level=logging.INFO, format='%(message)s')


# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(levelname)s - %(message)s')
# logging.disable(logging.DEBUG)
logging.disable(logging.INFO)


@dataclass
class Order:
    id: int
    volume: float
    order_type: str  # 'buy' or 'sell'
    open_time: Optional[datetime] = None
    close_price: Optional[float] = None
    close_time: Optional[datetime] = None
    profit: Optional[float] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class Steps:
    step: int
    prev_position: float
    current_position: float
    position_change: float
    current_price: float
    profit: float

    def to_dataframe(self):
        return pd.DataFrame([self.to_dict()])

    def to_dict(self):
        return asdict(self)

#! ISSUES
# 1) load DF seems to run too often -- maybe work with post init?


class TradingEnvSimple_V2(gym.Env):
    def __init__(self, df,  window_size=10,
                 load_specific: str = None,
                 track_orders: bool = True,
                 running_mode: str = 'training',
                 verbose_mode: str = "debug",
                 validation_file_path=None,
                 validation_file_paths=None,
                 resample_interval: str = '1h',
                 transaction_cost_rate: float = 0.001
                 ):
        super(TradingEnvSimple_V2, self).__init__()
        self.resample_interval = resample_interval
        self.load_specific = load_specific
        self.validation_file_path = validation_file_path
        self.validation_file_paths = validation_file_paths
        self.running_mode = running_mode
        # Load the initial DataFrame
        self.df = df
        # self.df = self._load_df()
        # Define feature columns
        self.feature_columns = [
            col for col in self.df.columns if col not in ['timestamp', 'close', 'close_norm', 'date', 'open', 'high', 'low']]
        self.features_array = self.df[self.feature_columns].values.astype(
            np.float32)
        # Store the total number of steps
        self.total_steps = len(self.df)
        self.window_size = window_size
        self.track_orders = track_orders
        self.current_step = 0
        self.total_pnl_pct = 0
        self.total_transaction_costs = 0
        self.step_cost = 0.0
        self.current_position = 0  # Net position (-1, 0, 1)
        self.balance = 100_000.0
        self.initial_balance = self.balance
        self.trade_size = 100_000
        self.prev_balance = self.balance
        # Step Tracking
        self.prev_position_array = []
        self.current_position_array = []
        self.position_chg_array = []
        self.pnl_array = []
        self.date_array = []
        self.price_array = []
        self.next_price_array = []
        self.cum_cost_array = []
        self.reward_array = []
        self.balance_array = []
        self.num_orders = []
        # For backtesting
        self.backtest_date_array = []
        self.backtest_pnl_array = []
        self.backtest_current_pos_array = []

        # For order tracking
        self.orders = []
        self.current_order = None
        self.order_counter = 0

        # Episode tracking
        self.episode_pnls = []
        self.episode_metrics = []
        self.loaded_file = ''

        self.transaction_cost_rate = transaction_cost_rate  # 0.01%

        # Define observation space
        # +3 for current_price (close_norm), balance, position
        num_features = self.window_size * self.features_array.shape[1] + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_features,),
            dtype=np.float32
        )
        # Define action space: Discrete actions: 0 (sell), 1 (hold/close), 2 (buy)
        self.action_space = spaces.Discrete(3)

    def set_test_file_path(self, file_path):
        self.load_specific = file_path

    def reset(self, seed=None, options=None):
        # self.df = self._load_df()
        # Recompute feature columns and features array
        feature_columns = [
            col for col in self.df.columns if col not in ['close', 'close_norm', 'date', 'open', 'high', 'low', 'FAKE_TO_TEST']
        ]
        # Ensure feature columns are consistent across datasets
        if hasattr(self, 'feature_columns'):
            assert feature_columns == self.feature_columns, "Feature columns mismatch between datasets"
        else:
            self.feature_columns = feature_columns
        self.features_array = self.df[self.feature_columns].values.astype(
            np.float32)
        if seed is not None:
            # Set the seed explicitly for reproducibility
            random.seed(seed)  # Python's built-in random module
            np.random.seed(seed)  # NumPy's random module
        self.current_step = 0
        self.total_pnl_pct = 0
        self.current_position = 0
        self.step_cost = 0.0
        self.total_transaction_costs = 0
        self.order_counter = 0
        self.balance = self.initial_balance = 100_000.0
        self.prev_balance = self.balance
        self.prev_position_array = []
        self.current_position_array = []
        self.position_chg_array = []
        self.next_price_array = []
        self.pnl_array = []
        self.date_array = []
        self.price_array = []
        self.cum_cost_array = []
        self.reward_array = []
        self.balance_array = []
        self.num_orders = []
        self.backtest_date_array = []
        self.backtest_pnl_array = []
        self.backtest_current_pos_array = []

        # Reset orders
        self.orders = []
        self.current_order = None

        observation = self._get_observation()
        info = {}
        return observation, info

    def _get_observation(self):
        # Calculate the start index for the window
        start_index = max(0, self.current_step - self.window_size + 1)
        end_index = self.current_step + 1  # Include current_step

        # Slice the precomputed features array
        features = self.features_array[start_index:end_index, :]

        # Ensure the features array has the correct shape
        if len(features) < self.window_size:
            # Pad with zeros if necessary
            padding = np.zeros(
                (self.window_size - len(features), features.shape[1]), dtype=np.float32)
            features = np.vstack((padding, features))

        features_flat = features.flatten()

        # Get the current normalized price
        current_price = self.df['close'].iloc[self.current_step]
        current_price = np.array([current_price], dtype=np.float32)
        # normalize the balance so that it is not outweighing anything
        normalized_balance = (
            self.balance - self.initial_balance) / self.initial_balance

        # Concatenate features
        observation = np.concatenate(
            [features_flat, current_price, [
                normalized_balance], [self.current_position]]
        ).astype(np.float32)

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(
                f"NaN or infinite value found in observation at step {self.current_step} in df: {self.loaded_file}")
        return observation

    def _calculate_reward(self, next_price, current_price):
        price_chg_pct = (next_price / current_price) - 1

        pnl = price_chg_pct * self.current_position * self.trade_size

        self.balance += pnl - self.step_cost
        reward = pnl - self.step_cost  # Actual P&L as reward

        pnl -= self.step_cost
        if np.isnan(reward) or np.isnan(pnl):
            print(
                f"[Reward Issue] NaN found in reward or pnl at step {self.current_step}")

        return float(reward), float(pnl)

    def close_position(self, current_date, current_price):
        if self.current_order is not None:
            self.current_order['close_date'] = current_date
            self.current_order['close_price'] = current_price
            # Calculate PnL
            pct_chg = current_price / self.current_order['open_price'] - 1
            pnl_usd = pct_chg * \
                self.current_order['position'] * self.trade_size
            pnl_usd -= self.step_cost  # Subtract transaction cost
            self.current_order['pnl_usd'] += pnl_usd
            self.current_order['pnl_pct'] = pnl_usd / \
                self.initial_balance * 100
            self.current_order['holding_period'] = (
                self.current_order['close_date'] - self.current_order['open_date']).days
            self.orders.append(self.current_order)
            self.current_order = None
        pass

    def _handle_trading(self, prev_position, desired_position, position_change, current_price, current_date):
        # Transaction cost is based on the absolute change in position
        trade_amount = position_change * self.trade_size  # Can be negative
        self.step_cost = abs(trade_amount) * self.transaction_cost_rate
        self.total_transaction_costs += self.step_cost

        # Order tracking logic
        # If opening a new position (from 0 to non-zero)
        if prev_position == 0 and desired_position != 0:
            self.order_counter += 1
            self.current_order = {
                'order_id': self.order_counter,
                'open_date': current_date,
                'open_price': current_price,
                'position': desired_position,
                'pnl_usd': -self.step_cost
            }
        # If closing a position (from non-zero to 0)
        elif prev_position != 0 and desired_position == 0:
            if self.current_order is not None:
                self.close_position(current_date, current_price)
                # self.current_order['close_date'] = current_date
                # self.current_order['close_price'] = current_price
                # # Calculate PnL
                # pnl_usd = (
                #     current_price - self.current_order['open_price']) * self.current_order['position'] * self.trade_size
                # pnl_usd -= self.step_cost  # Subtract transaction cost
                # self.current_order['pnl_usd'] = pnl_usd
                # self.current_order['pnl_pct'] = pnl_usd / \
                #     self.initial_balance * 100
                # self.orders.append(self.current_order)
                # self.current_order = None
        # If changing position (from +1 to -1 or vice versa)
        elif prev_position != desired_position and desired_position != 0 and prev_position != 0:
            # Close the current order
            if self.current_order is not None:
                self.current_order['close_date'] = current_date
                self.current_order['close_price'] = current_price
                pct_chg = current_price / self.current_order['open_price'] - 1
                pnl_usd = pct_chg * \
                    self.current_order['position'] * self.trade_size
                pnl_usd -= self.step_cost / 2
                self.current_order['pnl_usd'] += pnl_usd
                self.current_order['pnl_pct'] = pnl_usd / \
                    self.initial_balance * 100
                self.current_order['holding_period'] = (
                    self.current_order['close_date'] - self.current_order['open_date']).days
                self.orders.append(self.current_order)
            # Open a new order in the opposite direction
            self.order_counter += 1
            self.current_order = {
                'order_id': self.order_counter,
                'open_date': current_date,
                'open_price': current_price,
                'position': desired_position,
                'pnl_usd': -self.step_cost / 2
            }
        # If holding the same position, do nothing

    def get_prices(self, truncated, step):
        if not truncated:
            price = self.df['close'].iloc[step]
        else:
            price = self.df['close'].iloc[step-1]

        return float(price)

    def step(self, action):
        done = truncated = False
        self.step_cost = 0.0
        # Check if episode should terminate before proceeding
        if self.current_step >= len(self.df) - 1:
            truncated = True
            observation = self._get_observation()
            info = {}
            reward = 0.0
            return observation, reward, True, truncated, info

        current_date = self.df['timestamp'].iloc[self.current_step]
        prev_position = self.current_position

        # Map discrete action to desired position
        if action == 0:
            desired_position = -1  # Sell
        elif action == 1:
            desired_position = 0   # Hold/Close
        elif action == 2:
            desired_position = 1   # Buy
        else:
            raise ValueError(f"Invalid action: {action}")

        position_change = desired_position - self.current_position
        self.current_position = desired_position

        current_price = self.get_prices(truncated, self.current_step)
        # Increment step
        self.current_step += 1
        # Check if truncated
        truncated = self.current_step >= len(self.df) - 1
        next_price = self.get_prices(truncated, self.current_step)

        self.prev_balance = self.balance
        if position_change != 0:
            self._handle_trading(
                prev_position, desired_position, position_change, current_price, current_date)

        reward, pnl = self._calculate_reward(next_price, current_price)

        if self.balance <= 0:
            done = True
        # Prepare observation

        observation = self._get_observation()

        info = {}

        if done or truncated:
            total_pnl = self.balance - self.initial_balance
            total_pnl_percentage = (total_pnl / self.initial_balance) * 100
            self.episode_pnls.append(total_pnl_percentage)
            self._print_after_episode()
            # self._print_info_after_episode()
            self.get_order_history()
            # Close any open orders at the end of the episode
            if self.current_order is not None:
                final_date = self.df['date'].iloc[self.current_step - 1]
                final_price = self.df['close'].iloc[self.current_step - 1]
                self.current_order['close_date'] = final_date
                self.current_order['close_price'] = final_price
                pnl_usd = (
                    final_price - self.current_order['open_price']) * self.current_order['position'] * self.trade_size
                self.current_order['pnl_usd'] = pnl_usd
                self.current_order['pnl_pct'] = pnl_usd / \
                    self.initial_balance * 100
                self.orders.append(self.current_order)
                self.current_order = None
        if self.track_orders:
            self.step_track(prev_position,
                            self.current_position,
                            position_change,
                            pnl,
                            current_date,
                            current_price,
                            next_price,
                            reward,
                            self.balance,
                            self.order_counter,
                            )
        if self.track_orders:
            self._track_for_backtesting(
                current_date, pnl, self.current_position)

        if self.balance <= 0:
            print(f'ERROR: balance is: {self.balance}')

        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            print(
                f"WARNING: Observation contains NaN or inf values at step {self.current_step}")
            print(f'length of df = {len(self.df)}')
            print(f'observation: {observation}')
            print(f'Loaded file: {self.loaded_file}')
        if np.isnan(reward) or np.isinf(reward):
            print(
                f"WARNING: Reward contains NaN or inf value at step {self.current_step}: {reward}")
        return observation, reward, done, truncated, info

    def _track_for_backtesting(self, current_date, pnl, current_position) -> None:
        self.backtest_date_array.append(current_date)
        self.backtest_pnl_array.append(pnl)
        self.backtest_current_pos_array.append(current_position)
        pass

    def _track_orders(self) -> None:
        pass

    def step_track(self, prev_position, current_position, position_change, pnl, date, close_price, next_price, reward, balance, num_orders) -> None:
        self.prev_position_array.append(prev_position)
        self.current_position_array.append(current_position)
        self.position_chg_array.append(position_change)
        self.pnl_array.append(pnl)
        self.date_array.append(date)
        self.price_array.append(close_price)
        self.cum_cost_array.append(self.total_transaction_costs)
        self.next_price_array.append(next_price)
        self.reward_array.append(reward)
        self.balance_array.append(balance)
        self.num_orders.append(num_orders)
        pass

    def get_backtest_data(self):
        data = {
            'date': self.backtest_date_array,
            'current_position': self.backtest_current_pos_array,
            'pnl': self.backtest_pnl_array,
        }
        backtest_df = pd.DataFrame(data)
        return backtest_df

    def get_tracking_data(self):
        """Constructs a pandas DataFrame from the tracking arrays."""
        data = {
            'date': self.date_array,
            'prev_position': self.prev_position_array,
            'current_position': self.current_position_array,
            'position_change': self.position_chg_array,
            'close_price': self.price_array,
            'next_price': self.next_price_array,
            'pnl': self.pnl_array,
            'reward': self.reward_array,
            'cumulative_cost': self.cum_cost_array,
            'balance': self.balance_array,
            'num_order': self.num_orders,
        }
        tracking_df = pd.DataFrame(data)
        # Format pnl column to display 2 decimal places
        tracking_df['pnl'] = tracking_df['pnl']
        tracking_df['cumulative_cost'] = tracking_df['cumulative_cost']
        tracking_df['reward'] = tracking_df['reward']
        return tracking_df

    def get_full_history(self):
        orders_df = pd.DataFrame(self.orders)

    def get_order_history(self):
        """Returns a DataFrame containing the order history."""
        orders_df = pd.DataFrame(self.orders)
        # Calculate percentage change between open and close prices
        if not orders_df.empty:
            orders_df['price_change_pct'] = (
                orders_df['close_price'] / orders_df['open_price'] - 1) * 100

        return orders_df

    def render(self, mode='chart'):
        pass

    def close(self):
        pass

    def _print_after_episode(self):
        """Print episode summary with corrected metrics."""
        profitable_trades = sum(
            1 for order in self.orders if order['pnl_usd'] > 0)
        total_trades = len(self.orders)

        win_rate = (profitable_trades / total_trades) * \
            100 if total_trades > 0 else 0
        total_return = ((self.balance / self.initial_balance) - 1) * 100
        # win_rate = (self.winning_trades / max(1, self.total_trades)) * 100

        print("\nEpisode Summary:")
        print(f"Final Return: {total_return:.2f}%")
        print(f"Total PnL: {self.total_pnl:.2f}")
        print(f"Total Orders: {self.order_counter}")
        print(f"Winning Trades: {profitable_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Initial Balance: {self.initial_balance:.2f}")
        print(f"Final Balance: {self.balance:.2f}")
        print("-" * 50)
        pass

    def _print_info_after_episode(self) -> None:
        self.total_pnl_pct = ((self.balance / self.initial_balance)-1)*100
        episode_data = {
            'episode_pnl': (self.balance - self.initial_balance),
            'total_pnl_pct': (((self.balance / self.initial_balance) - 1) * 100),
            'current_step': self.current_step,
            'order_counter': self.order_counter,
            'total_transaction_costs': self.total_transaction_costs,
            'truncated': self.current_step >= len(self.df) - 1,
            'done': self.balance <= 0,
            'loaded_file': self.loaded_file,  # Assume you store the loaded file name
        }
        self.episode_metrics.append(episode_data)
        if self.running_mode == 'training':
            print(
                f'TRAIN Episode end - Balance: {self.balance:.2f} - P/L(%): {((self.balance / self.initial_balance)-1)*100:.2f} - Steps & Trades: {self.current_step} & {self.order_counter}')
        elif self.running_mode == 'validating':
            print(
                f'VALIDATE Episode end - Balance: {self.balance:.2f} - P/L(%): {((self.balance / self.initial_balance)-1)*100:.2f} - Steps & Trades: {self.current_step} & {self.order_counter}')
        elif self.running_mode == 'testing':
            print(
                f'TESTING Episode end - Balance: {self.balance:.2f} - P/L(%): {((self.balance / self.initial_balance)-1)*100:.2f} - Steps & Trades: {self.current_step} & {self.order_counter}')
        else:
            print('No running mode specified')
        pass

    def _load_df(self):
        dir_path = '/Users/floriankockler/Code/data_temp/'

        if not os.path.exists(dir_path):
            dir_path = '/Users/floriankockler/Code/sb3/datasets'
        if self.running_mode == 'validating':
            if self.validation_file_paths:
                file_path = random.choice(self.validation_file_paths)
            elif self.validation_file_path:
                file_path = self.validation_file_path
            else:
                selected_file = 'USDJPY.FOREX_1d_with_tech_raw.parquet'
                file_path = os.path.join(dir_path, selected_file)

        elif self.load_specific is not None:
            file_path = self.load_specific
        else:
            # Get all parquet files in the directory
            parquet_files = [f for f in os.listdir(
                dir_path) if f.endswith('.parquet') and not f.startswith('.')]
            # Filter files greater than 80MB
            large_files = [f for f in parquet_files if os.path.getsize(
                os.path.join(dir_path, f)) > 5 * 1024 * 1024]
            if not large_files:
                raise ValueError(
                    "No parquet files larger than 80MB found in the directory.")
            # Select a random file from the filtered list
            selected_file = random.choice(large_files)
            file_path = os.path.join(dir_path, selected_file)
        self.loaded_file = file_path
        logging.info(f"Loaded file: {file_path}")

        loaded_df = AddTechnicalIndicators(
            load_specific=file_path,
            normalize=True,
            compute_indicators_on='daily',
            resample_interval='1h',
        )

        # Check if loaded_df is a DataFrame

        if self.running_mode == 'validating':
            train_df, validate_df, _ = loaded_df.split_train_validate_test()
            df = validate_df.reset_index()
            # Check for NaN or infinite values
            if df.isna().any().any() or np.isinf(df.replace([np.inf, -np.inf], np.nan)).any().any():
                print(
                    f"Dataset contains NaN or infinite values in file: {file_path}")
                # Print columns containing NaN or inf
                nan_cols = df.columns[df.isna().any()].tolist()
                inf_cols = df.columns[np.isinf(df.replace(
                    [np.inf, -np.inf], np.nan)).any()].tolist()
                if nan_cols:
                    print(f"Columns with NaN values: {nan_cols}")
                if inf_cols:
                    print(f"Columns with infinite values: {inf_cols}")
            logging.info('validate df loaded')
        elif self.running_mode == 'testing':

            self.ticker = os.path.splitext(os.path.basename(file_path))[0]
            _, _, test_df = loaded_df.split_train_validate_test()
            df = test_df.reset_index()
            # Check for NaN or infinite values
            if df.isna().any().any() or np.isinf(df.replace([np.inf, -np.inf], np.nan)).any().any():
                print(
                    f"Dataset contains NaN or infinite values in file: {file_path}")
                # Print columns containing NaN or inf
                nan_cols = df.columns[df.isna().any()].tolist()
                inf_cols = df.columns[np.isinf(df.replace(
                    [np.inf, -np.inf], np.nan)).any()].tolist()
                if nan_cols:
                    print(f"Columns with NaN values: {nan_cols}")
                if inf_cols:
                    print(f"Columns with infinite values: {inf_cols}")
            logging.info('testing df loaded')
        else:
            train_df, _, _ = loaded_df.split_train_validate_test()
            df = train_df.reset_index()
            # Check for NaN or infinite values
            if df.isna().any().any() or np.isinf(df.replace([np.inf, -np.inf], np.nan)).any().any():
                print(
                    f"Dataset contains NaN or infinite values in file: {file_path}")
                # Print columns containing NaN or inf
                nan_cols = df.columns[df.isna().any()].tolist()
                inf_cols = df.columns[np.isinf(df.replace(
                    [np.inf, -np.inf], np.nan)).any()].tolist()
                if nan_cols:
                    print(f"Columns with NaN values: {nan_cols}")
                if inf_cols:
                    print(f"Columns with infinite values: {inf_cols}")
            logging.info('train df loaded')

        return df
