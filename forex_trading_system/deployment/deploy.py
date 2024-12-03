from data_management.preprocessor import DataPreprocessor
from data_management.indicator_manager import IndicatorManager
from typing import Optional, Tuple
from stable_baselines3.common.vec_env import VecNormalize
import oandapyV20.endpoints.accounts as accounts
from typing import Dict, Optional
from dataclasses import dataclass
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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import time
import os
import sys
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialize IndicatorManager and DataProcessor
indicator_manager = IndicatorManager()
data_processor = DataPreprocessor()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('deploy.py')

# OANDA API setup
# Ensure that API credentials are loaded securely, e.g., from environment variables
OANDA_API_KEY = '9317ace4596d61e3e98b1a53b2342483-45d3ad4084c80b111727a9fada9ef0ff'
OANDA_ACCOUNT_ID = '101-004-30348600-002'
OANDA_ENV = 'practice'  # or 'practice'

# Initialize OANDA API client
client = API(access_token=OANDA_API_KEY, environment=OANDA_ENV)

# Currency pairs
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
    'XAU_USD': 37.68,  # Gold
    'XAG_USD': 3_266  # Silver
}


# Global variables for data and positions
data_storage = {}
positions_storage = {}
data_lock = threading.Lock()
positions_lock = threading.Lock()

# Load models and normalization parameters
models = {}
for pair in currency_pairs:
    # Load the model
    model_path = f'./models/{pair}_model.zip'
    env_path = f'./models/{pair}_vecnormalize.pkl'

    # Load the environment normalization stats
    vec_normalize = VecNormalize.load(env_path)
    # Create a dummy environment and set the statistics
    env = DummyVecEnv([lambda: ForexTradingEnv(None, pair)])
    env = VecNormalize(env)
    env.obs_rms = vec_normalize.obs_rms
    env.ret_rms = vec_normalize.ret_rms
    env.training = False
    env.norm_reward = False

    # Load the model
    model = PPO.load(model_path, env=env)
    models[pair] = model


def calculate_indicators(df):
    # Resample 5-minute data to 1-hour data for indicator calculation
    df_hourly = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }).dropna()

    # Calculate indicators on 1-hour data using IndicatorManager
    indicator_df = indicator_manager.calculate_indicators(
        df_hourly,
        selected_indicators=None,  # Calculate all indicators
        indicator_timeframe=None   # Indicators are calculated on the resampled data
    )

    # Forward-fill the indicators back to the 5-minute DataFrame
    df = df.merge(indicator_df, left_index=True, right_index=True, how='left')
    df.fillna(method='ffill', inplace=True)

    # Drop rows with NaN values (if any remain)
    df.dropna(inplace=True)

    return df


def normalize_data(df):
    # Use DataProcessor to normalize data
    df_normalized = data_processor.normalize_simple(df)
    return df_normalized


def load_data_from_gcs(pair):
    # Initialize GCS client
    storage_client = storage.Client()
    bucket_name = 'fx_data_sets_15min_1h_tech_norm'  # Replace with your bucket name
    bucket = storage_client.bucket(bucket_name)
    #! change name below
    blob = bucket.blob(f'{pair}_data.parquet')
    temp_file = f'/tmp/{pair}_data.parquet'
    blob.download_to_filename(temp_file)
    df = pd.read_parquet(temp_file)
    # Ensure that timestamp is the index and is timezone-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    return df


def fetch_missing_candles(pair, last_timestamp):
    # Fetch missing candles from OANDA API starting from last_timestamp
    params = {
        "from": last_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "granularity": "M5",  # 5-minute candles
        "price": "M",  # Midpoint prices
    }
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    candles = r.response.get('candles', [])
    df_list = []
    for candle in candles:
        if candle['complete']:
            time_candle = pd.to_datetime(candle['time'])
            open_price = float(candle['mid']['o'])
            high_price = float(candle['mid']['h'])
            low_price = float(candle['mid']['l'])
            close_price = float(candle['mid']['c'])
            volume = int(candle['volume'])
            df_list.append({
                'timestamp': time_candle,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
            })
    if df_list:
        df_new = pd.DataFrame(df_list)
        df_new.set_index('timestamp', inplace=True)
        return df_new
    else:
        return pd.DataFrame()


def fetch_current_positions():
    r = positions.OpenPositions(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    open_positions = r.response.get('positions', [])
    with positions_lock:
        positions_storage.clear()
        for pos in open_positions:
            pair = pos['instrument']
            if pair in currency_pairs:
                long_units = float(pos.get('long', {}).get('units', 0))
                short_units = float(pos.get('short', {}).get('units', 0))
                if long_units > 0:
                    positions_storage[pair] = 'LONG'
                elif short_units < 0:
                    positions_storage[pair] = 'SHORT'
                else:
                    positions_storage[pair] = 'NO_POSITION'
            else:
                positions_storage[pair] = 'NO_POSITION'


def fetch_and_update_data(pair):
    with data_lock:
        df = data_storage.get(pair)
    if df is None:
        # Load existing dataset from GCS
        df = load_data_from_gcs(pair)
    # Check last available candle
    last_timestamp = df.index[-1]
    current_time = datetime.now(datetime.timezone.utc()).replace(
        tzinfo=pd.Timestamp.utcnow().tz)
    if current_time - last_timestamp >= timedelta(minutes=5):
        # Fetch missing candles
        df_new = fetch_missing_candles(pair, last_timestamp)
        if not df_new.empty:
            df = pd.concat([df, df_new])
            df = df[~df.index.duplicated(keep='last')]
            df.sort_index(inplace=True)
            # Calculate technical indicators on the entire DataFrame
            df = calculate_indicators(df)
            # Normalize data
            df = normalize_data(df)
            # Update data storage
            with data_lock:
                data_storage[pair] = df
        else:
            logger.info(f"No new data for {pair}")
    else:
        logger.info(f"No missing data for {pair}")


def update_data():
    for pair in currency_pairs.keys():
        try:
            fetch_and_update_data(pair)
            logger.info(f"Data updated for {pair}")
        except Exception as e:
            logger.error(f"Error updating data for {pair}: {e}")


def predict_and_trade(pair):
    with data_lock:
        df = data_storage[pair]
    # Prepare the sequence data
    sequence_length = 5  # Adjust based on your model
    obs = df.iloc[-sequence_length:]
    # Convert to numpy array
    obs_array = obs.values  # Shape: (sequence_length, num_features)
    # Flatten the observation if needed
    obs_array = obs_array.T.flatten()
    # Reshape to (1, observation_space)
    obs_array = obs_array.reshape((1, -1))
    # Get the model
    model = models[pair]
    # Normalize the observation
    obs_array = model.env.normalize_obs(obs_array)
    # Predict
    action, _ = model.predict(obs_array)
    # Map action to position
    action_mapping = {0: 'NO_POSITION', 1: 'LONG', 2: 'SHORT'}
    action_name = action_mapping.get(action[0])
    with positions_lock:
        current_position = positions_storage.get(pair, 'NO_POSITION')
    if current_position != action_name:
        # Execute trade via OANDA API
        # Close existing position if any
        if current_position != 'NO_POSITION':
            close_position(pair, current_position)
        # Open new position if action_name is not NO_POSITION
        if action_name != 'NO_POSITION':
            open_position(pair, action_name)
        with positions_lock:
            positions_storage[pair] = action_name
        logger.info(f"Executed action {action_name} for {pair}")
    else:
        logger.info(
            f"No action needed for {pair}, current position is {current_position}")


def open_position(pair, action_name):
    # Define the order data
    units = currency_pairs[pair]  # Get the notional amount for this pair
    if action_name == 'LONG':
        units = units  # Positive units for buy
    elif action_name == 'SHORT':
        units = -units  # Negative units for sell
    else:
        return
    data = {
        "order": {
            "instrument": pair,
            "units": str(units),
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }
    }
    r = orders.OrderCreate(accountID=OANDA_ACCOUNT_ID, data=data)
    client.request(r)
    logger.info(f"Opened {action_name} position for {pair}")


def close_position(pair, current_position):
    data = {}
    if current_position == 'LONG':
        data = {
            "longUnits": "ALL"
        }
    elif current_position == 'SHORT':
        data = {
            "shortUnits": "ALL"
        }
    else:
        return
    r = positions.PositionClose(
        accountID=OANDA_ACCOUNT_ID, instrument=pair, data=data)
    client.request(r)
    logger.info(f"Closed {current_position} position for {pair}")


def sync_positions():
    fetch_current_positions()
    logger.info("Positions synchronized with OANDA API")


def process_pair(pair):
    try:
        predict_and_trade(pair)
    except Exception as e:
        logger.error(f"Error processing {pair}: {str(e)}")


def run_trading_cycle():
    logger.info("Starting trading cycle")
    # Update data first
    update_data()
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(process_pair, currency_pairs.keys())
    logger.info("Trading cycle completed")


def initialize():
    # Initialize data storage
    for pair in currency_pairs.keys():
        try:
            # Load existing dataset from GCS
            df = load_data_from_gcs(pair)
            # Ensure that timestamp is the index and is timezone-aware
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            # Check for missing data and fetch if necessary
            fetch_and_update_data(pair)
            # Calculate technical indicators on the entire DataFrame
            df = calculate_indicators(df)
            # Normalize data
            df = normalize_data(df)
            # Update data storage
            with data_lock:
                data_storage[pair] = df
            logger.info(f"Initialized data for {pair}")
        except Exception as e:
            logger.error(f"Error initializing data for {pair}: {e}")
    # Synchronize positions at startup
    sync_positions()


initialize()
# Scheduler setup
scheduler = BackgroundScheduler()
scheduler.add_job(run_trading_cycle, 'cron', minute='*/5', second=0)
scheduler.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
    logger.info("Scheduler shut down")
