# File: trading/model_manager.py
# Path: forex_trading_system/trading/model_manager.py

# File: trading/model_manager.py

from typing import Dict, Optional, Tuple, List
import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import hashlib
import shutil
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from .agents.training_agent import TrainingAgent
from .environments.forex_env import ForexTradingEnv


@dataclass
class ModelMetrics:
    """Tracks model performance metrics."""
    # Performance metrics
    total_pnl: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_duration: float
    trades_per_day: float

    # Risk metrics
    value_at_risk: float
    expected_shortfall: float
    max_consecutive_losses: int
    position_ratios: Dict[str, float]

    # Training info
    training_time: float
    total_steps: int
    final_loss: float
    evaluation_metrics: Dict[str, float]

    def is_better_than(self, other: 'ModelMetrics', threshold: float = 1.05) -> bool:
        """Compare if this model is significantly better than another."""
        if other is None:
            return True

        improvements = [
            self.sharpe_ratio > other.sharpe_ratio * threshold,
            self.win_rate > other.win_rate * threshold,
            self.max_drawdown < other.max_drawdown / threshold,
            self.total_pnl > other.total_pnl * threshold
        ]

        return sum(improvements) >= 3  # Require at least 3 improvements


@dataclass
class ModelVersion:
    """Tracks information about a specific model version."""
    version_id: str
    timestamp: datetime
    metrics: ModelMetrics
    training_params: Dict
    data_hash: str
    is_deployed: bool = False
    performance_history: List[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp.isoformat(),
            'metrics': asdict(self.metrics),
            'training_params': self.training_params,
            'data_hash': self.data_hash,
            'is_deployed': self.is_deployed,
            'performance_history': self.performance_history or []
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['metrics'] = ModelMetrics(**data['metrics'])
        return cls(**data)


class ModelManager:
    """Manages model lifecycle including training, evaluation, and versioning."""

    def __init__(
        self,
        base_path: str = "/models",
        n_envs: int = 4,
        verbose: int = 1
    ):
        """
        Initialize model manager.

        Args:
            base_path: Base path for model storage
            n_envs: Number of environments for training
            verbose: Verbosity level
        """
        self.base_path = Path(base_path)
        self.n_envs = n_envs
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

        # Ensure directories exist
        self.base_path.mkdir(exist_ok=True)
        (self.base_path / "versions").mkdir(exist_ok=True)
        (self.base_path / "deployed").mkdir(exist_ok=True)

        # Track deployed models
        self.deployed_models: Dict[str, ModelVersion] = {}
        self._load_deployed_models()

        # Default training parameters
        self.default_params = {
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'learning_rate': 3e-4,
            'ent_coef': 0.01,
            'clip_range': 0.2,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            )
        }

    def train_model(
        self,
        df: pd.DataFrame,
        pair: str,
        params: Optional[Dict] = None,
        total_timesteps: int = 1_000_000,
        eval_freq: int = 15_000
    ) -> Tuple[PPO, ModelMetrics]:
        """
        Train a new model version.

        Args:
            df: Training data
            pair: Currency pair
            params: Training parameters (use default if None)
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
        """
        self.logger.info(f"Starting training for {pair}")
        start_time = datetime.now()

        # Calculate data hash for version tracking
        data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df).values
        ).hexdigest()

        # Initialize training agent
        agent = TrainingAgent(
            pair=pair,
            save_path=self.base_path / "versions" / pair,
            n_envs=self.n_envs
        )

        try:
            # Train model
            model, metrics = agent.train(
                df=df,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq
            )

            # Add training time to metrics
            metrics.training_time = (
                datetime.now() - start_time).total_seconds()

            # Create new version
            version = self._create_version(
                pair=pair,
                metrics=metrics,
                params=params or self.default_params,
                data_hash=data_hash
            )

            # Save version info
            self._save_version(pair, version, model)

            # Check if we should deploy
            if self._should_deploy(version, pair):
                self._deploy_model(pair, version, model)

            return model, metrics

        except Exception as e:
            self.logger.error(f"Training failed for {pair}: {str(e)}")
            raise

    def get_deployed_model(
        self,
        pair: str
    ) -> Tuple[Optional[PPO], Optional[ModelVersion]]:
        """Get currently deployed model for a pair."""
        if pair not in self.deployed_models:
            return None, None

        version = self.deployed_models[pair]
        model_path = self.base_path / "deployed" / pair / "model"

        try:
            model = PPO.load(model_path)
            return model, version
        except Exception as e:
            self.logger.error(
                f"Error loading deployed model for {pair}: {str(e)}")
            return None, version

    def evaluate_model(
        self,
        model: PPO,
        df: pd.DataFrame,
        pair: str,
        n_evaluations: int = 5
    ) -> ModelMetrics:
        """
        Evaluate model performance on specific data.

        Args:
            model: Model to evaluate
            df: Evaluation data
            pair: Currency pair
            n_evaluations: Number of evaluation runs
        """
        agent = TrainingAgent(
            pair=pair,
            save_path=self.base_path / "evaluation" / pair,
            n_envs=1  # Single env for evaluation
        )

        metrics_list = []
        for _ in range(n_evaluations):
            env = agent.create_env(df, is_training=False)
            metrics = agent._calculate_metrics(model, env)
            metrics_list.append(metrics)
            env.close()

        # Average metrics
        return self._average_metrics(metrics_list)

    def update_deployment(
        self,
        pair: str,
        metrics: ModelMetrics
    ) -> bool:
        """
        Update deployment metrics and check for retraining needs.

        Args:
            pair: Currency pair
            metrics: Latest performance metrics

        Returns:
            bool: True if retraining is recommended
        """
        if pair not in self.deployed_models:
            return False

        version = self.deployed_models[pair]

        # Update performance history
        if version.performance_history is None:
            version.performance_history = []

        version.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics)
        })

        # Save updated version info
        self._save_version(pair, version)

        # Check if retraining is needed
        return self._needs_retraining(version, metrics)

    def _create_version(
        self,
        pair: str,
        metrics: ModelMetrics,
        params: Dict,
        data_hash: str
    ) -> ModelVersion:
        """Create new model version."""
        version_id = f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ModelVersion(
            version_id=version_id,
            timestamp=datetime.now(),
            metrics=metrics,
            training_params=params,
            data_hash=data_hash,
            performance_history=[]
        )

    def _save_version(
        self,
        pair: str,
        version: ModelVersion,
        model: Optional[PPO] = None
    ) -> None:
        """Save version information and optionally the model."""
        version_dir = self.base_path / "versions" / pair / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save version info
        with open(version_dir / "version.json", 'w') as f:
            json.dump(version.to_dict(), f, indent=2)

        # Save model if provided
        if model is not None:
            model.save(version_dir / "model")

    def _should_deploy(self, version: ModelVersion, pair: str) -> bool:
        """Decide if new version should be deployed."""
        if pair not in self.deployed_models:
            return True

        current_version = self.deployed_models[pair]
        return version.metrics.is_better_than(current_version.metrics)

    def _deploy_model(
        self,
        pair: str,
        version: ModelVersion,
        model: PPO
    ) -> None:
        """Deploy a model version."""
        deploy_dir = self.base_path / "deployed" / pair

        # Create deployment directory
        deploy_dir.mkdir(parents=True, exist_ok=True)

        # Save model and normalizer
        model.save(deploy_dir / "model")
        if isinstance(model.env, VecNormalize):
            model.env.save(deploy_dir / "vec_normalize.pkl")

        # Update version status
        version.is_deployed = True
        self._save_version(pair, version)

        # Update deployed models registry
        self.deployed_models[pair] = version

        self.logger.info(
            f"Deployed new model version {version.version_id} for {pair}")

    def _needs_retraining(
        self,
        version: ModelVersion,
        current_metrics: ModelMetrics
    ) -> bool:
        """Determine if model needs retraining based on performance degradation."""
        if not version.performance_history:
            return False

        # Calculate performance trends
        recent_metrics = [ModelMetrics(**h['metrics'])
                          for h in version.performance_history[-10:]]

        # Check for significant degradation
        degradation_checks = [
            current_metrics.sharpe_ratio < 0.7 * version.metrics.sharpe_ratio,
            current_metrics.win_rate < 0.8 * version.metrics.win_rate,
            current_metrics.max_drawdown > 1.5 * version.metrics.max_drawdown
        ]

        return sum(degradation_checks) >= 2

    def _average_metrics(self, metrics_list: List[ModelMetrics]) -> ModelMetrics:
        """Calculate average metrics across multiple evaluations."""
        metrics_dict = {field: []
                        for field in ModelMetrics.__dataclass_fields__}

        for metrics in metrics_list:
            for field, value in asdict(metrics).items():
                metrics_dict[field].append(value)

        averaged = {
            field: (np.mean(values) if field !=
                    'position_ratios' else values[0])
            for field, values in metrics_dict.items()
        }

        return ModelMetrics(**averaged)

    def _load_deployed_models(self) -> None:
        """Load information about currently deployed models."""
        deployed_dir = self.base_path / "deployed"

        if not deployed_dir.exists():
            return

        for pair_dir in deployed_dir.iterdir():
            if pair_dir.is_dir():
                version_file = pair_dir / "version.json"
                if version_file.exists():
                    with open(version_file, 'r') as f:
                        version_data = json.load(f)
                        version = ModelVersion.from_dict(version_data)
                        self.deployed_models[pair_dir.name] = version


# from typing import Dict, Optional, Tuple, List
# import os
# from pathlib import Path
# import json
# import torch
# import numpy as np
# import pandas as pd
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
# import optuna
# from datetime import datetime
# import logging
# from dataclasses import dataclass
# from collections import defaultdict
# from environments.forex_env import ForexTradingEnv
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots


# @dataclass
# class ModelMetrics:
#     """Tracks model performance metrics."""
#     # Episode-level metrics
#     total_rewards: float
#     total_pnl: float  # Actual PnL in base currency
#     win_rate: float
#     max_drawdown: float
#     sharpe_ratio: float
#     sortino_ratio: float

#     # Trade-level metrics
#     total_trades: int
#     trades_per_episode: float  # Average number of trades per episode
#     avg_pnl_per_trade: float  # Average PnL per trade in base currency
#     total_trading_costs: float
#     avg_trade_duration: float  # Average holding time in hours
#     # Ratio of time spent in each position type
#     position_ratios: Dict[str, float]

#     # Training metrics
#     training_time: float
#     episodes_completed: int


# class CustomCallback(BaseCallback):
#     """Custom callback for monitoring training progress."""

#     def __init__(self, check_freq: int, model_path: str, verbose: int = 1):
#         super(CustomCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.model_path = model_path
#         self.best_mean_reward = -np.inf

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:
#             # Get current reward
#             x, y = self.model.ep_info_buffer.get_mean_reward()
#             mean_reward = y[-1] if len(y) > 0 else -np.inf

#             # Save best model
#             if mean_reward > self.best_mean_reward:
#                 self.best_mean_reward = mean_reward
#                 self.model.save(
#                     os.path.join(self.model_path, 'best_model')
#                 )
#         return True


# class ModelManager:
#     """Manages model training, evaluation, and versioning."""

#     def __init__(
#         self,
#         base_path: str = "/Volumes/ssd_fat2/ai6_trading_bot/models",
#         n_envs: int = 4,
#         verbose: int = 1
#     ):
#         """
#         Initialize model manager.

#         Args:
#             base_path: Path for model storage
#             n_envs: Number of environments for parallel training
#             verbose: Verbosity level
#         """
#         self.base_path = Path(base_path)
#         self.n_envs = n_envs
#         self.verbose = verbose
#         self.logger = logging.getLogger(__name__)
#         self.normalizers = {}

#         # Ensure directories exist
#         self.base_path.mkdir(exist_ok=True)
#         (self.base_path / "checkpoints").mkdir(exist_ok=True)
#         (self.base_path / "metrics").mkdir(exist_ok=True)

#         # Default hyperparameters (can be optimized by Optuna)
#         self.default_params = {
#             'n_steps': 2048,
#             'batch_size': 64,
#             'n_epochs': 10,
#             'learning_rate': 3e-4,
#             'ent_coef': 0.01,
#             'clip_range': 0.2,
#             'gae_lambda': 0.95,
#             'max_grad_norm': 0.5,
#             'vf_coef': 0.5,
#             'policy_kwargs': dict(
#                 net_arch=[dict(pi=[64, 64], vf=[64, 64])]
#             )
#         }

#     def create_env(
#         self,
#         df: pd.DataFrame,
#         pair: str,
#         is_train: bool = True,
#         n_envs: Optional[int] = None
#     ) -> VecNormalize:
#         """Create normalized vectorized environments."""
#         n_envs = n_envs or self.n_envs if is_train else 1

#         def create_env(
#             self,
#             df: pd.DataFrame,
#             pair: str,
#             is_train: bool = True,
#             n_envs: Optional[int] = None
#         ) -> VecNormalize:
#             """Create normalized vectorized environments."""
#             n_envs = n_envs or self.n_envs if is_train else 1

#             def make_env(rank: int):
#                 def _init():
#                     env = ForexTradingEnv(
#                         df=df,
#                         pair=pair,
#                         random_start=is_train
#                     )
#                     if rank == 0:
#                         check_env(env)
#                     env = Monitor(env, info_keywords=(
#                         'total_pnl',
#                         'trade_count',
#                         'trading_costs',
#                         'position_duration',
#                         'position_type'
#                     ))
#                     return env
#                 return _init

#             vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)]) if n_envs > 1 \
#                 else DummyVecEnv([make_env(0)])

#             # Create or load normalizer
#             if is_train:
#                 norm_env = VecNormalize(
#                     vec_env,
#                     norm_obs=True,
#                     norm_reward=True,
#                     clip_obs=10.,
#                     clip_reward=10.,
#                     gamma=0.99,
#                     epsilon=1e-8
#                 )
#                 self.normalizers[pair] = norm_env
#             else:
#                 # For evaluation, load existing normalizer if available
#                 norm_env = self.normalizers.get(pair, None)
#                 if norm_env is None:
#                     norm_env = VecNormalize(
#                         vec_env,
#                         norm_obs=True,
#                         norm_reward=False,  # Don't normalize rewards during evaluation
#                         training=False
#                     )
#                 else:
#                     norm_env = VecNormalize.load(
#                         self.base_path / pair / "vec_normalize.pkl",
#                         vec_env
#                     )
#                     norm_env.training = False  # Don't update normalizer during evaluation

#             return norm_env

#     def train_model(
#         self,
#         df: pd.DataFrame,
#         pair: str,
#         params: Optional[Dict] = None,
#         total_timesteps: int = 1_000_000,
#         checkpoint_freq: int = 10000
#     ) -> Tuple[PPO, ModelMetrics]:
#         """
#         Train a new model for a currency pair.

#         Args:
#             df: Training data
#             pair: Currency pair
#             params: Model hyperparameters (use default if None)
#             total_timesteps: Total training steps
#             checkpoint_freq: Frequency of checkpoints
#         """
#         self.logger.info(f"Starting training for {pair}")
#         start_time = datetime.now()

#         # Create vectorized environment
#         env = self.create_env(df, pair)

#         # Initialize model
#         model = PPO(
#             "MultiInputPolicy",
#             env,
#             verbose=self.verbose,
#             tensorboard_log=str(self.base_path / "logs"),
#             **params or self.default_params
#         )

#         # Setup callbacks
#         checkpoint_path = str(self.base_path / "checkpoints" / pair)
#         callbacks = [
#             CustomCallback(
#                 check_freq=checkpoint_freq,
#                 model_path=checkpoint_path
#             ),
#             CheckpointCallback(
#                 save_freq=checkpoint_freq,
#                 save_path=checkpoint_path,
#                 name_prefix=pair
#             )
#         ]

#         # Train model
#         try:
#             model.learn(
#                 total_timesteps=total_timesteps,
#                 callback=callbacks,
#                 progress_bar=True
#             )

#             # Evaluate model
#             metrics = self.evaluate_model(model, df, pair)

#             # Save metrics
#             metrics.training_time = (
#                 datetime.now() - start_time).total_seconds()
#             self._save_metrics(pair, metrics)

#             # Save final model
#             final_path = self.base_path / pair / f"final_model"
#             final_path.parent.mkdir(exist_ok=True)
#             model.save(str(final_path))

#             # Save normalizer along with the model
#             if pair in self.normalizers:
#                 normalizer_path = self.base_path / pair / "vec_normalize.pkl"
#                 self.normalizers[pair].save(str(normalizer_path))

#             return model, metrics

#         except Exception as e:
#             self.logger.error(f"Training failed for {pair}: {str(e)}")
#             raise

#     def evaluate_model(
#         self,
#         model: PPO,
#         df: pd.DataFrame,
#         pair: str,
#         n_evaluations: int = 5,
#         eval_window_size: Optional[int] = None,
#         use_validation_set: bool = True
#     ) -> Tuple[ModelMetrics, List[Dict]]:
#         """Evaluate model performance across different time periods."""
#         # Split data for validation if requested
#         if use_validation_set:
#             split_idx = int(len(df) * 0.8)
#             eval_df = df[split_idx:]
#         else:
#             eval_df = df

#         # Set evaluation window size
#         if eval_window_size is None:
#             eval_window_size = len(eval_df) // 5

#         evaluation_periods = []
#         episode_metrics = defaultdict(list)

#         # Evaluate on different time periods
#         for i in range(n_evaluations):
#             if len(eval_df) <= eval_window_size:
#                 start_idx = 0
#             else:
#                 max_start = len(eval_df) - eval_window_size
#                 start_idx = i * (max_start // n_evaluations)

#             end_idx = min(start_idx + eval_window_size, len(eval_df))
#             evaluation_df = eval_df[start_idx:end_idx]

#             # Create environment with specific data window
#             env = self.create_env(
#                 df=evaluation_df,
#                 pair=pair,
#                 is_train=False,
#                 n_envs=1
#             )

#             # Run evaluation episode
#             obs = env.reset()
#             done = False
#             episode_reward = 0
#             episode_pnl = 0
#             episode_trade_count = 0
#             episode_cost = 0
#             trade_pnls = []
#             trade_durations = []

#             while not done:
#                 action, _ = model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)
#                 info = info[0]

#                 episode_reward += reward
#                 episode_pnl += info['total_pnl']
#                 episode_cost += info['trading_costs']

#                 if info.get('trade_closed', False):
#                     trade_pnls.append(info['trade_pnl'])
#                     trade_durations.append(info['trade_duration'])
#                     episode_trade_count += 1

#             # Store period results
#             evaluation_periods.append({
#                 'start_time': evaluation_df.index[0],
#                 'end_time': evaluation_df.index[-1],
#                 'pnl': episode_pnl,
#                 'pnl_series': trade_pnls,
#                 'trades': episode_trade_count,
#                 'costs': episode_cost,
#                 'avg_trade_duration': np.mean(trade_durations) if trade_durations else 0,
#                 'avg_pnl_per_trade': np.mean(trade_pnls) if trade_pnls else 0
#             })

#             # Collect metrics
#             episode_metrics['rewards'].append(episode_reward)
#             episode_metrics['pnls'].append(episode_pnl)
#             episode_metrics['trades'].append(episode_trade_count)
#             episode_metrics['costs'].append(episode_cost)
#             episode_metrics['trade_pnls'].extend(trade_pnls)
#             episode_metrics['trade_durations'].extend(trade_durations)

#         # Calculate aggregate metrics
#         metrics = ModelMetrics(
#             total_rewards=np.mean(episode_metrics['rewards']),
#             total_pnl=np.mean(episode_metrics['pnls']),
#             win_rate=np.mean(
#                 [pnl > 0 for pnl in episode_metrics['trade_pnls']]),
#             max_drawdown=self._calculate_max_drawdown(
#                 np.cumsum(episode_metrics['pnls'])),
#             sharpe_ratio=self._calculate_sharpe_ratio(
#                 episode_metrics['trade_pnls']),
#             sortino_ratio=self._calculate_sortino_ratio(
#                 episode_metrics['trade_pnls']),
#             total_trades=sum(episode_metrics['trades']),
#             trades_per_episode=np.mean(episode_metrics['trades']),
#             avg_pnl_per_trade=np.mean(
#                 episode_metrics['trade_pnls']) if episode_metrics['trade_pnls'] else 0,
#             total_trading_costs=np.mean(episode_metrics['costs']),
#             avg_trade_duration=np.mean(
#                 episode_metrics['trade_durations']) if episode_metrics['trade_durations'] else 0,
#             position_ratios=self._calculate_position_ratios(
#                 evaluation_periods),
#             training_time=0.0,
#             episodes_completed=n_evaluations
#         )

#         # Create visualization
#         self.visualize_evaluation_results(
#             df=df,
#             metrics=metrics,
#             evaluation_periods=evaluation_periods,
#             save_path=str(self.base_path / pair / "evaluation_results.html")
#         )

#         return metrics, evaluation_periods

#     def optimize_hyperparameters(
#         self,
#         df: pd.DataFrame,
#         pair: str,
#         n_trials: int = 50,
#         n_timesteps: int = 100_000
#     ) -> Dict:
#         """Optimize hyperparameters using Optuna."""
#         def objective(trial):
#             params = {
#                 'n_steps': trial.suggest_int('n_steps', 1024, 4096, 1024),
#                 'batch_size': trial.suggest_int('batch_size', 32, 256, 32),
#                 'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
#                 'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
#                 'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
#                 'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 1.0),
#                 'max_grad_norm': trial.suggest_uniform('max_grad_norm', 0.3, 0.7),
#                 'vf_coef': trial.suggest_uniform('vf_coef', 0.4, 0.6),
#                 'policy_kwargs': dict(
#                     net_arch=[dict(
#                         pi=[64, 64],
#                         vf=[64, 64]
#                     )]
#                 )
#             }

#             model, metrics = self.train_model(
#                 df=df,
#                 pair=pair,
#                 params=params,
#                 total_timesteps=n_timesteps
#             )

#             return metrics.total_rewards

#         study = optuna.create_study(direction='maximize')
#         study.optimize(objective, n_trials=n_trials)

#         return study.best_params

#     def _calculate_max_drawdown(self, balance_history: np.ndarray) -> float:
#         """Calculate maximum drawdown from balance history."""
#         peak = np.maximum.accumulate(balance_history)
#         drawdown = (peak - balance_history) / peak
#         return np.max(drawdown)

#     def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
#         """Calculate Sharpe ratio from returns."""
#         if len(returns) < 2:
#             return 0.0
#         return np.mean(returns) / (np.std(returns) + 1e-6)

#     def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
#         """Calculate Sortino ratio from returns."""
#         if len(returns) < 2:
#             return 0.0
#         negative_returns = returns[returns < 0]
#         downside_std = np.std(negative_returns) if len(
#             negative_returns) > 0 else 1e-6
#         return np.mean(returns) / downside_std

#     def save_metrics(self, pair: str, metrics: ModelMetrics) -> None:
#         """Save enhanced metrics with pretty formatting."""
#         metrics_path = self.base_path / "metrics" / f"{pair}_metrics.json"

#         # Format metrics for better readability
#         formatted_metrics = {
#             "Performance Metrics": {
#                 "Total PnL": f"{metrics.total_pnl:.2f}",
#                 "Win Rate": f"{metrics.win_rate:.2%}",
#                 "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
#                 "Max Drawdown": f"{metrics.max_drawdown:.2%}"
#             },
#             "Trading Statistics": {
#                 "Total Trades": metrics.total_trades,
#                 "Trades per Episode": f"{metrics.trades_per_episode:.1f}",
#                 "Average PnL per Trade": f"{metrics.avg_pnl_per_trade:.2f}",
#                 "Average Trade Duration (hours)": f"{metrics.avg_trade_duration:.1f}",
#                 "Total Trading Costs": f"{metrics.total_trading_costs:.2f}"
#             },
#             "Position Analysis": {
#                 "Position Ratios": {
#                     pos: f"{ratio:.2%}"
#                     for pos, ratio in metrics.position_ratios.items()
#                 }
#             },
#             "Training Info": {
#                 "Training Time (seconds)": metrics.training_time,
#                 "Episodes Completed": metrics.episodes_completed,
#                 "Total Reward": f"{metrics.total_rewards:.2f}"
#             }
#         }

#         with open(metrics_path, 'w') as f:
#             json.dump(formatted_metrics, f, indent=4)

#     def visualize_evaluation_results(
#         self,
#         df: pd.DataFrame,
#         metrics: ModelMetrics,
#         evaluation_periods: List[Dict],
#         save_path: Optional[str] = None
#     ) -> None:
#         """Create comprehensive visualization of evaluation results."""
#         fig = make_subplots(
#             rows=3, cols=2,
#             subplot_titles=(
#                 'Cumulative PnL by Period',
#                 'Performance Metrics',
#                 'Trade Analysis',
#                 'PnL Distribution',
#                 'Position Distribution',
#                 'Trading Activity Heatmap'
#             ),
#             vertical_spacing=0.12,
#             horizontal_spacing=0.1,
#             specs=[
#                 [{"type": "scatter"}, {"type": "bar"}],
#                 [{"type": "scatter"}, {"type": "violin"}],
#                 [{"type": "pie"}, {"type": "heatmap"}]
#             ]
#         )

#         # 1. Cumulative PnL for each period
#         colors = ['blue', 'green', 'red', 'purple', 'orange']
#         for i, period in enumerate(evaluation_periods):
#             period_df = df[period['start_time']:period['end_time']]
#             cumulative_pnl = np.cumsum(period['pnl_series'])

#             fig.add_trace(
#                 go.Scatter(
#                     x=period_df.index,
#                     y=cumulative_pnl,
#                     name=f'Period {i+1}',
#                     line=dict(color=colors[i % len(colors)]),
#                     showlegend=True
#                 ),
#                 row=1, col=1
#             )

#         # 2. Key Performance Metrics
#         metrics_comparison = {
#             'Win Rate': metrics.win_rate,
#             'Trades/Day': metrics.trades_per_episode,
#             'Sharpe': metrics.sharpe_ratio,
#             'Sortino': metrics.sortino_ratio,
#             'Max DD': metrics.max_drawdown
#         }

#         fig.add_trace(
#             go.Bar(
#                 x=list(metrics_comparison.keys()),
#                 y=list(metrics_comparison.values()),
#                 showlegend=False
#             ),
#             row=1, col=2
#         )

#         # 3. Trade Analysis - Scatter plot of trade durations vs. PnL
#         fig.add_trace(
#             go.Scatter(
#                 x=[period['avg_trade_duration']
#                     for period in evaluation_periods],
#                 y=[period['avg_pnl_per_trade']
#                     for period in evaluation_periods],
#                 mode='markers+text',
#                 text=[f'Period {i+1}' for i in range(len(evaluation_periods))],
#                 textposition='top center',
#                 showlegend=False
#             ),
#             row=2, col=1
#         )

#         # 4. PnL Distribution
#         all_pnls = []
#         for period in evaluation_periods:
#             all_pnls.extend(period['pnl_series'])

#         fig.add_trace(
#             go.Violin(
#                 y=all_pnls,
#                 box_visible=True,
#                 meanline_visible=True,
#                 showlegend=False
#             ),
#             row=2, col=2
#         )

#         # 5. Position Distribution
#         position_labels = ['Long', 'Short', 'No Position']
#         position_values = [
#             metrics.position_ratios.get('long', 0),
#             metrics.position_ratios.get('short', 0),
#             metrics.position_ratios.get('none', 0)
#         ]

#         fig.add_trace(
#             go.Pie(
#                 labels=position_labels,
#                 values=position_values,
#                 showlegend=False
#             ),
#             row=3, col=1
#         )

#         # 6. Trading Activity Heatmap
#         activity_matrix = np.zeros((7, 24))  # days x hours
#         for period in evaluation_periods:
#             period_df = df[period['start_time']:period['end_time']]
#             trade_times = period_df[period_df['trade_executed']].index
#             for time in trade_times:
#                 activity_matrix[time.weekday()][time.hour] += 1

#         fig.add_trace(
#             go.Heatmap(
#                 z=activity_matrix,
#                 x=list(range(24)),  # hours
#                 y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
#                 colorscale='Viridis',
#                 showlegend=False
#             ),
#             row=3, col=2
#         )

#         # Update layout
#         fig.update_layout(
#             height=1200,
#             width=1600,
#             title_text="Model Evaluation Results",
#             showlegend=True,
#             legend=dict(
#                 orientation="h",
#                 yanchor="bottom",
#                 y=1.02,
#                 xanchor="right",
#                 x=1
#             )
#         )

#         # Add axis labels
#         fig.update_xaxes(title_text="Time", row=1, col=1)
#         fig.update_yaxes(title_text="Cumulative PnL", row=1, col=1)
#         fig.update_yaxes(title_text="Value", row=1, col=2)
#         fig.update_xaxes(
#             title_text="Average Trade Duration (hours)", row=2, col=1)
#         fig.update_yaxes(title_text="Average PnL per Trade", row=2, col=1)
#         fig.update_xaxes(title_text="PnL Distribution", row=2, col=2)
#         fig.update_xaxes(title_text="Hour of Day", row=3, col=2)
#         fig.update_yaxes(title_text="Day of Week", row=3, col=2)

#         if save_path:
#             fig.write_html(save_path)
#         else:
#             fig.show()
