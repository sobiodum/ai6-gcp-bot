# File: trading/agents/training_agent.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import torch
from pathlib import Path
import logging
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.monitor import Monitor

from ..environments.forex_env import ForexTradingEnv, Actions


class TrainingStats:
    """Tracks detailed training statistics."""

    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.trade_counts: List[int] = []
        self.win_rates: List[float] = []
        self.trade_durations: List[float] = []
        self.drawdowns: List[float] = []
        self.position_ratios: List[Dict[str, float]] = []
        self.losses: Dict[str, List[float]] = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': []
        }
        self.learning_rates: List[float] = []
        self.timestamps: List[datetime] = []

    def add_episode_stats(
        self,
        reward: float,
        length: int,
        trades: int,
        win_rate: float,
        avg_duration: float,
        drawdown: float,
        positions: Dict[str, float]
    ):
        """Add statistics for a completed episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.trade_counts.append(trades)
        self.win_rates.append(win_rate)
        self.trade_durations.append(avg_duration)
        self.drawdowns.append(drawdown)
        self.position_ratios.append(positions)
        self.timestamps.append(datetime.now())

    def add_training_stats(
        self,
        policy_loss: float,
        value_loss: float,
        entropy_loss: float,
        learning_rate: float
    ):
        """Add training-related statistics."""
        self.losses['policy_loss'].append(policy_loss)
        self.losses['value_loss'].append(value_loss)
        self.losses['entropy_loss'].append(entropy_loss)
        self.learning_rates.append(learning_rate)

    def get_recent_stats(self, window: int = 100) -> Dict:
        """Get average statistics over recent episodes."""
        if not self.episode_rewards:
            return {}

        window = min(window, len(self.episode_rewards))
        return {
            'avg_reward': np.mean(self.episode_rewards[-window:]),
            'avg_length': np.mean(self.episode_lengths[-window:]),
            'avg_trades': np.mean(self.trade_counts[-window:]),
            'avg_win_rate': np.mean(self.win_rates[-window:]),
            'avg_duration': np.mean(self.trade_durations[-window:]),
            'avg_drawdown': np.mean(self.drawdowns[-window:]),
            'policy_loss': np.mean(self.losses['policy_loss'][-window:]),
            'value_loss': np.mean(self.losses['value_loss'][-window:]),
            'entropy_loss': np.mean(self.losses['entropy_loss'][-window:]),
            'learning_rate': self.learning_rates[-1] if self.learning_rates else None
        }


class TrainingMonitor(BaseCallback):
    """Monitors and records training progress."""

    def __init__(
        self,
        eval_freq: int,
        stats: TrainingStats,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.stats = stats
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        """Called after each step during training."""
        try:
            # Get training info from model
            if len(self.model.ep_info_buffer) > 0 and hasattr(self.training_env, 'envs'):
                # Get latest episode info
                info = self.model.ep_info_buffer[-1]
                env = self.training_env.envs[0]

                self.stats.add_episode_stats(
                    reward=info['r'],
                    length=info['l'],
                    trades=env.total_trades,
                    win_rate=env.win_rate,
                    avg_duration=env.avg_trade_duration,
                    drawdown=env.max_drawdown,
                    positions=env.position_ratios
                )

            # Get training losses
            if self.model.logger is not None:
                self.stats.add_training_stats(
                    policy_loss=self.model.logger.name_to_value.get(
                        'policy_loss', 0),
                    value_loss=self.model.logger.name_to_value.get(
                        'value_loss', 0),
                    entropy_loss=self.model.logger.name_to_value.get(
                        'entropy_loss', 0),
                    learning_rate=self.model.learning_rate
                )

            return True

        except Exception as e:
            self.logger.error(f"Error in training monitor: {str(e)}")
            return True


class TrainingAgent:
    """Handles model training and direct environment interaction."""

    def __init__(
        self,
        pair: str,
        save_path: Path,
        n_envs: int = 4,
        verbose: int = 1
    ):
        """
        Initialize training agent.

        Args:
            pair: Currency pair being trained
            save_path: Path for temporary training artifacts
            n_envs: Number of parallel environments
            verbose: Verbosity level
        """
        self.logger = logging.getLogger(__name__)
        self.pair = pair
        self.save_path = Path(save_path)
        self.n_envs = n_envs
        self.verbose = verbose
        self.training_stats = TrainingStats()

        # Default hyperparameters (can be overridden)
        self.hyperparameters = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'use_sde': False,
            'sde_sample_freq': -1,
            'policy_kwargs': dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            )
        }

    def create_env(
        self,
        df: pd.DataFrame,
        is_training: bool = True
    ) -> VecNormalize:
        """Create vectorized and normalized environments."""
        def make_env():
            def _init():
                env = ForexTradingEnv(
                    df=df.copy(),
                    pair=self.pair,
                    random_start=is_training
                )
                env = Monitor(
                    env,
                    filename=None,
                    info_keywords=(
                        'total_pnl',
                        'trade_count',
                        'win_rate',
                        'drawdown',
                        'position_duration'
                    )
                )
                return env
            return _init

        # Create vectorized environment
        if self.n_envs > 1 and is_training:
            vec_env = SubprocVecEnv([make_env() for _ in range(self.n_envs)])
        else:
            vec_env = DummyVecEnv([make_env()])

        # Add normalization
        env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.,
            clip_reward=10.,
            gamma=self.hyperparameters['gamma'],
            epsilon=1e-08
        )

        return env

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int,
        eval_freq: int,
        hyperparameters: Optional[Dict] = None
    ) -> Tuple[PPO, Dict]:
        """
        Train the model.

        Args:
            df: Training data
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            hyperparameters: Optional hyperparameters override

        Returns:
            Tuple of (trained model, training statistics)
        """
        if hyperparameters:
            self.hyperparameters.update(hyperparameters)

        self.logger.info(f"Starting training for {self.pair}")
        self.logger.info(f"Hyperparameters: {self.hyperparameters}")

        try:
            # Create environments
            train_env = self.create_env(df, is_training=True)
            eval_env = self.create_env(df, is_training=False)

            # Initialize model
            model = PPO(
                "MultiInputPolicy",
                train_env,
                verbose=self.verbose,
                tensorboard_log=str(self.save_path / "logs"),
                **self.hyperparameters
            )

            # Setup callbacks
            callbacks = [
                TrainingMonitor(
                    eval_freq=eval_freq,
                    stats=self.training_stats,
                    verbose=self.verbose
                ),
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.save_path / "eval"),
                    n_eval_episodes=10,
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False
                )
            ]

            # Train model
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )

            # Calculate final training statistics
            final_stats = self._calculate_final_stats(model, eval_env)

            return model, final_stats

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise
        finally:
            if 'train_env' in locals():
                train_env.close()
            if 'eval_env' in locals():
                eval_env.close()

    def _calculate_final_stats(
        self,
        model: PPO,
        eval_env: VecNormalize,
        n_eval_episodes: int = 10
    ) -> Dict:
        """Calculate comprehensive final training statistics."""
        # Get recent training stats
        training_stats = self.training_stats.get_recent_stats()

        # Run final evaluation episodes
        eval_rewards = []
        eval_metrics = {
            'trades': [],
            'win_rates': [],
            'drawdowns': [],
            'durations': []
        }

        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward

                # Collect metrics
                if done:
                    info = info[0]  # Get info from first environment
                    eval_rewards.append(episode_reward)
                    eval_metrics['trades'].append(info.get('trade_count', 0))
                    eval_metrics['win_rates'].append(info.get('win_rate', 0))
                    eval_metrics['drawdowns'].append(info.get('drawdown', 0))
                    eval_metrics['durations'].append(
                        info.get('position_duration', 0))

        # Combine training and evaluation statistics
        final_stats = {
            'training': training_stats,
            'evaluation': {
                'mean_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'mean_trades': np.mean(eval_metrics['trades']),
                'mean_win_rate': np.mean(eval_metrics['win_rates']),
                'mean_drawdown': np.mean(eval_metrics['drawdowns']),
                'mean_duration': np.mean(eval_metrics['durations'])
            }
        }

        return final_stats
