# File: trading/agents/training_agent.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
import gymnasium as gym
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

    def __init__(self,save_dir: Optional[Path] = None):
        self.save_dir = save_dir or Path("training_stats")
        self.save_dir.mkdir(exist_ok=True)

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
        # Balance progression
        self.balances: List[float] = []
        self.peak_balances: List[float] = []

    def add_episode_stats(
        self,
        reward: float,
        length: int,
        trades: int,
        win_rate: float,
        avg_duration: float,
        drawdown: float,
        positions: Dict[str, float],
        balance: float,
        peak_balance: float
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
    def save_to_disk(self, filename: Optional[str] = None):
        """Save training statistics to disk."""
        if filename is None:
            filename = f"training_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.save_dir / filename
        
        stats_dict = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'trade_counts': self.trade_counts,
            'win_rates': self.win_rates,
            'trade_durations': self.trade_durations,
            'drawdowns': self.drawdowns,
            'position_ratios': self.position_ratios,
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'balances': self.balances,
            'peak_balances': self.peak_balances,
            'timestamps': [ts.isoformat() for ts in self.timestamps]
        }
        
        with open(filepath, 'w') as f:
            json.dump(stats_dict, f, indent=2)

    def load_from_disk(self, filename: str):
        """Load training statistics from disk."""
        filepath = self.save_dir / filename
        with open(filepath, 'r') as f:
            stats_dict = json.load(f)
            
        self.episode_rewards = stats_dict['episode_rewards']
        self.episode_lengths = stats_dict['episode_lengths']
        self.trade_counts = stats_dict['trade_counts']
        self.win_rates = stats_dict['win_rates']
        self.trade_durations = stats_dict['trade_durations']
        self.drawdowns = stats_dict['drawdowns']
        self.position_ratios = stats_dict['position_ratios']
        self.losses = stats_dict['losses']
        self.learning_rates = stats_dict['learning_rates']
        self.balances = stats_dict['balances']
        self.peak_balances = stats_dict['peak_balances']
        self.timestamps = [datetime.fromisoformat(ts) for ts in stats_dict['timestamps']]
    
    def plot_metrics(self, save_path: Optional[str] = None):
        """Plot comprehensive training metrics."""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 3)
        
        # Performance metrics
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.episode_rewards)
        ax1.set_title('Episode Rewards')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.balances)
        ax2.plot(self.peak_balances, '--')
        ax2.set_title('Account Balance')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(self.drawdowns)
        ax3.set_title('Drawdown')
        
        # Trading metrics
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(self.trade_counts)
        ax4.set_title('Trades per Episode')
        
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(self.win_rates)
        ax5.set_title('Win Rate')
        
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(self.trade_durations)
        ax6.set_title('Avg Trade Duration')
        
        # Training metrics
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.plot(self.losses['policy_loss'])
        ax7.set_title('Policy Loss')
        
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(self.losses['value_loss'])
        ax8.set_title('Value Loss')
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.plot(self.learning_rates)
        ax9.set_title('Learning Rate')
        
        # Position analysis
        ax10 = fig.add_subplot(gs[3, :])
        position_df = pd.DataFrame(self.position_ratios)
        position_df.plot(kind='area', stacked=True, ax=ax10)
        ax10.set_title('Position Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
class TrainingMonitor(BaseCallback):
    """Monitors and records training progress."""
    
    def __init__(
        self,
        eval_freq: int,
        stats: TrainingStats,
        save_freq: int = 10_000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.stats = stats
        self.save_freq = save_freq
        
    
    def _on_step(self) -> bool:
        try:
            if len(self.model.ep_info_buffer) > 0 and hasattr(self.training_env, 'envs'):
                info = self.model.ep_info_buffer[-1]
                env = self.training_env.envs[0].unwrapped
                
                # Add episode statistics
                self.stats.add_episode_stats(
                    reward=info['r'],
                    length=info['l'],
                    trades=env.total_trades,
                    win_rate=env.win_rate,
                    avg_duration=env.avg_trade_duration,
                    drawdown=env.max_drawdown,
                    positions=env.position_ratios,
                    balance=env.balance,
                    peak_balance=env.peak_balance
                )
                
                # Add training statistics
                if self.model.logger is not None:
                    self.stats.add_training_stats(
                        policy_loss=self.model.logger.name_to_value.get('policy_loss', 0),
                        value_loss=self.model.logger.name_to_value.get('value_loss', 0),
                        entropy_loss=self.model.logger.name_to_value.get('entropy_loss', 0),
                        learning_rate=self.model.learning_rate
                    )
                
                # Periodic saving
                if self.n_calls % self.save_freq == 0:
                    self.stats.save_to_disk()
                    # Also save plots
                    self.stats.plot_metrics(
                        save_path=str(self.stats.save_dir / f"training_plots_{self.n_calls}.png")
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
        verbose: int = 0
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

    def _calculate_metrics(self, model: PPO, env: gym.Env) -> Dict:
        """Calculate performance metrics for a model on an environment."""
        done = False
        truncated = False
        observation = env.reset()  # VecEnv just returns the observation
        total_reward = 0
        trades = 0
        wins = 0

        while not done:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)  # VecEnv returns 4 values
            total_reward += reward[0]  # Get scalar reward from array
            
            # Get info from first environment
            step_info = info[0] if isinstance(info, list) else info
            
            # Track trades
            if step_info.get('trade_closed', False):
                trades += 1
                if step_info.get('trade_pnl', 0) > 0:
                    wins += 1

        # Get final info from first environment
        final_info = info[0] if isinstance(info, list) else info
        
        return {
            'total_pnl': final_info.get('total_pnl', 0.0),
            'win_rate': final_info.get('win_rate', 0.0),
            'sharpe_ratio': total_reward / (np.std([reward[0]]) + 1e-6),
            'max_drawdown': final_info.get('drawdown', 0.0),
            'total_trades': final_info.get('total_trades', 0),
            'final_balance': final_info.get('balance', 0.0)
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
