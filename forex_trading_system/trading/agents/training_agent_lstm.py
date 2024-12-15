from data_management.dataset_manager import DatasetManager
import pandas as pd
import sys
import os
import uuid
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import optuna
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.evaluation import evaluate_policy


class FXTradingTrainer:
    def __init__(
        self,
        env_class,
        data_path: str,
        pair: str,
        sequence_length: int = 5,
        model_class: Type[BaseAlgorithm] = PPO,
        use_sequences: bool = False,
        tensorboard_log: str = "logs",
        seed: int = None,
        device: Union[th.device, str] = "auto",
    ):
        """
        Initialize the FX Trading trainer.
        """
        self.env_class = env_class
        self.pair = pair
        self.sequence_length = sequence_length

        # Load and split dataset
        self.dataset_manager = DatasetManager()
        df = pd.read_parquet(data_path)
        self.train_df, self.val_df, self.test_df = self.dataset_manager.split_dataset(
            df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        # Generate a unique experiment ID
        self.uuid_str = f"_{uuid.uuid4()}" if seed is None else f"_seed_{seed}"

        # Set up logging paths
        self.tensorboard_log = tensorboard_log
        self.save_path = os.path.join(
            tensorboard_log, f"fx_trading_{pair}{self.uuid_str}")

        # Store configuration
        self.model_class = model_class
        self.use_sequences = use_sequences
        self.device = device
        self.seed = seed

        # Create environments
        self.env = self._create_env(self.train_df, is_eval=False)

        # Initialize model as None (will be created during training)
        self.model = None

    def _create_env(self, df: pd.DataFrame, is_eval: bool = False) -> VecNormalize:
        """
        Create and wrap an environment with proper settings.
        """
        # Create base environment
        env = self.env_class(
            df=df,
            pair=self.pair,
            sequence_length=self.sequence_length,
        )

        # Wrap in Monitor for logging
        env = Monitor(env)

        # Wrap in VecEnv (required for SB3)
        env = DummyVecEnv([lambda: env])

        # Add VecCheckNan to catch numerical issues early
        env = VecCheckNan(env, raise_exception=True, warn_once=True)

        # Add normalization wrapper with different settings for eval
        env = VecNormalize(
            env,
            norm_obs=True,  # Always normalize observations
            norm_reward=not is_eval,  # Don't normalize rewards during evaluation
            clip_obs=10.0,
            clip_reward=10.0,
        )

        # For evaluation environments, disable training mode
        if is_eval:
            env.training = False

        return env

    def train(
        self,
        total_timesteps: int,
        eval_freq: int = 10000,
        save_freq: int = 10000,
        hyperparams: Optional[Dict[str, Any]] = None,
    ) -> BaseAlgorithm:
        """
        Train the model.
        """
        # Set up default hyperparameters if none provided
        if hyperparams is None:
            if self.use_sequences:
                hyperparams = self._get_default_recurrent_hyperparams()
            else:
                hyperparams = self._get_default_hyperparams()

        # Create the model
        self.model = self.model_class(
            "MlpPolicy" if not self.use_sequences else "MlpLstmPolicy",
            self.env,
            tensorboard_log=self.tensorboard_log,
            seed=self.seed,
            device=self.device,
            **hyperparams
        )

        # Set up callbacks
        callbacks = self._make_callbacks(eval_freq, save_freq)

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
        )

        return self.model

    def _make_callbacks(
        self,
        eval_freq: int,
        save_freq: int,
        n_eval_episodes: int = 5,
    ) -> list[BaseCallback]:
        """
        Create training callbacks for evaluation and checkpointing.
        """
        callbacks = []

        # Evaluation callback
        if eval_freq > 0:
            # Create evaluation environment using validation dataset
            eval_env = self._create_env(self.val_df, is_eval=True)

            # Create the callback
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=self.save_path,
                n_eval_episodes=n_eval_episodes,
                eval_freq=max(eval_freq // eval_env.num_envs, 1),
                log_path=self.save_path,
                deterministic=True,
            )
            callbacks.append(eval_callback)

        # Checkpoint callback
        if save_freq > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=max(save_freq // self.env.num_envs, 1),
                save_path=self.save_path,
                name_prefix=f"fx_model_{self.pair}",
            )
            callbacks.append(checkpoint_callback)

        return callbacks

    def optimize(
        self,
        total_timesteps: int,
        n_trials: int = 10,
        n_startup_trials: int = 5,
        n_evaluations: int = 2,
        studyname: str = "tpe",
        eval_freq: int = 5000,
        n_jobs: int = 1,  # Added parameter for parallel trials
        show_progress_bar: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.

        Args:
            total_timesteps: Timesteps for each trial
            n_trials: Number of trials to run
            n_startup_trials: Number of random trials before using TPE
            n_evaluations: Number of episodes for each evaluation
            eval_freq: How often to evaluate during training

        Returns:
            Best hyperparameters found
        """
        def objective(trial: optuna.Trial) -> float:
            """Optimization objective for Optuna."""

            # Sample hyperparameters using RL Zoo's ranges
            hyperparams = (
                self._sample_recurrent_ppo_params(trial)
                if self.use_sequences
                else self._sample_ppo_params(trial)
            )

            # Create evaluation environment using validation dataset
            eval_env = self._create_env(self.val_df, is_eval=True)

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=None,
                n_eval_episodes=n_evaluations,
                eval_freq=eval_freq,
                deterministic=True,
            )

            # Train with these hyperparameters
            model = self.model_class(
                "MlpPolicy" if not self.use_sequences else "MlpLstmPolicy",
                self.env,
                tensorboard_log=self.tensorboard_log,
                seed=self.seed,
                device=self.device,
                **hyperparams
            )

            try:
                model.learn(total_timesteps=total_timesteps,
                            callback=eval_callback)
                # Return negative reward for minimization
                return eval_callback.best_mean_reward
            except Exception as e:
                print(f"Trial failed: {e}")
                return float("-inf")

        # Create and run the study with TPE sampler and median pruner
        study = optuna.create_study(
            sampler=TPESampler(n_startup_trials=n_startup_trials),
            pruner=MedianPruner(n_startup_trials=n_startup_trials),
            direction="maximize",
            study_name=studyname,
            storage="sqlite:///optuna_lstm.db",
            load_if_exists=True,
        )

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=n_jobs,  # Added n_jobs parameter for parallelization
                show_progress_bar=show_progress_bar
            )
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print(f"Value: {-trial.value}")
        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        return study.best_params

    def _sample_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample PPO hyperparameters following SB3 RL Zoo's approach.
        Using their exact parameter ranges for optimality.
        """
        batch_size = trial.suggest_categorical(
            "batch_size", [8, 16, 32, 64, 128, 256, 512])
        n_steps = trial.suggest_categorical(
            "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        gamma = trial.suggest_categorical(
            "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
        clip_range = trial.suggest_categorical(
            "clip_range", [0.1, 0.2, 0.3, 0.4])
        n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
        gae_lambda = trial.suggest_categorical(
            "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        max_grad_norm = trial.suggest_categorical(
            "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        vf_coef = trial.suggest_float("vf_coef", 0, 1)
        net_arch = trial.suggest_categorical(
            "net_arch", ["tiny", "small", "medium"])

        # Convert architecture to actual network sizes
        net_arch = {
            "tiny": dict(pi=[64], vf=[64]),
            "small": dict(pi=[64, 64], vf=[64, 64]),
            "medium": dict(pi=[256, 256], vf=[256, 256])
        }[net_arch]

        return {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "n_epochs": n_epochs,
            "gae_lambda": gae_lambda,
            "max_grad_norm": max_grad_norm,
            "vf_coef": vf_coef,
            "policy_kwargs": dict(
                net_arch=net_arch,
            ),
        }

    def _sample_recurrent_ppo_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample RecurrentPPO hyperparameters following SB3 RL Zoo's approach.
        Extends PPO parameters with LSTM-specific options.
        """
        # Get base PPO parameters
        params = self._sample_ppo_params(trial)

        # Add LSTM-specific parameters
        enable_critic_lstm = trial.suggest_categorical(
            "enable_critic_lstm", [False, True])
        lstm_hidden_size = trial.suggest_categorical(
            "lstm_hidden_size", [16, 32, 64, 128, 256])

        # Update policy kwargs with LSTM parameters
        params["policy_kwargs"].update(
            dict(
                enable_critic_lstm=enable_critic_lstm,
                lstm_hidden_size=lstm_hidden_size,
            )
        )

        return params

    def evaluate_on_test(self, n_eval_episodes: int = 20):
        """
        Evaluate the current model on the test set.
        """
        if self.model is None:
            raise ValueError("No model to evaluate - train first!")

        # Create test environment
        test_env = self._create_env(self.test_df, is_eval=True)

        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(
            self.model,
            test_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True
        )

        print(f"Test set evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def _get_default_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for PPO."""
        return {
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "policy_kwargs": dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])]
            ),
        }

    def _get_default_recurrent_hyperparams(self) -> Dict[str, Any]:
        """Get default hyperparameters for RecurrentPPO."""
        return {
            "n_steps": 2048,
            "batch_size": 64,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "ent_coef": 0.0,
            "clip_range": 0.2,
            "n_epochs": 10,
            "gae_lambda": 0.95,
            "max_grad_norm": 0.5,
            "vf_coef": 0.5,
            "policy_kwargs": dict(
                lstm_hidden_size=64,
                enable_critic_lstm=True,
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],
            ),
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save the model and normalization stats."""
        if self.model is None:
            raise ValueError("No model to save - train first!")

        save_path = path or os.path.join(self.save_path, "final_model")

        # Save the model
        self.model.save(save_path)

        # Save normalization stats
        self.env.save(os.path.join(
            os.path.dirname(save_path), "vecnormalize.pkl"))

    def load(self, path: str) -> BaseAlgorithm:
        """
        Load a saved model and normalization stats.
        """
        # Load normalization stats if they exist
        stats_path = os.path.join(os.path.dirname(path), "vecnormalize.pkl")
        if os.path.exists(stats_path):
            self.env = VecNormalize.load(stats_path, self.env)
            # Ensure the stats don't update during testing
            self.env.training = False
            self.env.norm_reward = False

        # Load the model
        self.model = self.model_class.load(path, env=self.env)
        return self.model
