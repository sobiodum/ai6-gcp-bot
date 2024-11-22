# File: trading/model_manager.py
# Path: forex_trading_system/trading/model_manager.py

# File: trading/model_manager.py

from typing import Dict, Optional, Tuple, List
import os, sys
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
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
# if project_root not in sys.path:
#     sys.path.append(project_root)

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
        verbose: int = 0
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
        description: str = "",
        eval_freq: int = 15_000,
        tags: List[str] = []
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
            # Create training config
            # config = TrainingConfig(
            #     total_timesteps=total_timesteps,
            #     batch_size=params.get('batch_size', 64),
            #     learning_rate=params.get('learning_rate', 3e-4),
            #     n_steps=params.get('n_steps', 2048),
            #     ent_coef=params.get('ent_coef', 0.01),
            #     n_epochs=params.get('n_epochs', 10),
            #     gamma=params.get('gamma', 0.99),
            #     policy_kwargs=params.get('policy_kwargs', {})
            # )

            # Train model
            model, metrics_dict = agent.train(
                df=df,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq
            )

            # version = ExperimentVersion(
            #     version_id=f"{pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            #     timestamp=datetime.now(),
            #     pair=pair,
            #     config=config,
            #     metrics=metrics,

            #     description=description,
            #     tags=tags,
            #     training_history=self.training_history,
            #     validation_history=self.validation_history
            # )
            # self.experiment_tracker.add_version(version)

            # Add training time to metrics
            # metrics.training_time = (
            #     datetime.now() - start_time).total_seconds()

            metrics = ModelMetrics(
                total_pnl=metrics_dict.get('total_pnl', 0.0),
                win_rate=metrics_dict.get('win_rate', 0.0),
                max_drawdown=metrics_dict.get('max_drawdown', 0.0),
                sharpe_ratio=metrics_dict.get('sharpe_ratio', 0.0),
                sortino_ratio=metrics_dict.get('sortino_ratio', 0.0),
                avg_trade_duration=metrics_dict.get('avg_trade_duration', 0.0),
                trades_per_day=metrics_dict.get('trades_per_day', 0.0),
                value_at_risk=metrics_dict.get('value_at_risk', 0.0),
                expected_shortfall=metrics_dict.get('expected_shortfall', 0.0),
                max_consecutive_losses=metrics_dict.get('max_consecutive_losses', 0),
                position_ratios=metrics_dict.get('position_ratios', {}),
                training_time=(datetime.now() - start_time).total_seconds(),
                total_steps=total_timesteps,
                final_loss=metrics_dict.get('final_loss', 0.0),
                evaluation_metrics=metrics_dict.get('evaluation_metrics', {})
            )

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
            raw_metrics = agent._calculate_metrics(model, env)
            
            # Convert raw metrics to ModelMetrics instance
            metrics = ModelMetrics(
                total_pnl=raw_metrics['total_pnl'],
                win_rate=raw_metrics['win_rate'],
                max_drawdown=raw_metrics.get('max_drawdown', 0.0),
                sharpe_ratio=raw_metrics['sharpe_ratio'],
                sortino_ratio=0.0,  # Calculate if available
                avg_trade_duration=raw_metrics.get('avg_trade_duration', 0.0),
                trades_per_day=raw_metrics.get('trades_per_day', 0.0),
                value_at_risk=0.0,  # Calculate if needed
                expected_shortfall=0.0,  # Calculate if needed
                max_consecutive_losses=0,  # Calculate if needed
                position_ratios={'long': 0.0, 'short': 0.0},  # Update if available
                training_time=0.0,  # Not relevant for evaluation
                total_steps=0,  # Not relevant for evaluation
                final_loss=0.0,  # Not relevant for evaluation
                evaluation_metrics=raw_metrics  # Store raw metrics
            )
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
        if not metrics_list:
            return ModelMetrics(
                total_pnl=0.0,
                win_rate=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                avg_trade_duration=0.0,
                trades_per_day=0.0,
                value_at_risk=0.0,
                expected_shortfall=0.0,
                max_consecutive_losses=0,
                position_ratios={'long': 0.0, 'short': 0.0},
                training_time=0.0,
                total_steps=0,
                final_loss=0.0,
                evaluation_metrics={}
            )

        # Initialize aggregations
        summed_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'avg_trade_duration': 0.0,
            'trades_per_day': 0.0,
            'value_at_risk': 0.0,
            'expected_shortfall': 0.0,
            'max_consecutive_losses': 0,
            'position_ratios': {'long': 0.0, 'short': 0.0},
            'training_time': 0.0,
            'total_steps': 0,
            'final_loss': 0.0
        }

        # Sum all metrics
        for metrics in metrics_list:
            for field, value in asdict(metrics).items():
                if field != 'evaluation_metrics' and field != 'position_ratios':
                    summed_metrics[field] += value
                elif field == 'position_ratios':
                    for pos_type, ratio in value.items():
                        summed_metrics['position_ratios'][pos_type] += ratio

        # Calculate averages
        n = len(metrics_list)
        averaged_metrics = {
            key: (value / n if not isinstance(value, dict) else 
                 {k: v / n for k, v in value.items()})
            for key, value in summed_metrics.items()
        }

        # Create and return averaged ModelMetrics instance
        return ModelMetrics(
            total_pnl=averaged_metrics['total_pnl'],
            win_rate=averaged_metrics['win_rate'],
            max_drawdown=averaged_metrics['max_drawdown'],
            sharpe_ratio=averaged_metrics['sharpe_ratio'],
            sortino_ratio=averaged_metrics['sortino_ratio'],
            avg_trade_duration=averaged_metrics['avg_trade_duration'],
            trades_per_day=averaged_metrics['trades_per_day'],
            value_at_risk=averaged_metrics['value_at_risk'],
            expected_shortfall=averaged_metrics['expected_shortfall'],
            max_consecutive_losses=round(averaged_metrics['max_consecutive_losses']),
            position_ratios=averaged_metrics['position_ratios'],
            training_time=averaged_metrics['training_time'],
            total_steps=round(averaged_metrics['total_steps']),
            final_loss=averaged_metrics['final_loss'],
            evaluation_metrics={}  # Not averaged
        )

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


