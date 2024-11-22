from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import hashlib
from pathlib import Path
import pandas as pd

@dataclass
class TrainingConfig:
    """Tracks training configuration and hyperparameters."""
    total_timesteps: int
    batch_size: int
    learning_rate: float
    n_steps: int
    ent_coef: float
    n_epochs: int
    gamma: float
    policy_kwargs: Dict[str, Any]
    normalize_advantage: bool = True
    clip_range: float = 0.2
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5

@dataclass
class TrainingMetrics:
    """Tracks detailed training metrics."""
    total_pnl: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    avg_trade_duration: float
    trades_per_day: float
    value_at_risk: float
    expected_shortfall: float
    max_consecutive_losses: int
    position_ratios: Dict[str, float]
    training_time: float
    total_steps: int
    final_loss: float
    
    # Validation metrics
    val_total_pnl: float
    val_win_rate: float
    val_sharpe_ratio: float
    val_max_drawdown: float
    
    # Early stopping metrics
    best_val_metric: float
    steps_without_improvement: int

@dataclass
class ExperimentVersion:
    """Enhanced experiment version tracking."""
    version_id: str
    timestamp: datetime
    pair: str
    config: TrainingConfig
    metrics: TrainingMetrics

    description: str
    tags: List[str]
    is_deployed: bool = False
    parent_version: Optional[str] = None
    
    # Training history
    training_history: List[Dict[str, float]]
    validation_history: List[Dict[str, float]]
    
    def save(self, base_path: Path) -> None:
        """Save experiment version to disk."""
        version_dir = base_path / self.pair / self.version_id
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save version info
        with open(version_dir / "version.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_parquet(version_dir / "training_history.parquet")
        
        # Save validation history
        val_df = pd.DataFrame(self.validation_history)
        val_df.to_parquet(version_dir / "validation_history.parquet")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            'version_id': self.version_id,
            'timestamp': self.timestamp.isoformat(),
            'pair': self.pair,
            'config': asdict(self.config),
            'metrics': asdict(self.metrics),
    
 
            'description': self.description,
            'tags': self.tags,
            'is_deployed': self.is_deployed,
            'parent_version': self.parent_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentVersion':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['config'] = TrainingConfig(**data['config'])
        data['metrics'] = TrainingMetrics(**data['metrics'])
        return cls(**data)
    
    def compare_with(self, other: 'ExperimentVersion') -> Dict[str, float]:
        """Compare metrics with another version."""
        improvements = {}
        for field in TrainingMetrics.__dataclass_fields__:
            current = getattr(self.metrics, field)
            previous = getattr(other.metrics, field)
            if isinstance(current, (int, float)):
                pct_change = ((current - previous) / abs(previous)) * 100 if previous != 0 else float('inf')
                improvements[field] = pct_change
        return improvements

class ExperimentTracker:
    """Manages experiment versions and comparisons."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.base_path.mkdir(exist_ok=True)
        self.versions: Dict[str, Dict[str, ExperimentVersion]] = {}
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load all saved versions."""
        for pair_dir in self.base_path.iterdir():
            if pair_dir.is_dir():
                pair = pair_dir.name
                self.versions[pair] = {}
                
                for version_dir in pair_dir.iterdir():
                    if version_dir.is_dir():
                        version_file = version_dir / "version.json"
                        if version_file.exists():
                            with open(version_file, "r") as f:
                                data = json.load(f)
                                version = ExperimentVersion.from_dict(data)
                                self.versions[pair][version.version_id] = version
    
    def add_version(self, version: ExperimentVersion) -> None:
        """Add new experiment version."""
        if version.pair not in self.versions:
            self.versions[version.pair] = {}
        
        self.versions[version.pair][version.version_id] = version
        version.save(self.base_path)
    
    def get_best_version(self, pair: str, metric: str = "val_sharpe_ratio") -> Optional[ExperimentVersion]:
        """Get best version for a pair based on specified metric."""
        if pair not in self.versions or not self.versions[pair]:
            return None
            
        return max(
            self.versions[pair].values(),
            key=lambda v: getattr(v.metrics, metric)
        )
    
    def get_version_summary(self, pair: str) -> pd.DataFrame:
        """Get summary of all versions for a pair."""
        if pair not in self.versions:
            return pd.DataFrame()
            
        summaries = []
        for version in self.versions[pair].values():
            summary = {
                'version_id': version.version_id,
                'timestamp': version.timestamp,
                'val_sharpe_ratio': version.metrics.val_sharpe_ratio,
                'val_win_rate': version.metrics.val_win_rate,
                'val_max_drawdown': version.metrics.val_max_drawdown,
                'description': version.description,
                'tags': ','.join(version.tags)
            }
            summaries.append(summary)
            
        return pd.DataFrame(summaries).sort_values('timestamp', ascending=False)