from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import numpy as np
import os, sys
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
import uuid
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from ..environments.forex_env import ForexTradingEnv, Actions
from .trade_ledger import TradeLedger, Trade



class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.results_path = base_path / "evaluation_results"
        self.results_path.mkdir(exist_ok=True)
        
    def evaluate_model(
        self, 
        model: PPO, 
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pair: str,
        version_id: str
    ) -> Dict:
        """Comprehensive model evaluation on all datasets."""
        
        results = {}
        ledgers = {}
        
        # Evaluate on each dataset
        for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            metrics, ledger = self._evaluate_single_dataset(model, df, pair)
            results[name] = metrics
            ledgers[name] = ledger
            
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.results_path / f"{pair}_{version_id}_{timestamp}"
        result_dir.mkdir(exist_ok=True)
        
        # Save trade ledgers
        for name, ledger in ledgers.items():
            ledger.export_to_excel(result_dir / f"{name}_trades.xlsx")
            
        # Generate and save visualization
        self._create_evaluation_plots(results, ledgers, result_dir)
        
        # Save summary metrics
        with open(result_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)
            
        return results, ledgers
    
    def _evaluate_single_dataset(
        self,
        model: PPO,
        df: pd.DataFrame,
        pair: str,
        save_dir: Path
    ) -> Tuple[Dict, TradeLedger]:
        """Evaluate model on a single dataset."""
        ledger = TradeLedger()
        
        # Create environment with the same settings as training
        def make_env():
            def _init():
                env = ForexTradingEnv(
                    df=df.copy(),
                    pair=pair,
                    random_start=False  # Deterministic for evaluation
                )
                return Monitor(env)
            return _init
        
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env()])
        
        # Wrap with VecNormalize and copy statistics from model's env
        if hasattr(model, 'env') and isinstance(model.env, VecNormalize):
            vec_env = VecNormalize(
                vec_env,
                training=False,  # Don't update statistics during evaluation
                norm_obs=model.env.norm_obs,
                norm_reward=False,  # Don't normalize rewards during evaluation
                clip_obs=model.env.clip_obs,
                clip_reward=model.env.clip_reward,
                gamma=model.env.gamma,
                epsilon=model.env.epsilon
            )
            # Copy running statistics
            vec_env.obs_rms = model.env.obs_rms
            vec_env.ret_rms = model.env.ret_rms
        
        obs = vec_env.reset()
        done = False
        trades_recorded = 0
        
        try:
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = vec_env.step(action)
                
                # Get info from first (and only) environment
                step_info = info[0] if isinstance(info, list) else info
                
                if step_info.get('trade_closed', False):
                    # print(f"Trade detected: {step_info}")  # Debug print
                    entry_time = pd.Timestamp(step_info['entry_time'])
                    exit_time = pd.Timestamp(step_info['exit_time'])
                    trade = Trade(
                        trade_id=str(uuid.uuid4()),
                        pair=pair,
                        entry_time=step_info['entry_time'],
                        exit_time=step_info['exit_time'],
                        entry_price=step_info['entry_price'],
                        exit_price=step_info['exit_price'],
                        position_type=step_info['position_type'],
                        size=step_info['position_size'],
                        pnl=step_info['trade_pnl'],
                        pnl_percentage=(step_info['trade_pnl'] / step_info['position_size']) * 100,
                        holding_period=exit_time - entry_time,  # Add holding_period to Trade
                        market_state=step_info.get('market_state', {})
                    )
                    ledger.add_trade(trade)
                    trades_recorded += 1
        
        finally:
            vec_env.close()
        print(f"Total trades recorded: {trades_recorded}")
        metrics = ledger.calculate_metrics()
        print(f"Ledger metrics: {metrics}")
        return metrics, ledger
    
    def _create_evaluation_plots(
        self,
        results: Dict,
        ledgers: Dict[str, TradeLedger],
        save_dir: Path
    ):
        """Create comprehensive evaluation plots."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # 1. PnL Curves
        ax1 = fig.add_subplot(gs[0, 0])
        for name, ledger in ledgers.items():
            df = ledger.to_dataframe()
            if not df.empty:
                cumulative_pnl = df['pnl'].cumsum()
                ax1.plot(cumulative_pnl.index, cumulative_pnl.values, label=name)
        ax1.set_title('Cumulative PnL')
        ax1.legend()
        
        # 2. Win Rates Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        win_rates = {name: results[name]['win_rate'] for name in results}
        ax2.bar(win_rates.keys(), win_rates.values())
        ax2.set_title('Win Rates by Dataset')
        
        # 3. Trade Duration Distribution
        ax3 = fig.add_subplot(gs[1, 0])
        for name, ledger in ledgers.items():
            df = ledger.to_dataframe()
            if not df.empty:
                durations = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
                sns.kdeplot(durations, label=name, ax=ax3)
        ax3.set_title('Trade Duration Distribution (hours)')
        ax3.legend()
        
        # 4. PnL Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        for name, ledger in ledgers.items():
            df = ledger.to_dataframe()
            if not df.empty:
                sns.kdeplot(df['pnl'], label=name, ax=ax4)
        ax4.set_title('PnL Distribution')
        ax4.legend()
        
        # 5. Trade Analysis by Hour
        ax5 = fig.add_subplot(gs[2, 0])
        test_ledger = ledgers['test']
        df = test_ledger.to_dataframe()
        if not df.empty:
            hourly_pnl = df.groupby(df['entry_time'].dt.hour)['pnl'].mean()
            ax5.bar(hourly_pnl.index, hourly_pnl.values)
            ax5.set_title('Average PnL by Hour (Test Set)')
        
        # 6. Position Type Analysis
        ax6 = fig.add_subplot(gs[2, 1])
        position_stats = df.groupby('position_type')['pnl'].agg(['count', 'mean'])
        position_stats.plot(kind='bar', ax=ax6)
        ax6.set_title('Position Type Analysis')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_plots.png')
        plt.close()