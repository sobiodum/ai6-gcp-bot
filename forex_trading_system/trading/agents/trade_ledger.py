import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import uuid

@dataclass
class Trade:
    """Represents a single completed trade."""
    trade_id: str
    pair: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_type: str  # 'long' or 'short'
    size: float
    pnl: float
    pnl_percentage: float
    holding_period: timedelta
    entry_reason: Optional[Dict] = None  # Store features/indicators that triggered entry
    exit_reason: Optional[Dict] = None   # Store features/indicators that triggered exit
    market_state: Optional[Dict] = None  # Store market conditions during trade

class TradeLedger:
    """Tracks and analyzes trading activity."""
    
    def __init__(self):
        self.trades: List[Trade] = []
        self._df: Optional[pd.DataFrame] = None
    
    def add_trade(self, trade: Trade):
        """Add a trade to the ledger."""
        self.trades.append(trade)
        self._df = None  # Reset cached DataFrame
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame for analysis."""
        if self._df is None:
            self._df = pd.DataFrame([vars(trade) for trade in self.trades])
            if not self._df.empty:
                self._df.set_index('trade_id', inplace=True)
        return self._df
    
    def calculate_metrics(self) -> Dict:
        """Calculate detailed trading metrics."""
        df = self.to_dataframe()
        if df.empty:
            return {}
            
        metrics = {
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl'] > 0]),
            'losing_trades': len(df[df['pnl'] < 0]),
            'win_rate': len(df[df['pnl'] > 0]) / len(df),
            'total_pnl': df['pnl'].sum(),
            'average_pnl': df['pnl'].mean(),
            'max_drawdown': self._calculate_drawdown(df),
            'avg_trade_duration': (df['exit_time'] - df['entry_time']).mean(),
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            'long_trades': len(df[df['position_type'] == 'long']),
            'short_trades': len(df[df['position_type'] == 'short']),
            'profit_factor': abs(df[df['pnl'] > 0]['pnl'].sum()) / abs(df[df['pnl'] < 0]['pnl'].sum())
        }
        
        return metrics
    
    def _calculate_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown from trade history."""
        cumulative_pnl = df['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdowns = (cumulative_pnl - running_max)
        return abs(drawdowns.min())
    
    def export_to_excel(self, filepath: str):
        """Export trade ledger to Excel with multiple sheets for analysis."""
        with pd.ExcelWriter(filepath) as writer:
            # Trade list
            self.to_dataframe().to_excel(writer, sheet_name='Trades')
            
            # Summary metrics
            metrics = self.calculate_metrics()
            pd.DataFrame([metrics]).to_excel(writer, sheet_name='Summary')
            
            # Monthly analysis
            df = self.to_dataframe()
            if not df.empty:
                monthly = df.set_index('exit_time').resample('M').agg({
                    'pnl': 'sum',
                    'trade_id': 'count'
                }).rename(columns={'trade_id': 'trades'})
                monthly.to_excel(writer, sheet_name='Monthly')
                
                # Win rate by hour
                hourly_stats = df.groupby(df['entry_time'].dt.hour).agg({
                    'pnl': ['count', lambda x: (x > 0).mean()]
                })
                hourly_stats.columns = ['trades', 'win_rate']
                hourly_stats.to_excel(writer, sheet_name='Hourly Analysis')