# File: visualization/chart_manager.py
# Path: forex_trading_system/visualization/chart_manager.py

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime


class ChartManager:
    """Manages creation and display of trading charts using plotly."""

    def create_charts(
        self,
        df: pd.DataFrame,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        show_candlesticks: bool = False
    ) -> None:
        """
        Create comprehensive trading analysis charts.

        Args:
            df: DataFrame containing price data and indicators
            start_time: Optional start time filter
            end_time: Optional end time filter
            show_candlesticks: Whether to show candlesticks (True) or line chart (False)
        """
        # Filter data if time range is specified
        if start_time:
            df = df[df.index >= pd.Timestamp(start_time)]
        if end_time:
            df = df[df.index <= pd.Timestamp(end_time)]

        # Create subplots for different indicators
        fig = make_subplots(
            rows=7, cols=1,
            subplot_titles=(
                'Price with SMAs',
                'MACD',
                'RSI',
                'Bollinger Bands',
                'ADX/DMI',
                'Ichimoku Cloud',
                'ATR'
            ),
            vertical_spacing=0.05,
            row_heights=[1, 0.7, 0.7, 1, 0.7, 1, 0.7]
        )

        # 1. Price and SMAs
        if show_candlesticks:
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name='Price'
                ),
                row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    name='Close Price',
                    line=dict(color='black', width=1)
                ),
                row=1, col=1
            )

        # Add SMAs
        for period in [20, 50]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'sma_{period}'],
                    name=f'SMA {period}',
                    line=dict(width=1)
                ),
                row=1, col=1
            )

        # 2. MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd'],
                name='MACD',
                line=dict(color='blue', width=1)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['macd_signal'],
                name='Signal',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['macd_hist'],
                name='MACD Histogram'
            ),
            row=2, col=1
        )

        # 3. RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )
        # Add RSI threshold lines
        fig.add_hline(y=70, line_color="red", line_dash="dash", row=3, col=1)
        fig.add_hline(y=30, line_color="green", line_dash="dash", row=3, col=1)

        # 4. Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close',
                line=dict(color='black', width=1)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_upper'],
                name='Upper BB',
                line=dict(color='gray', dash='dash', width=1)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_middle'],
                name='Middle BB',
                line=dict(color='blue', width=1)
            ),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['bb_lower'],
                name='Lower BB',
                line=dict(color='gray', dash='dash', width=1)
            ),
            row=4, col=1
        )

        # 5. ADX and DMI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx'],
                name='ADX',
                line=dict(color='black', width=1)
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['plus_di'],
                name='+DI',
                line=dict(color='green', width=1)
            ),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['minus_di'],
                name='-DI',
                line=dict(color='red', width=1)
            ),
            row=5, col=1
        )

        # 6. Ichimoku Cloud
        ichimoku_traces = [
            ('tenkan_sen', 'Conversion Line', 'blue'),
            ('kijun_sen', 'Base Line', 'red'),
            ('senkou_span_a', 'Leading Span A', 'green'),
            ('senkou_span_b', 'Leading Span B', 'red'),

        ]

        for col, name, color in ichimoku_traces:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    name=name,
                    line=dict(color=color, width=1)
                ),
                row=6, col=1
            )

        # Add price to Ichimoku chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                name='Close Price',
                line=dict(color='black', width=1)
            ),
            row=6, col=1
        )

        # 7. ATR
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['atr'],
                name='ATR',
                line=dict(color='blue', width=1)
            ),
            row=7, col=1
        )

        # Update layout
        fig.update_layout(
            height=2000,  # Increased height for better visibility
            title_text="Trading Analysis Dashboard",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False
        )

        # Update y-axes labels
        y_axis_labels = [
            'Price', 'MACD', 'RSI', 'Price',
            'ADX/DMI', 'Price', 'ATR'
        ]

        for i, label in enumerate(y_axis_labels, start=1):
            fig.update_yaxes(title_text=label, row=i, col=1)

        # Update x-axis to show date format
        for i in range(1, 8):
            fig.update_xaxes(
                row=i,
                col=1,
                rangeslider_visible=False,
                showgrid=True
            )

        # Show the plot
        fig.show()

    def create_single_chart(
        self,
        df: pd.DataFrame,
        chart_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        show_candlesticks: bool = False
    ) -> None:
        """
        Create a single chart for a specific indicator or price.

        Args:
            df: DataFrame containing price data and indicators
            chart_type: Type of chart to create ('price', 'macd', 'rsi', etc.)
            start_time: Optional start time filter
            end_time: Optional end time filter
            show_candlesticks: Whether to show candlesticks (True) or line chart (False)
        """
        # Filter data if time range is specified
        if start_time:
            df = df[df.index >= pd.Timestamp(start_time)]
        if end_time:
            df = df[df.index <= pd.Timestamp(end_time)]

        fig = go.Figure()

        if chart_type == 'price':
            if show_candlesticks:
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price'
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['close'],
                        name='Close Price',
                        line=dict(color='black')
                    )
                )

            # Add SMAs
            for period in [20, 50]:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[f'sma_{period}'],
                        name=f'SMA {period}'
                    )
                )

        elif chart_type == 'macd':
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd'],
                    name='MACD'
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name='Signal'
                )
            )
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['macd_hist'],
                    name='Histogram'
                )
            )

        # Add more chart types as needed...

        fig.update_layout(
            title=f"{chart_type.upper()} Chart",
            xaxis_title="Date",
            yaxis_title=chart_type.upper(),
            height=600,
            showlegend=True
        )

        fig.show()
