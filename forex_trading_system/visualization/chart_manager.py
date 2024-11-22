import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime
import matplotlib.pyplot as plt

import matplotlib.dates as mdates


class ChartManager:
    def chart(
        self,
        df: pd.DataFrame,
        ticker: str = 'No ticker provided',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        show_candlesticks: bool = False
            

  
    ) -> None:
        # Filter data if time range is specified
        if start_time:
            df = df[df.index >= pd.Timestamp(start_time)]
        if end_time:
            df = df[df.index <= pd.Timestamp(end_time)]

        # Create the figure and subplots
        fig = plt.figure(figsize=(12, 20))
        plt.style.use('seaborn-v0_8-whitegrid')

        gs = fig.add_gridspec(9, 1, height_ratios=[2, 2, 1,1, 1, 1, 1,1,1])
        ax1 = fig.add_subplot(gs[0])          # Candlestick chart
        ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Ichimoku Cloud
        ax3 = fig.add_subplot(gs[2], sharex=ax1)  # MACD
        ax4 = fig.add_subplot(gs[3], sharex=ax1)  # RSI
        ax5 = fig.add_subplot(gs[4], sharex=ax1)  # Bollinger Bandwidth
        ax7 = fig.add_subplot(gs[5], sharex=ax1)  # ATR
        ax8 = fig.add_subplot(gs[6], sharex=ax1)  # ADX/DMI
        ax9 = fig.add_subplot(gs[7], sharex=ax1)  # %B
        ax10 = fig.add_subplot(gs[8], sharex=ax1)  # Bollinger bandwidth




        ax1.plot(df.index, df['close'], linewidth=0.5,
                label=ticker, color='black')
        ax1.set_title("EUR/USD")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")

        # Second subplot (Ichimoku Cloud)

        ax2.set_title("Ichimoku Cloud")
        ax2.plot(df.index, df['tenkan_sen'], linewidth=0.5,
                label='Conversion Line (Tenkan-sen)', color='blue')
        ax2.plot(df.index, df['kijun_sen'], linewidth=0.5,
                label='Base Line (Kijun-sen)', color='red')
        ax2.plot(df.index, df['senkou_span_a'],
                linewidth=0.5, label='Senkou Span A', color='green')
        ax2.plot(df.index, df['senkou_span_b'],
                linewidth=0.5, label='Senkou Span B', color='orange')
        ax2.plot(df.index, df['close'], linewidth=1.5,
                label='Close Price', color='black')
        ax2.fill_between(
            df.index,
            df['senkou_span_a'],
            df['senkou_span_b'],
            where=(df['senkou_span_a'] >= df['senkou_span_b']),
            color='green',
            alpha=0.3
        )
        ax2.fill_between(
            df.index,
            df['senkou_span_a'],
            df['senkou_span_b'],
            where=(df['senkou_span_a'] < df['senkou_span_b']),
            color='red',
            alpha=0.3
        )

        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        # (MACD)

        ax3.set_title("MACD")
        ax3.bar(df.index, df['macd_hist'], width=1,
                label='MACD Histogram', alpha=0.3)
        ax3.plot(df.index, df['macd'], linewidth=0.5, label='MACD')
        ax3.plot(df.index, df['macd_signal'], linewidth=0.5, label='MACD Signal')
        ax3.set_xlabel("Date")
        ax3.set_ylabel("MACD")
        ax3.legend()


        # RSI

        ax4.set_title("RSI 14")
        ax4.plot(df.index, df['rsi'], linewidth=0.5, label='RSI', color='blue')
        ax4.axhline(y=70, color='red', linewidth=0.5, linestyle='--')
        ax4.axhline(y=30, color='green', linewidth=0.5, linestyle='--')

        # Fill area where RSI is above 70 (Overbought region)
        ax4.fill_between(df.index, 70, df['rsi'], where=(
            df['rsi'] >= 70), color='red', alpha=0.3)

        # Fill area where rsi is below 30 (Oversold region)
        ax4.fill_between(df.index, df['rsi'], 30, where=(
            df['rsi'] <= 30), color='green', alpha=0.3)

        ax4.set_xlabel("Date")
        ax4.set_ylim(0, 100)
        ax4.set_ylabel("RSI")
        ax4.legend()

        # Fifth subplot (Bollinger Bandwidth)

        ax5.set_title("Bollinger Bandwidth")
        ax5.set_title("Bollinger Bands and Spot Price")
        ax5.plot(df.index, df['bb_lower'], label='Lower Bollinger Band',
                color='purple', linestyle='--', linewidth=0.5)
        ax5.plot(df.index, df['bb_middle'], label='Middle Bollinger Band (SMA)',
                color='blue', linestyle='--', linewidth=0.5)
        ax5.plot(df.index, df['bb_upper'], label='Upper Bollinger Band',
                color='purple', linestyle='--', linewidth=0.5)

        # Plot Spot Price
        ax5.plot(df.index, df['close'], linewidth=1, label='Spot Price', color='black')

        ax5.set_xlabel("Date")
        ax5.set_ylabel("Price")
        ax5.legend()


  

        
        # ATR

        ax7.set_title("ATR")
        ax7.plot(df.index, df['atr'], linewidth=0.5, label='ATR', color='black')
        ax7.set_ylim(auto=True)
        ax7.set_xlabel("Date")
        ax7.set_ylabel("Range")
        ax7.legend(loc='upper right')



        # ADX / ADMI
        ax8.set_title("ADX/DMI")
        ax8.plot(df.index, df['adx'], linewidth=0.5, label='ADX', color='black')
        ax8.plot(df.index, df['plus_di'], linewidth=0.5, label='plus_di', color='green')
        ax8.plot(df.index, df['minus_di'], linewidth=0.5, label='-DI', color='red')

        ax8.set_xlabel("Date")
        ax8.set_ylim(auto=True)
        ax8.set_ylabel("Percentage")
        ax8.legend()
        

        # Plot Bollinger %B
        ax9.set_title("Bollinger %B")
        ax9.plot(df.index, df['bb_percent'], linewidth=0.5, label='Bollinger %B', color='blue')
        ax9.axhline(y=0, color='green', linestyle='--', linewidth=0.5, label='Lower Band (0)')
        ax9.axhline(y=50, color='orange', linestyle='--', linewidth=0.5, label='Middle Band (50)')
        ax9.axhline(y=100, color='red', linestyle='--', linewidth=0.5, label='Upper Band (100)')
        ax9.set_xlabel("Date")
        ax9.set_ylabel("Percentage (%)")
        ax9.legend(loc='upper right')


        # Plot Bollinger Bandwidth
        ax10.set_title("Bollinger Bandwidth")
        ax10.plot(df.index, df['bb_bandwidth'], linewidth=0.5, label='Bandwidth (%)', color='purple')
        ax10.set_xlabel("Date")
        ax10.set_ylabel("Bandwidth (%)")
        ax10.legend()


        # Apply the formatter to each x-axis
        date_formatter = mdates.DateFormatter('%Y-%m')
        ax1.xaxis.set_major_formatter(date_formatter)
        ax2.xaxis.set_major_formatter(date_formatter)
        ax3.xaxis.set_major_formatter(date_formatter)
        ax4.xaxis.set_major_formatter(date_formatter)
        ax7.xaxis.set_major_formatter(date_formatter)
        ax8.xaxis.set_major_formatter(date_formatter)
        ax9.xaxis.set_major_formatter(date_formatter)
        ax10.xaxis.set_major_formatter(date_formatter)

        # Automatically adjust spacing to avoid overlap
        plt.tight_layout()

        # Show the plot
        plt.show()
