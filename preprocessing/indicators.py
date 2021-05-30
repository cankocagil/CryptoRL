
import pandas as pd
from ta.trend import SMAIndicator, macd, PSARIndicator
from ta.volatility import BollingerBands
from ta.momentum import rsi

import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from matplotlib import ticker
from datetime import datetime
import os
import numpy as np


def AddIndicators(df):
    # Add Simple Moving Average (SMA) indicators
    df["sma7"] = SMAIndicator(close=df["Close"], window=7, fillna=True).sma_indicator()
    df["sma25"] = SMAIndicator(close=df["Close"], window=25, fillna=True).sma_indicator()
    df["sma99"] = SMAIndicator(close=df["Close"], window=99, fillna=True).sma_indicator()
    
    # Add Bollinger Bands indicator
    indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Parabolic Stop and Reverse (Parabolic SAR) indicator
    indicator_psar = PSARIndicator(high=df["High"], low=df["Low"], close=df["Close"], step=0.02, max_step=2, fillna=True)
    df['psar'] = indicator_psar.psar()

    # Add Moving Average Convergence Divergence (MACD) indicator
    df["MACD"] = macd(close=df["Close"], window_slow=26, window_fast=12, fillna=True) # mazas

    # Add Relative Strength Index (RSI) indicator
    df["RSI"] = rsi(close=df["Close"], window=14, fillna=True) # mazas
    
    return df

def Plot_OHCL(df):
    df_original = df.copy()
    # necessary convert to datetime
    df["Date"] = pd.to_datetime(df.Date)
    df["Date"] = df["Date"].apply(mpl_dates.date2num)

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    # We are using the style ‘ggplot’
    plt.style.use('ggplot')
    
    # figsize attribute allows us to specify the width and height of a figure in unit inches
    fig = plt.figure(figsize=(16,8)) 

    # Create top subplot for price axis
    ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)

    # Create bottom subplot for volume which shares its x-axis
    ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

    candlestick_ohlc(ax1, df.values, width=0.8/24, colorup='green', colordown='red', alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    plt.xlabel('Date')
    plt.xticks(rotation=45)

    # Add Simple Moving Average
    ax1.plot(df["Date"], df_original['sma7'],'-', legend = 'SMA7')
    ax1.plot(df["Date"], df_original['sma25'],'-', legend = 'SMA25')
    ax1.plot(df["Date"], df_original['sma99'],'-', legend = 'SMA99')

    # Add Bollinger Bands
    ax1.plot(df["Date"], df_original['bb_bbm'],'-', legend = 'BB_BBM')
    ax1.plot(df["Date"], df_original['bb_bbh'],'-', legend = 'BB_BBH')
    ax1.plot(df["Date"], df_original['bb_bbl'],'-', legend = 'BB_BBL')

    # Add Parabolic Stop and Reverse
    ax1.plot(df["Date"], df_original['psar'],'.', legend = 'PSAR')



    # # Add Moving Average Convergence Divergence
    ax2.plot(df["Date"], df_original['MACD'],'-', legend = 'MovAveConvDiv')

    # # Add Relative Strength Index
    ax2.plot(df["Date"], df_original['RSI'],'-', legend = 'RSI')

    ax1.legend()
    ax2.legend()

    
    # beautify the x-labels (Our Date format)
    ax1.xaxis.set_major_formatter(mpl_dates.DateFormatter('%y-%m-%d'))# %H:%M:%S'))
    fig.autofmt_xdate()
    fig.tight_layout()

    #plt.draw()
    plt.show()


if __name__ == "__main__":   
    df = pd.read_csv('./data/pricedata.csv')
    df = df.sort_values('Date')
    df = AddIndicators(df)

    test_df = df[-400:]

    Plot_OHCL(df)
