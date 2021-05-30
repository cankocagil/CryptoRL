
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from matplotlib import ticker
from datetime import datetime
import os
#import cv2
import numpy as np



class TradingGraph:
    """A Bitcoin OHLCV trading visualization using matplotlib made to render environments"""


    styles =  ['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic',
              'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',
              'seaborn', 'seaborn-bright', 'seaborn-colorblind', 
              'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 
              'seaborn-deep', 'seaborn-muted','seaborn-notebook', 'seaborn-paper',
              'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
              'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']

    colors = ['orange', 'cyan', 'purple', 
             'blue', 'magenta', 'yellow', 
             'black', 'red', 'green']

    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, volume, net_worth, trades
    # call render every step
    def __init__(self,
                 render_range:int = 360,
                 show_reward:bool =  False,
                 show_indicators:bool = False,
                 display_hour_and_minute:bool = False,
                 title:str = "BTC Historical Data Live Session",
                 style:str = 'ggplot'):

        assert style in self.styles, 'Style not available'

        self.volume = deque(maxlen=render_range)
        self.net_worth = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range)
        self.render_range = render_range
        self.show_reward = show_reward
        self.show_indicators = show_indicators
        self.display_hour_and_minute = display_hour_and_minute
        self.per_minute = 24 * 60

        # We are using the style ‘ggplot’, or can be preferred from above
        plt.style.use(style)
    

        # close all plots if there are open
        plt.close('all')

        self.title = title
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8)) 
        self.fig.suptitle(title)

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        
        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
        
        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y %H:%M')
        

        if self.display_hour_and_minute:
            self.date_format = mpl_dates.DateFormatter('%d-%m-%Y %H:%M')

        # Add paddings to make graph easier to view
        #plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        #self.fig.tight_layout()

        # define if show indicators
        if self.show_indicators:
            self.create_indicators_lists()

        #plt.show()

    def create_indicators_lists(self):
        # Create a new axis for indicatorswhich shares its x-axis with volume
        self.ax4 = self.ax2.twinx()
        
        self.sma7 = deque(maxlen=self.render_range)
        self.sma25 = deque(maxlen=self.render_range)
        self.sma99 = deque(maxlen=self.render_range)       
        self.bb_bbm = deque(maxlen=self.render_range)
        self.bb_bbh = deque(maxlen=self.render_range)
        self.bb_bbl = deque(maxlen=self.render_range)       
        self.psar = deque(maxlen=self.render_range)
        self.MACD = deque(maxlen=self.render_range)
        self.RSI = deque(maxlen=self.render_range)


    def plot_indicators(self, df, data_render_range):
        self.sma7.append(df["sma7"])
        self.sma25.append(df["sma25"])
        self.sma99.append(df["sma99"])

        self.bb_bbm.append(df["bb_bbm"])
        self.bb_bbh.append(df["bb_bbh"])
        self.bb_bbl.append(df["bb_bbl"])

        self.psar.append(df["psar"])

        self.MACD.append(df["MACD"])
        self.RSI.append(df["RSI"])

        # Add Simple Moving Average
        self.ax1.plot(data_render_range, self.sma7,'-', label = 'SMA7')
        self.ax1.plot(data_render_range, self.sma25,'-', label= 'SMA25')
        self.ax1.plot(data_render_range, self.sma99,'-', label = 'SMA99')

        # Add Bollinger Bands
        self.ax1.plot(data_render_range, self.bb_bbm,'-', label = 'BB_BBM')
        self.ax1.plot(data_render_range, self.bb_bbh,'-', label = 'BB_BBH')
        self.ax1.plot(data_render_range, self.bb_bbl,'-', label = 'BB_BBL')

        # Add Parabolic Stop and Reverse
        self.ax1.plot(data_render_range, self.psar,'.', label = 'PSAR')

        self.ax4.clear()
        # # Add Moving Average Convergence Divergence
        self.ax4.plot(data_render_range, self.MACD,'r-', label = 'MovAveConvDiv')

        # # Add Relative Strength Index
        self.ax4.plot(data_render_range, self.RSI,'g-', label = 'RSI')

        self.ax1.legend(loc = 'upper left')
        self.ax4.legend(loc = 'upper left')

    # Render the environment to the screen
    #def render(self, Date, Open, High, Low, Close, volume, net_worth, trades):
    def render(self,
              df:pd.DataFrame,
              net_worth:float,
              trades:dict,
              initial_net_worth:float = 10000):
        
        Date = df["Date"]
        Open = df["Open"]
        High = df["High"]
        Low = df["Low"]
        Close = df["Close"]
        Volume = df["Volume"]

        # append volume and net_worth to deque list
        self.volume.append(Volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        Date = mpl_dates.date2num([pd.to_datetime(Date)])[0]
        self.render_data.append([Date, Open, High, Low, Close])


                
        profit_percent = round((net_worth - initial_net_worth) / initial_net_worth * 100, 2)
        suptitle = self.title + '\n' + 'Net worth: $' + str(round(net_worth,4)) + ' | Profit: ' + str(profit_percent) + '%'
        self.fig.suptitle(suptitle)


        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/(24*60), colorup='green', colordown='red', alpha=0.8)


        try:
            self.ax1.annotate('{0:.2f}'.format(Open), (Date, Open),
                        xytext=(Date, High),
                        bbox=dict(boxstyle='round',
                                    fc='w', ec='k', lw=1),
                        color="r",
                        fontsize="small")        
        except:
            pass


        # Put all dates to one list and fill ax2 sublot with volume
        data_render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(data_render_range, self.volume, 0,)

        if self.show_indicators:
            self.plot_indicators(df, data_render_range)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(data_render_range, self.net_worth, color="blue", label = 'Net Worth')
        
        # Show legend, which uses the label we defined for the plot above
        self.ax3.legend()
        

        try:
            last_net_worth = self.net_worth[-1]
            last_date = Date
            # Annotate the current net worth on the net worth graph
            self.ax3.annotate('{0:.2f}'.format(self.net_worth), (last_date, last_net_worth),
                                    xytext=(last_date, last_net_worth),
                                    bbox=dict(boxstyle='round',
                                                fc='w', ec='k', lw=1),
                                    color="green",
                                    fontsize="small")
        except:
            pass

        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:,1:])
        maximum = np.max(np.array(self.render_data)[:,1:])
        RANGE = maximum - minimum

        
        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            
            if trade_date in data_render_range:
                if trade['type'] == 'buy':
                    high_low = trade['Low'] - RANGE*0.02
                    ycoords = trade['Low'] - RANGE*0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High'] + RANGE*0.02
                    ycoords = trade['High'] + RANGE*0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="v")

                if self.show_reward:
                    try:
                        self.ax1.annotate('{0:.2f}'.format(trade['Reward']), 
                        (trade_date-0.02, high_low), xytext=(trade_date-0.02, ycoords),
                        bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), 
                        color = 'purple',
                        fontsize="small")
                    except:
                        pass
        
        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Net worth')

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        plt.show(block=False)

        # Test!:
        #plt.draw()

        # Necessary to view frames before they are unrendered
        plt.pause(0.001)

    def close(self):
        plt.close()








































        """Display image with OpenCV - no interruption

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        
        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot",image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        else:
            return img

        """


if __name__ == "__main__":
    #print(plt.style.available)
    ...


