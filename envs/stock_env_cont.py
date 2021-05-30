import os, copy, time, sys, random
from matplotlib.pyplot import axis
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd, numpy as np
from collections import deque

#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')
from datetime import datetime

# Custom Imports:
from render.graph import TradingGraph
# Normalization of observations:
from sklearn import preprocessing
from stock_env import StockEnvironment


class StockEnvironmentContinious(StockEnvironment):

    """A Bitcoin trading environment for OpenAI gym """
    metadata = {'render.modes': ['human', 'system', 'none']}
    scaler = preprocessing.MinMaxScaler()
    standardizer = preprocessing.StandardScaler()
    power_transformer = preprocessing.PowerTransformer(method='box-cox', standardize=False)
    viewer = None

    def __init__(self,     
                 df:pd.DataFrame, 
                 lookback_window_size:int = 50, 
                 initial_balance:float = 10000,
                 commission:float = 0.00075,
                 reward_strategy:str = 'base',
                 serial:bool=False,
                 render_range:int = 100,
                 show_reward:bool=False,
                 show_indicators:bool=False,
                 normalize_value = 40000,
                 slippage:float = 0.01,
                 normalize_obs:bool=True,
                 name:str ="Custom Trading Environment"):

        super(StockEnvironmentContinious, self).__init__(
                 df,
                 lookback_window_size,  
                 initial_balance, 
                 commission, 
                 reward_strategy,
                 serial,
                 render_range,
                 show_reward,
                 show_indicators,
                 normalize_value, 
                 slippage,
                 normalize_obs,
                 name
        )


    def discretize_action(self, action):

        amount = abs(action)
        
        if 0.1 > action > - 0.1:
            action = 0 # Hold
        elif action >= 0.1 :
            action = 1 # Buy
        else:
            action = 2 # Sell

        return action, amount



    # Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        #current_price = random.uniform(
        #    self.df.loc[self.current_step, 'Open'],
        #    self.df.loc[self.current_step, 'Close'])
        current_price = self._get_current_price()

        Date = self.df.loc[self.current_step, 'Date'] # for visualization
        High = self.df.loc[self.current_step, 'High'] # for visualization
        Low = self.df.loc[self.current_step, 'Low'] # for visualization

        # Slippage refers to the difference between the expected price of a trade
        # and the price at which the trade is executed. Slippage can occur at any time but 
        # is most prevalent during periods of higher volatility when market orders are used    
        self.adjust_slippage()


        action_disc, amount = self.discretize_action(action)

        print(action, amount, action_disc)
      
        if action_disc == 0: # Hold
            pass

        # Buy if current balance > initial balance:  
        elif action_disc == 1 and self.balance > self.initial_balance / 100:
            #1 + (self.commissionPercent / 100)) * (1 + (self.maxSlippagePercent / 100))
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price * amount
            adjust_price = (1 + self.commission) * (1 + self.slippage) 
            self.balance -= (self.crypto_bought * current_price ) * adjust_price
            self.crypto_held += self.crypto_bought

            self.trades.append({'Date'  : Date,
                                'High'  : High,
                                'Low'   : Low,
                                'total' : self.crypto_bought,
                                'type'  : "buy",
                                'current_price': current_price})

            self.episode_orders += 1

        elif action_disc == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held * amount
            adjust_price = 1 # (1 - self.commission) * (1 - self.slippage) 
            self.balance += (self.crypto_sold * current_price)  * adjust_price
            self.crypto_held -= self.crypto_sold

            self.trades.append({'Date'  : Date,
                                'High'  : High,
                                'Low'   : Low,
                                'total' : self.crypto_sold,
                                'type'  :  "sell",
                                'current_price': current_price})
            self.episode_orders += 1

        
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(self.get_order_history())

        if self.reward_strategy == 'incremental':
            # Receive calculated reward
            reward = self.get_reward()
            
        else:           
            reward = self.net_worth - self.prev_net_worth


        self.prev_net_worth = self.net_worth
        done = self.net_worth <= self.initial_balance / 2

        obs = self._next_observation()


        info = {
                'trade'  :    self.trades[-1] if len(self.trades) > 0 else [],
                'order'  :    {'balance'      : self.balance, 
                              'net_worth'     : self.net_worth,
                              'crypto_bought' : self.crypto_bought,
                              'crypto_sold'   : self.crypto_sold,
                              'crypto_held'   : self.crypto_held},
                'reward' :    reward,
                'shapes' :    self.shapes
                }  

        return obs, reward, done, info

