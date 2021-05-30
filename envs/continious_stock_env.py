import os, copy, time, sys, random
import pandas as pd, numpy as np
from collections import deque
from datetime import datetime
import pickle
import warnings



# Normalization of observations:
from sklearn import preprocessing


import numpy as np

# Custom Imports:
from render.graph import TradingGraph
from envs.to_gym import ContiniousGymEnvironment
from utils.eval import eval_policy, evaluate
import preprocessing.transforms as T



class ContiniousStockEnvironment(ContiniousGymEnvironment):

    """A Quantitative Finance trading environment for OpenAI gym """
    metadata = {'render.modes': ['human', 'system', 'none']}
    scaler = preprocessing.MinMaxScaler()
    standardizer = preprocessing.StandardScaler()
    viewer = None

    def __init__(self, df:pd.DataFrame, lookback_window_size:int = 50, initial_balance:float = 10000,
                 commission:float = 0.00075, reward_strategy:str = 'base', sequential:bool=False,
                 render_range:int = 100, show_reward:bool=False, show_indicators:bool=False, debug:bool = False,
                 normalize_value = 40000, slippage:float = 0.01, transform_obs:str = 'None',
                 normalize_obs:bool=True, num_indicator:int = 9, name:str ="Custom Trading Environment"):
        
        super(ContiniousStockEnvironment, self).__init__(lookback_window_size, num_indicator)

        assert reward_strategy in ['base','incremental', 'benchmark'], f'Unknown reward stratedy {reward_strategy}'
        assert transform_obs in ['None','minmax', 'mean', 'diff', 'log_diff'], f'Unknown transformation for observation space {transform_obs}'

        if not isinstance(df, (pd.DataFrame)):
            warnings.warn(f'df object must be a pd.DataFrame, got {type(df)}') 

        # Name of the environment:
        self.name = name

        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.reward_strategy = reward_strategy
        
        self.debug = debug
        
        if self.debug:
            self.shapes = {'df': self.df.shape}

        self.commission = commission
        self.slippage = slippage       
        self.sequential = sequential
        self.normalize_obs = normalize_obs
        self.transform_obs = transform_obs
        
        self.transform_names = ['None', 'minmax', 'mean_norm', 'diff', 'log_diff']
        self.transforms = [T.identity, T.max_min_normalize, T.mean_normalize, T.difference, T.log_and_difference]

        self.transform_dict = {name:transform for name, transform in zip(self.transform_names, self.transforms)}

        self.normalize_value = normalize_value # not used anymore

        self.ohlcv_cols = ['Open','High','Low','Close', 'Volume']
        self.indicator_cols = ['sma7', 'sma25','sma99','bb_bbm','bb_bbh', 'bb_bbl', 'psar', 'MACD', 'RSI']
        
        # How many bars of history we want to render:
        self.render_range = render_range # render range in visualization
        self.show_reward = show_reward # show order reward in rendered visualization
        self.show_indicators = show_indicators # show main indicators in rendered visualization


        self._start_trade_session()
        
    def _start_trade_session(self):
        """ Initalizes the order, market and indicator history with fixed sizes """
       
        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)
        self.indicators_history = deque(maxlen=self.lookback_window_size)
        
    
    def adjust_slippage(self, threshold:int = 0.9):
        """ Slippage Modifier """
        chance = np.random.uniform(0,1)

        if chance > threshold:
            self.slippage = np.random.uniform(0.01, 0.5)
        else:
            self.slippage = np.random.uniform(0.00001, 0.01)


    def get_order_history(self):
        """  Returns portolio of current state """
        return [self.balance, 
                self.net_worth,
                self.crypto_bought,
                self.crypto_sold,
                self.crypto_held]
    
    def apply_transform_obs(self, state):
        """  Applies the transformation to the observations """
        return self.transform_dict[self.transform_obs](state)
      
      
    
    def get_market_history(self, current_step):
        """ Returns  OHLCV """
        return  [self.df.loc[current_step, 'Open'],
                 self.df.loc[current_step, 'High'],
                 self.df.loc[current_step, 'Low'],
                 self.df.loc[current_step, 'Close'],
                 self.df.loc[current_step, 'Volume']]

    
    def get_indicator_history(self, current_step):
        """ Returns list of indicators in the current step """
        return [self.df.loc[current_step, 'sma7'],
                self.df.loc[current_step, 'sma25'],
                self.df.loc[current_step, 'sma99'],
                self.df.loc[current_step, 'bb_bbm'],
                self.df.loc[current_step, 'bb_bbh'],
                self.df.loc[current_step, 'bb_bbl'],
                self.df.loc[current_step, 'psar'],
                self.df.loc[current_step, 'MACD'],
                self.df.loc[current_step, 'RSI']]
    
    def _reset_session(self):
        """ Starts a new session """
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.punish_value = 0
        self.rewards = deque(maxlen=self.render_range)
        self.trades = deque(maxlen=self.render_range)
        

    def reset(self, n_iter = 500):
        """ Resets the environment and returns new observation """
        
        if self.sequential:
            self.visualization = TradingGraph(render_range = self.render_range, 
                                              show_reward = self.show_reward,
                                              show_indicators = self.show_indicators) 
        
        
        self._reset_session()
        self.env_steps_size = n_iter
       

        # used for training dataset, random traversing:
        if not self.sequential:
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - n_iter)
            self.end_step = self.start_step + n_iter
            
        # used for testing dataset
        else: 
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i

            self.orders_history.append(self.get_order_history())
            self.market_history.append(self.get_market_history(current_step))
            self.indicators_history.append(self.get_indicator_history(current_step))

        if self.debug:
            self.shapes['market_history'] = np.shape(self.market_history)
            self.shapes['orders_history'] = np.shape(self.orders_history)
            self.shapes['indicators_history'] = np.shape(self.indicators_history)

        state = np.concatenate([self.market_history, self.orders_history, self.indicators_history], axis=1)

        
        if self.normalize_obs:
            state = self.scaler.fit_transform(state)
            
        
        if self.transform_obs is not None:
            state = self.apply_transform_obs(state)

        return state
    

    
    def _next_observation(self):
        """ Get new the data points """

        self.market_history.append(self.get_market_history(self.current_step))
        self.indicators_history.append(self.get_indicator_history(self.current_step))

        obs = np.concatenate([self.market_history, self.orders_history, self.indicators_history], axis=1) 


        if self.normalize_obs:
            obs = self.scaler.fit_transform(obs)
            
        if self.transform_obs is not None:
            obs = self.apply_transform_obs(obs)

        return obs

    def _get_current_price(self):
        """ Returns current Open price """
        return self.df.loc[self.current_step, 'Open'] + (np.random.rand())
    
    def _get_random_current_price(self):
        """ Returns current price from Open and Close values randomly """
        return random.uniform(
            self.df.loc[self.current_step, 'Open'],
            self.df.loc[self.current_step, 'Close']
        )
   
    def get_dcl(self):
        """ Returns dict of DCL """
        return {'Date'  : self.df.loc[self.current_step, 'Date'],
                'High'  : self.df.loc[self.current_step, 'High'],
                'Low'   : self.df.loc[self.current_step, 'Low']}
    
    def action_decompose(self, action):
        """ Returns decomposed action, action type and amount """
        action = action[0]
        amount = abs(action)
        
        if 0.1 > action > - 0.1:
            action = 0 # Hold
        elif action >= 0.1 :
            action = 1 # Buy
        else:
            action = 2 # Sell
        return action, amount

    def step(self, action):
        """ Performs one step (BUY, SELL, HOLD) with given action """
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1
        
        
        action_type, amount = self.action_decompose(action)

        
        assert action_type in [0, 1, 2], f'Unknown action type found : {action_type}, should be in [0, 1, 2]'
        assert 0.0 <= amount <= 1.0, f'Unknown amount type found : {amount}, should be in [0, 0.1, 0.2, ... , 1]'

        current_price = self._get_current_price()
 
        self.adjust_slippage()
        
        # Hold:
        if action_type == 0:
            pass
        

        # Buy: 
        elif action_type == 1 and self.balance > self.initial_balance * 0.01:
            
            self.crypto_bought = self.balance / current_price * amount
            adjust_price = (1 + self.commission) * (1 + self.slippage) 
            self.balance -= (self.crypto_bought * current_price) * adjust_price
            self.crypto_held += self.crypto_bought

            self.trades.append({**self.get_dcl(),
                                'total' : self.crypto_bought,
                                'type'  : "buy",
                                'current_price': current_price})

            self.episode_orders += 1
        
        # Sell:
        elif action_type == 2 and self.crypto_held > 0:
            # Sell 
            self.crypto_sold = self.crypto_held * amount
            adjust_price = (1 - self.commission) * (1 - self.slippage) 
            self.balance += (self.crypto_sold * current_price)  * adjust_price
            self.crypto_held -= self.crypto_sold

            self.trades.append({**self.get_dcl(),
                                'total' : self.crypto_sold,
                                'type'  :  "sell",
                                'current_price': current_price})
            self.episode_orders += 1

        
        self.net_worth = self.balance + self.crypto_held * current_price
        self.orders_history.append(self.get_order_history())

        
        reward = self.get_reward()
        done =  self.get_done()
        obs = self._next_observation()
        info = self.get_info() 
        
        
        self.prev_net_worth = self.net_worth

        return obs, reward, done, info
    
    def get_reward(self):
        """ Returns reward with initially given stragedy """
        
        if self.reward_strategy == 'incremental':
            reward = self.get_incremental_reward()
            
        elif self.reward_strategy == 'benchmark':
            reward = self.get_benchmark_reward()
            
        else:           
            reward = self.get_base_reward()
            
        return reward
    
    def get_base_reward(self):
        """ Vanilla reward function, performs temporal difference between current and previous net worth"""
        return self.net_worth - self.prev_net_worth
        
    
    def get_done(self):
        """ Returns True if net worth is equal or smaller to %50 of initial balance """
        done = self.net_worth <= (self.initial_balance / 2)
        return done if isinstance(done, bool) else bool(done)
        
    
    def get_info(self):
        """ Returns a dict of last trade, orders, and shape of the matrices is debug True"""
        return {
                'trade'  :    self.trades[-1] if len(self.trades) > 0 else [],
                'order'  :    {'balance'      : self.balance, 
                              'net_worth'     : self.net_worth,
                              'crypto_bought' : self.crypto_bought,
                              'crypto_sold'   : self.crypto_sold,
                              'crypto_held'   : self.crypto_held},
                'shapes' :    self.shapes if self.debug else None
                } 

    def get_benchmark_reward(self):
        """ Returns reward as the squared distance between benchmark profit and current profit """
        profit_percent = (self.net_worth - self.initial_balance) / self.initial_balance * 100
        benchmark_profit = (self._get_current_price()  / self.df.loc[self.start_step, 'Open'] - 1) * 100
        diff = profit_percent - benchmark_profit
        reward = np.sign(diff) * (diff)**2
        return reward
        
    def get_incremental_reward(self):
        self.punish_value += self.net_worth * 0.00001
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - self.trades[-1]['total']*self.trades[-1]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - self.trades[-2]['total']*self.trades[-2]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
        else:
            return 0 - self.punish_value
        
    def seed(self, random_state = 42):
        """ Seed environment for reproducibility """
        np.random.seed(random_state)
        random.seed(random_state)
        
    def _render_to_file(self, filename='render_logs.txt'):
        """ Write the profit, balance, shares held, net worth to file"""
        profit = self.net_worth - self.initial_balance
        file = open(os.path.join('./logs',filename), 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.crypto_held}\n')
        file.write(f'Net worth: {self.net_worth}\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()

    def render(self, mode='human', close=False, visualize:bool = False,
               render_to_file:bool = False, print_details:bool = False):
        """ Renders the environment by 3 different way: 
            
            1) Printing net worth and profit
            2) Writing the profit, balance, shares held, net worth to file
            3) Visualizations of live trading
        """

        if print_details:
            print('_' * 30)
            print(f'Step: {self.current_step}, \n Net Worth: {self.net_worth},\n Profit: {self.net_worth - self.initial_balance} \n')

        if render_to_file:
            self._render_to_file()

        if visualize:
            img = self.visualization.render(self.df.loc[self.current_step], self.net_worth, self.trades)
            return img
        

    def close(self):
        """ Closes the session """
        pass
