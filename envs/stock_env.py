import os, copy, time, sys, random
from matplotlib.pyplot import axis
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pandas as pd, numpy as np
from collections import deque
from tensorboardX import SummaryWriter
from tensorflow.keras.optimizers import Adam, RMSprop
#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='gpu')

from datetime import datetime

# Custom Imports:
from render.graph import TradingGraph
# Normalization of observations:
from sklearn import preprocessing



class StockEnvironment(object):

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

        assert reward_strategy in ['base','incremental']

        # To-do: Type checking
        if isinstance(df, (pd.DataFrame)):
            pass
            #raise TypeError(f'df object must be a DataFrame, got {type(df)}')

        # Name of the environment:
        self.name = name

        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.reward_strategy = reward_strategy

        self.shapes = {'df': self.df.shape}

        # How many bars of history we want to render:
        self.render_range = render_range # render range in visualization
        self.show_reward = show_reward # show order reward in rendered visualization
        self.show_indicators = show_indicators # show main indicators in rendered visualization


        self.commission = commission
        self.slippage = slippage       
        self.serial = serial
        self.normalize_obs = normalize_obs

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values 
        # for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.indicators_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value

        self.ohlcv_cols = ['Open','High','Low','Close', 'Volume']
        self.indicator_cols = ['sma7', 'sma25','sma99','bb_bbm','bb_bbh', 'bb_bbl', 'psar', 'MACD', 'RSI']

    def adjust_slippage(self, threshold_chance:int = 0.9):
        chance = np.random.uniform(0,1)

        if chance > threshold_chance:
            self.slippage = np.random.uniform(0.01, 0.5)
        else:
            self.slippage = np.random.uniform(0.00001, 0.01)


    def get_order_history(self):
        return [self.balance, 
                self.net_worth,
                self.crypto_bought,
                self.crypto_sold,
                self.crypto_held]

    
    def get_market_history(self, current_step):
        return  [self.df.loc[current_step, 'Open'],
                 self.df.loc[current_step, 'High'],
                 self.df.loc[current_step, 'Low'],
                 self.df.loc[current_step, 'Close'],
                 self.df.loc[current_step, 'Volume']]

    
    def get_indicator_history(self, current_step):
        return [self.df.loc[current_step, 'sma7'],
                self.df.loc[current_step, 'sma25'],
                self.df.loc[current_step, 'sma99'],
                self.df.loc[current_step, 'bb_bbm'],
                self.df.loc[current_step, 'bb_bbh'],
                self.df.loc[current_step, 'bb_bbl'],
                self.df.loc[current_step, 'psar'],
                self.df.loc[current_step, 'MACD'],
                self.df.loc[current_step, 'RSI']]


    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        """
            Random traversing is important aspect for our agent since we will have much more unique data when
            trained for long time. Hence, we create more unique data points with combinations of account balance,
            trades taken, and previously seen price action.

        Args:
            env_steps_size (int, optional): [description]. Defaults to 0.

        Returns:
            [type]: [description]
        """

        # Rendering environment:
        self.visualization = TradingGraph(render_range = self.render_range, 
                                          show_reward = self.show_reward,
                                          show_indicators = self.show_indicators) 

        self.trades = deque(maxlen=self.render_range) # limited orders memory for visualization
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0 # track episode orders count
        self.prev_episode_orders = 0 # track previous episode orders count
        self.rewards = deque(maxlen=self.render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0


        # used for training dataset, random traversing:
        if env_steps_size > 0: 
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size

        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i

            #self.orders_history.append(self.get_order_history())
            #self.market_history.append(self.get_market_history(current_step))
            #self.indicators_history.append(self.get_indicator_history(current_step))
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        ])

            self.indicators_history.append(
                [self.df.loc[current_step, 'sma7'] / self.normalize_value,
                                        self.df.loc[current_step, 'sma25'] / self.normalize_value,
                                        self.df.loc[current_step, 'sma99'] / self.normalize_value,
                                        self.df.loc[current_step, 'bb_bbm'] / self.normalize_value,
                                        self.df.loc[current_step, 'bb_bbh'] / self.normalize_value,
                                        self.df.loc[current_step, 'bb_bbl'] / self.normalize_value,
                                        self.df.loc[current_step, 'psar'] / self.normalize_value,
                                        self.df.loc[current_step, 'MACD'] / 400,
                                        self.df.loc[current_step, 'RSI'] / 100
                                        ])
        #self.market_history = self.min_max_scaler(self.market_history)
        #self.orders_history = self.min_max_scaler(self.orders_history)
        #self.indicators_history = self.min_max_scaler(self.indicators_history)


        #self.market_history = self.scaler.fit_transform(self.market_history)
        #self.orders_history = self.scaler.fit_transform(self.orders_history)
        #self.indicators_history = self.scaler.fit_transform(self.indicators_history)

        self.shapes['market_history'] = np.shape(self.market_history)
        self.shapes['orders_history'] = np.shape(self.orders_history)
        self.shapes['indicators_history'] = np.shape(self.indicators_history)

        #state = np.concatenate([self.market_history, self.orders_history, self.indicators_history], axis=1) / self.normalize_value
        
        #state = np.concatenate((state, self.indicators_history), axis=1)

        state = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
        state = np.concatenate((state, self.indicators_history), axis=1)

        if self.normalize_obs:
            state = self.scaler.fit_transform(state)

        #self.market_history = pd.DataFrame(self.market_history, columns = self.ohlcv_cols)
        #self.indicators_history = pd.DataFrame(self.indicators_history, columns = self.indicator_cols)
        #self.orders_history = pd.DataFrame(self.orders_history)

        #self.market_history = deque(self.market_history, maxlen=self.lookback_window_size)
        #self.indicators_history = deque(self.indicators_history, maxlen=self.lookback_window_size)
        #self.orders_history = deque(self.orders_history, maxlen=self.lookback_window_size)
        return state

    def min_max_scaler(self, data, manuel:bool = False):

        if manuel:
            min_val = np.min(data, axis = 0)
            max_val = np.max(data, axis = 0)
            data = np.asarray(data)
            res = (data - min_val) / (max_val - min_val)
        else:       
            res = self.scaler.fit_transform(data)
        return deque(res, maxlen = self.lookback_window_size)

    # Get the data points for the given current_step
    def _next_observation(self):

        #self.market_history.append(self.get_market_history(self.current_step))
        #self.indicators_history.append(self.get_indicator_history(self.current_step))

        #self.market_history = self.min_max_scaler(self.market_history)
        #self.orders_history = self.min_max_scaler(self.orders_history)
        #self.indicators_history = self.min_max_scaler(self.indicators_history)
        
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    ])

        self.indicators_history.append([self.df.loc[self.current_step, 'sma7'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'sma25'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'sma99'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'bb_bbm'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'bb_bbh'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'bb_bbl'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'psar'] / self.normalize_value,
                                    self.df.loc[self.current_step, 'MACD'] / 400,
                                    self.df.loc[self.current_step, 'RSI'] / 100
                                    ])
        
        obs = np.concatenate((self.market_history, self.orders_history), axis=1) / self.normalize_value
        obs = np.concatenate((obs, self.indicators_history), axis=1)
        #self.market_history = self.scaler.fit_transform(self.market_history)
        #self.orders_history = self.scaler.fit_transform(self.orders_history)
        #self.indicators_history = self.scaler.fit_transform(self.indicators_history)

        #obs = np.concatenate([self.market_history, self.orders_history, self.indicators_history], axis=1) / self.normalize_value
        
        #obs = np.concatenate([obs, self.indicators_history], axis=1)    

        if self.normalize_obs:
            obs = self.scaler.fit_transform(obs)


        #self.market_history = pd.DataFrame(self.market_history, columns=self.ohlcv_cols)
        #self.indicators_history = pd.DataFrame(self.indicators_history, columns=self.indicator_cols)
        #self.orders_history = pd.DataFrame(self.orders_history)
        
        #self.market_history = deque(self.market_history, maxlen=self.lookback_window_size)
        #self.indicators_history = deque(self.indicators_history, maxlen=self.lookback_window_size)
        #self.orders_history = deque(self.orders_history, maxlen=self.lookback_window_size)
        
        return obs

    def _get_current_price(self):
        """

        Returns:
            current_price (float) :Current Value of BTC with random disturbance
        """
        return self.df.loc[self.current_step, 'Open'] + (np.random.rand() / 100)

    # Execute one time step within the environment
    def step(self, action, amount:float = 1):
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

        if action == 0: # Hold
            pass
        

        # Buy if current balance > initial balance:  
        elif action == 1 and self.balance > self.initial_balance / 100:
            #1 + (self.commissionPercent / 100)) * (1 + (self.maxSlippagePercent / 100))
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price * amount
            adjust_price = 1 #(1 + self.commission) * (1 + self.slippage) 
            self.balance -= (self.crypto_bought * current_price ) * adjust_price
            self.crypto_held += self.crypto_bought

            self.trades.append({'Date'  : Date,
                                'High'  : High,
                                'Low'   : Low,
                                'total' : self.crypto_bought,
                                'type'  : "buy",
                                'current_price': current_price})

            self.episode_orders += 1

        elif action == 2 and self.crypto_held > 0:
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

        self.orders_history.append([self.balance, 
                                    self.net_worth,
                                    self.crypto_bought,
                                    self.crypto_sold,
                                    self.crypto_held])

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
                'order'  :    {'balance'       : self.balance, 
                              'net_worth'     : self.net_worth,
                              'crypto_bought' : self.crypto_bought,
                              'crypto_sold'   : self.crypto_sold,
                              'crypto_held'   : self.crypto_held},
                'reward' :    reward,
                'shapes' :    self.shapes
                }  

        return obs, reward, done, info

    
    def get_reward(self):
        """

            Reward Stragery will be analyzed later...

        """
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

    def _render_to_file(self, filename='render_logs.txt'):
        profit = self.net_worth - self.initial_balance
        file = open(os.path.join('./logs',filename), 'a+')
        file.write(f'Step: {self.current_step}\n')
        file.write(f'Balance: {self.balance}\n')
        file.write(f'Shares held: {self.crypto_held}\n')
        file.write(f'Net worth: {self.net_worth}\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()

    

    # render environment
    def render(self,
               visualize:bool = False,
               render_to_file:bool = True, 
               print_details:bool = False):
        """
        Environment renderer.

        Args:
            visualize (bool, optional): [description]. Defaults to False.
            render_to_file (bool, optional): [description]. Defaults to True.
            print_details (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """

        if print_details:
            print('_' * 30)
            print(f'Step: {self.current_step}, \n Net Worth: {self.net_worth},\n Profit: {self.net_worth - self.initial_balance} \n')

        if render_to_file:
            self._render_to_file()

        if visualize:
            # Render the environment to the screen
            img = self.visualization.render(self.df.loc[self.current_step],
                                            self.net_worth,
                                            self.trades)
            return img








