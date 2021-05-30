import gym
from gym import spaces
import numpy as np

class DiscreteGymEnvironment(gym.Env):

    def __init__(self, lookback_window_size:int = 300, num_indicator:int = 9):

        super(DiscreteGymEnvironment, self).__init__()

        # Actions of the format Buy 1/10, Sell 3/10, Hold, etc.
        # Action space is represented as discrete set of 3 options
        # i.e., buy, sell and hold. And, another discrete set of 10
        # amounts (1/10,...,10/10). When the buy is selected, we will buy
        # amount * self.balance worth of crypto. For the sell action, we will sell
        # amount * self.btc_held worth of BTC. Hold action ignores the amount and do
        # nothing
        # Actions of the format Buy 1/10, Sell 3/10, Hold (amount ignored), etc.
        self.action_space = spaces.MultiDiscrete([3, 10])

        # Prices contains the OHCLV (Open,High,Close,Low,Volume) values, net worth and trade history
        # Observes the OHCLV values, net worth, and trade history
        self.observation_space = spaces.Box(
            low=0, high=1, shape= (lookback_window_size, 10 + num_indicator), dtype=np.float16)
        
        
    def action_decompose(self, action):
        """ Returns decomposed action, action type and amount """
        action_type = action[0]
        amount = action[1] / 10
        return action_type, amount
        
class ContiniousGymEnvironment(gym.Env):

    def __init__(self, lookback_window_size:int = 300, num_indicator:int = 9):

        super(ContiniousGymEnvironment, self).__init__()

        self.action_space = spaces.Box(low = -1, high = 1, shape = (1,)) 
        self.observation_space = spaces.Box(
            low=0, high=1, shape= (lookback_window_size, 10 + num_indicator), dtype=np.float16)
        
        
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
    