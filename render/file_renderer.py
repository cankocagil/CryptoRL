import os
import sys
import warnings
import logging
import pandas as pd

class SpreadSheet(object):
    """
        SpreadSheet object for keeping history of trades.

    """

    def __init__(self,
                root_dir:str = './render/spreadsheets',
                checkpoint_dir:bool = False):
     
        self.root_dir = root_dir
        self.trades = pd.DataFrame()
        self.agent_history = pd.DataFrame()


        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

            print(f'Directory {root_dir} is created!')

        
        self.trades_file = os.path.join(root_dir,'trades') + '.csv'
        self.agent_history_file = os.path.join(root_dir,'agent_history') + '.csv'

        

        if  os.path.exists(self.trades_file):
            self.trades = pd.read_csv(self.trades_file)
        
        if  os.path.exists(self.agent_history_file):
            self.agent_history = pd.read_csv(self.agent_history_file)

            

    def add(self,
            time:str,
            current_step:int,
            balance:float, 
            btc_held:float,
            net_worth:float,
            profit:float,
            trade:dict,
            verbose:bool=False,
            ):

        """

            Given the necessary information on the current market and balance, keep logs in txt file with specified path.

            Arguments:
                       current_step:int  = Current step in active data frame
                       balance:float     = Current Balance
                       btc_held:float    = Current BTC held
                       net_worth:float   = Current net worth
                       profit:float      = Current net worth - initial balance 
                       verbose:bool      = False = True enables logs printings
                       


     """ 
        self.load()
     
        self.agent_history.append({
            'time'         : time,
            'balance'      : balance,
            'current_step' : current_step,
            'btc_held'     : btc_held,
            'net_worth'    : net_worth,
            'profit'       : profit
        })
        
        self.trades.append(trade)  


        self.save()

    def save(self):
        self.agent_history.to_csv(self.agent_history_file)
        self.trades.to_csv(self.trades_file)

    def load(self):
        self.agent_history = pd.read_csv(self.agent_history_file)
        self.trades = pd.read_csv(self.trades_file)

       
      
    def clear(self):
        """
            Erasing the contents of the log file.
        """

        self.agent_history = self.agent_history[0:0]
        self.trades = self.trades[0:0]