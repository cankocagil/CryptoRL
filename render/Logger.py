import os
import sys

class Logger(object):
    """
        Logger class to keep logs of rendered files.

    """

    def __init__(self, root_dir:str = './logs', filename:str = 'render_logs.txt'):
        self.filename = filename
        self.root_dir = root_dir


        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

            print(f'Path {root_dir} is created succesfully to keep Logs!')


        print(f'Path {root_dir} is already available to keep Logs!')


    def render_to_file(self,
                       current_step:int,
                       balance:float, 
                       btc_held:float,
                       net_worth:float,
                       profit:float,
                       verbose:bool=False,
                       path_dir:str = 'utils'):

        """

            Given the necessary information on the current market and balance, keep logs in txt file with specified path.

            Arguments:
                       current_step:int  = Current step in active data frame
                       balance:float     = Current Balance
                       btc_held:float    = Current BTC held
                       net_worth:float   = Current net worth
                       profit:float      = Current net worth - initial balance 
                       verbose:bool      = False = True enables logs printings
                       path_dir:str      = Path of main file to keep logs in utils directory


        """
        self.path_dir = path_dir
        
        file = open(os.path.join(path_dir,self.root_dir,self.filename), 'a+')
        file.write(f'Step: {current_step}\n')
        file.write(f'Balance: {balance}\n')
        file.write(f'Shares held: {btc_held}\n')
        file.write(f'Net worth: {net_worth}\n')
        file.write(f'Profit: {profit}\n\n')
        file.close()


        if verbose:
            print(f'Succesfully Logged in {os.path.join(self.root_dir,self.filename)}')



    def clear(self):
        """
            Erasing the contents of the log file.
        """

        file = open(os.path.join(self.path_dir,self.root_dir,self.filename), 'a+')
        file.truncate(0)
        file.close()