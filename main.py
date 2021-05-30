
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='gpu')
import pandas as pd

# Custom Imports:
from engine.train import train 
from engine.test import test 
#from engine.random import random_games 
from agents.agent import Agent
from envs.stock_env import StockEnvironment
from utils.pipelines import preprocess_df, preprocess_price
from utils.split import time_split


if __name__ == "__main__":            
    df = pd.read_csv('./data/BTCUSDT-1m-data_historical.csv')
    df = preprocess_df(df)
    #df = pd.read_csv('./data/pricedata.csv').sort_values('Date')
    #df = AddIndicators(df)

    lookback_window_size = 200 # 6 hours
    test_window = 120 # Since 2021
    train_df, test_df = time_split(df, lookback_window_size, test_window)

    agent = Agent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5,
                  optimizer=Adam, batch_size = 32, model="LSTM")

    train_env = StockEnvironment(df = train_df, lookback_window_size = lookback_window_size, 
                                 normalize_obs=False, initial_balance = 10000, render_range = 300,
                                 reward_strategy='incremental', name = "Custom Trading Environment v1")

    train(train_env, agent, train_episodes=2, training_batch_size=500)


    test_env = False

    if test_env:
        test_env = StockEnvironment(test_df, lookback_window_size=lookback_window_size, initial_balance = 10000,
                                    normalize_obs=False, render_range = lookback_window_size, show_reward=True, show_indicators=True)


        folder, name = '2021_01_18_22_18_Crypto_trader', '1933.71_Crypto_trader'
        folder_3000, name3000 = '2021-05-06 18: 01_trader', '3135.59_Crypto_trader'
        folderCNN, nameCNN = '2021-05-08 14: 32_trader', '1464.28_Crypto_trader'
        folderLSTM, nameLSTM = '2021-05-09 15: 17_trader', '87.48_Crypto_trader'
        
        test(test_env, agent, visualize=True, test_episodes=1,
            folder=folderCNN, name=nameCNN, comment="")
        

        #random_games(test_env, visualize=True, test_episodes=1)

        print(test_env.shapes)