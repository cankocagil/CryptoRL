from datetime import datetime
import os 
import pandas as pd
import numpy as np
from preprocessing.indicators import AddIndicators


def preprocess_price(df, add_indicators:bool = True):
    return AddIndicators(df)

def preprocess_df(df, add_indicators:bool = True):

    df = df[['Date','Open','High','Low','Close', 'Volume']]
    df = df.sort_values('Date')
    df['Date'] = df['Date'].apply(lambda x : x[:16])

    assert unique_cols(df['Date'].apply(lambda x : len(x))), 'Date length must be unique'

    if add_indicators:
        print('Adding indicators...')
        if not os.path.exists('./data/BTCUSDT-1m-data_historical_indicators.csv'):
            df = AddIndicators(df)
            df.to_csv('./data/BTCUSDT-1m-data_historical_indicators.csv')
        else:
            print('Data is fetched from path!')
            df = pd.read_csv('./data/BTCUSDT-1m-data_historical_indicators.csv')
    return df



def unique_cols(df):
    a = df.to_numpy()
    return (a[0] == a).all(0)


def write_to_file(Date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
    for i in net_worth: 
        Date += " {}".format(i)
    #print(Date)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file = open("logs/"+filename, 'a+')
    file.write(Date+"\n")
    file.close()
