import numpy as np

def time_split(df, lookback_window_size, test_window):
    train_df = df[:-test_window-lookback_window_size]
    test_df = df[-test_window-lookback_window_size:]

    return train_df, test_df
