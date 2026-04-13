import numpy as np

def add_returns(df):
    df = df.copy()
    df["fut_returns"] = np.log(df["adj_fut_close"] / df["adj_fut_close"].shift(1))
    #df = df.dropna()
    return df