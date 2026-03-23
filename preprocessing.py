import numpy as np

def add_returns(df):
    df = df.copy()
    df["returns"] = np.log(df["close"] / df["close"].shift(1))
    df = df.dropna()
    return df