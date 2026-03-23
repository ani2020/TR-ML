import pandas as pd
import numpy as np


def add_features(df):
    df = df.copy()

    # --- RETURNS ---
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)

    # --- MOMENTUM ---
    df["momentum_5"] = df["return_1"].rolling(5).mean()
    df["momentum_10"] = df["return_1"].rolling(10).mean()

    # --- VOLATILITY ---
    df["volatility_5"] = df["return_1"].rolling(5).std()
    df["volatility_10"] = df["return_1"].rolling(10).std()

    # --- MOVING AVERAGES ---
    df["ma_10"] = df["close"].rolling(10).mean()
    df["ma_20"] = df["close"].rolling(20).mean()

    # --- TREND ---
    df["ma_ratio"] = df["ma_10"] / df["ma_20"]
    df["price_ma_ratio"] = df["close"] / df["ma_10"]

    # --- MEAN REVERSION (Z-SCORE) ---
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()

    df["zscore"] = (df["close"] - rolling_mean) / rolling_std

    # --- RSI ---
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # --- CLEAN ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=True)

    return df

def add_basic_features(df):
    df = df.copy()

    # rolling volatility
    df["volatility"] = df["returns"].rolling(10).std()

    # momentum
    df["momentum"] = df["returns"].rolling(5).mean()

    df = df.dropna().reset_index(drop=True)

    return df