import pandas as pd
import pandas_ta as ta
import numpy as np
from garch_model import GARCHModel
from datetime import datetime


def add_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    print(f"Index is : {df.index}")

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

    # Range
    df["rangeocp"] = 100 * (df["open"] - df["close"])/df["close"]

    df["rangehlp"] = 100 * (df["high"] - df["low"])/df["close"]

    # VWAP
    df.ta.vwap(append=True)
    df = df.rename(columns={"VWAP_D": "vwap_d"})

    # ATR
    df.ta.atr(append=True)
    df = df.rename(columns={"ATRr_14": "atrr_14"})

    # --- GARCH VOLATILITY ---
    garch = GARCHModel()

    df = garch.fit_predict(df)

    # fallback if NaN
    df["garch_vol"] = df["garch_vol"].fillna(df["volatility_10"])

    # volatility change
    df["garch_vol_change"] = df["garch_vol"] - df["volatility_10"]

    # volatility ratio
    df["vol_ratio"] = df["garch_vol"] / (df["volatility_10"] + 1e-9)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=False)

    # --- CLEAN ---
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=False)

    return df

def add_select_features(df, config=None):
    df = df.copy()
    use = config or {}
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    #print(f"Index is : {df.index}")

    # RETURNS
    if use.get("returns", True):
        df["return_1"] = df["close"].pct_change(1)
        df["return_3"] = df["close"].pct_change(3)
        df["return_5"] = df["close"].pct_change(5)

    # MOMENTUM
    if use.get("momentum", True):
        df["momentum_5"] = df["return_1"].rolling(5).mean()

    # VOLATILITY
    if use.get("volatility", True):
        df["volatility_10"] = df["return_1"].rolling(10).std()

    # TREND
    if use.get("trend", True):
        df["ma_10"] = df["close"].rolling(10).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_ratio"] = df["ma_10"] / df["ma_20"]

    # MEAN REVERSION
    if use.get("mean_reversion", True):
        mean = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["zscore"] = (df["close"] - mean) / std

    # Range
    if use.get("day_range_oc", True):
        df["rangeocp"] = 100 * (df["open"] - df["close"])/df["close"]

    if use.get("day_range_hl", True):
        df["rangehlp"] = 100 * (df["high"] - df["low"])/df["close"]

    # VWAP
    if use.get("vwap", True) and "vwap_d" not in df.columns:
        df.ta.vwap(append=True)
        df = df.rename(columns={"VWAP_D": "vwap_d"})

    # ATR
    if use.get("atr", True) and "atrr_14" not in df.columns:
        df.ta.atr(append=True)
        df = df.rename(columns={"ATRr_14": "atrr_14"})

    # --- GARCH VOLATILITY ---
    if use.get("garch", True):

        garch = GARCHModel()

        df = garch.fit_predict(df)

        # fallback if NaN
        df["garch_vol"] = df["garch_vol"].fillna(df["volatility_10"])

        # volatility change
        df["garch_vol_change"] = df["garch_vol"] - df["volatility_10"]

        # volatility ratio
        df["vol_ratio"] = df["garch_vol"] / (df["volatility_10"] + 1e-9)        

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna().reset_index(drop=False)
    
    df['date'] = pd.to_datetime(df['date'])
    #print(df.info())
    return df

def add_basic_features(df):
    df = df.copy()

    # rolling volatility
    df["volatility"] = df["returns"].rolling(10).std()

    # momentum
    df["momentum"] = df["returns"].rolling(5).mean()

    df = df.dropna().reset_index(drop=True)

    return df