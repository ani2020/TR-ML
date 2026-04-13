import pandas as pd
import pandas_ta as ta
import numpy as np
from garch_model_spot import GARCHModel
from datetime import datetime

def safe_add_feature(featurename: str, newdf, dfinput):

    if featurename not in newdf.columns and featurename in dfinput.columns:
        newdf[featurename] = dfinput[featurename]


def add_features(df):
    df = df.copy()
    if 'date' not in df.index and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
        df = df.set_index('date')
    print(f"Index is : {df.index}")

    # --- RETURNS ---
    df["f_return_1"] = np.log(df["close"] / df["close"].shift(1))
    df["f_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["f_return_10"] = np.log(df["close"] / df["close"].shift(10))

    # --- MOMENTUM ---
    df["f_momentum_5"] = df["f_return_1"].rolling(5).mean()
    df["f_momentum_10"] = df["f_return_1"].rolling(10).mean()
    df["f_momentum_20"] = df["f_return_1"].rolling(20).mean()

    # --- VOLATILITY ---
    df["f_volatility_5"] = df["f_return_1"].rolling(5).std()
    df["f_volatility_10"] = df["f_return_1"].rolling(10).std()

    # --- MOVING AVERAGES ---
    df["f_ma_10"] = df["close"].rolling(10).mean()
    df["f_ma_20"] = df["close"].rolling(20).mean()

    # --- TREND ---
    df["f_ma_ratio"] = df["f_ma_10"] / df["f_ma_20"]
    df["f_price_ma_ratio"] = df["close"] / df["f_ma_10"]

    # --- MEAN REVERSION (Z-SCORE) ---
    rolling_mean = df["close"].rolling(20).mean()
    rolling_std = df["close"].rolling(20).std()

    df["f_zscore"] = (df["close"] - rolling_mean) / rolling_std

    # --- RSI ---
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["f_rsi"] = 100 - (100 / (1 + rs))

    # Range
    df["f_rangeocp"] = 100 * (df["open"] - df["close"])/df["close"]

    df["f_rangehlp"] = 100 * (df["high"] - df["low"])/df["close"]

    # VWAP
    if "f_vwap_d" not in df.columns:
        df.ta.vwap(append=True)
        df = df.rename(columns={"VWAP_D": "f_vwap_d"})

    # ATR
    if "f_atrr_14" not in df.columns:
        df.ta.atr(append=True)
        df = df.rename(columns={"ATRr_14": "f_atrr_14"})

    # --- GARCH VOLATILITY ---
    garch = GARCHModel()

    df = garch.fit_predict(df)

    # fallback if NaN
    df["f_garch_vol"] = df["f_garch_vol"].fillna(df["f_volatility_10"])

    # volatility change
    df["f_garch_vol_change"] = df["f_garch_vol"] - df["f_volatility_10"]

    # volatility ratio
    df["f_vol_ratio"] = df["f_garch_vol"] / (df["f_volatility_10"] + 1e-9)

    df['f_vix_divergence'] = df['f_vix_ret'] + df['f_return_1']
    df['f_price_vix_signal'] = df['f_return_1'] * df['f_vix_ret']

    # --- CLEAN ---
    #df = df.replace([np.inf, -np.inf], np.nan)
    #df = df.dropna().reset_index(drop=False)

    return df

def add_select_features(dfi, config=None):

    df = dfi[['fut_expiry', 'open','high','close','low', 'volume', 
              'fut_open','fut_high','fut_low','fut_close','fut_prevclose', 'fut_volume',
              'fut_oi','fut_chgoi','lot','spot',
             'adj_fut_open','adj_fut_high','adj_fut_low','adj_fut_close','adj_fut_prevclose',
              'vix_open','vix_high','vix_low','vix_close','vix_prevclose' ]].copy()
    safe_add_feature("date", df, dfi)
    df = dfi.copy()
    #df = pd.DataFrame()
    use = config or {}
    if 'date' not in dfi.index and 'date' in dfi.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    #print(f"Index is : {df.index}")

    # RETURNS
    if use.get("returns", False):
        df["f_return_1"] = dfi["f_return_1"]
        df["f_return_5"] = dfi["f_return_5"]
        df["f_return_10"] = dfi["f_return_10"]

    # MOMENTUM
    if use.get("momentum", False):
        df["f_momentum_5"] = dfi["f_momentum_5"]

    # VOLATILITY
    if use.get("volatility", False):
        df["f_volatility_10"] = dfi["f_volatility_10"]

    # TREND
    if use.get("trend", False):
        df["f_ma_10"] = dfi["f_ma_10"]
        df["f_ma_20"] = dfi["f_ma_20"] 
        df["f_ma_ratio"] = dfi["f_ma_ratio"]

    # MEAN REVERSION
    if use.get("mean_reversion", False):
        df["f_zscore"] = dfi["close"]

    # Range
    if use.get("day_range_oc", False):
        df["f_rangeocp"] = dfi["f_rangeocp"]

    if use.get("day_range_hl", False):
        df["f_rangehlp"] = dfi["f_rangehlp"]

    # VWAP
    if use.get("vwap", False) and "f_vwap_d" not in df.columns:
        df["f_vwap_d"] = dfi["f_vwap_d"]

    # ATR
    if use.get("atr", False) and "f_atrr_14" not in df.columns:
        df["f_atrr_14"] = dfi["f_atrr_14"]

    # -- VIX features
    if use.get("vix", False) and 'f_vix_ret' not in df.columns:
        df['f_vix_ret'] = dfi['f_vix_ret']
        

    if use.get("interaction", False):
        df['f_vix_divergence'] = dfi['f_vix_divergence']
        df['f_price_vix_signal'] = dfi['f_price_vix_signal']

    # --- GARCH VOLATILITY ---
    if use.get("garch", False):

        # fallback if NaN
        df["f_garch_vol"] = dfi["f_garch_vol"]

        # volatility change
        df["f_garch_vol_change"] = dfi["f_garch_vol_change"]

        # volatility ratio
        df["f_vol_ratio"] = dfi["f_vol_ratio"]        

    #print(df.info())
    return df

def add_basic_features(df):
    df = df.copy()

    # rolling volatility
    df["volatility"] = df["returns"].rolling(10).std()

    # momentum
    df["momentum"] = df["returns"].rolling(5).mean()

    df = df.dropna().reset_index(drop=False)

    return df