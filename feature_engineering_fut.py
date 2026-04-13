import pandas as pd
import pandas_ta as ta
import numpy as np
from garch_model_fut import GARCHModel
from datetime import datetime


def safe_add_feature(featurename: str, newdf, dfinput):

    if featurename not in newdf.columns and featurename in dfinput.columns:
        newdf[featurename] = dfinput[featurename]

def fut_add_features(df):
    df = df.copy()
    if 'date' not in df.index and 'date' in df.columns and 'fut_expiry' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['fut_expiry'] = pd.to_datetime(df['fut_expiry'])
        df = df.set_index('date')
    #print(f"Index is : {df.index} number of records: {len(df)}")

    # --- RETURNS ---
    df['f_return_1'] = np.log(df["close"] / df["close"].shift(1))
    df['f_fut_return_1'] = np.log(df["adj_fut_close"] / df["adj_fut_close"].shift(1))
    df['f_fut_return_5'] = np.log(df["adj_fut_close"] / df["adj_fut_close"].shift(5))
    df['f_fut_return_10'] = np.log(df["adj_fut_close"] / df["adj_fut_close"].shift(10))
    
    # -- Basis features

    df["f_basis"] = df['fut_close'] - df['close']
    df['f_basis_pct'] = df["f_basis"] / df['close']
    df['f_basis_smooth'] = df['f_basis_pct'].ewm(span=5).mean()
    df['f_basis_zscore'] = (
        (df['f_basis_pct'] - df['f_basis_pct'].rolling(20).mean()) /
        df['f_basis_pct'].rolling(20).std())

    # --- MOMENTUM ---
    df["f_fut_momentum_5"] = df["f_fut_return_1"].rolling(5).mean()
    df["f_fut_momentum_10"] = df["f_fut_return_1"].rolling(10).mean()
    df["f_fut_momentum_20"] = df["f_fut_return_1"].rolling(10).mean()

    # --- VOLATILITY ---
    df["f_fut_volatility_5"] = df["f_fut_return_1"].rolling(5).std()
    df["f_fut_volatility_10"] = df["f_fut_return_1"].rolling(10).std()
    df["f_fut_volatility_20"] = df["f_fut_return_1"].rolling(10).std()
    df['f_fut_vol_regime'] = df['f_fut_volatility_10'] > df['f_fut_volatility_20']
    df['f_fut_range'] = (df["adj_fut_high"] - df["adj_fut_low"]) / df["adj_fut_close"]
    df['f_fut_body'] = (df["adj_fut_close"] - df["adj_fut_open"]) / df["adj_fut_open"]

    # --- MOVING AVERAGES ---
    df["f_fut_ma_5"] = df["adj_fut_close"].rolling(5).mean()
    df["f_fut_ma_10"] = df["adj_fut_close"].rolling(10).mean()
    df["f_fut_ma_20"] = df["adj_fut_close"].rolling(20).mean()

    # --- TREND ---
    df["f_fut_ma_ratio"] = df["f_fut_ma_5"] / df["f_fut_ma_20"]
    df["f_fut_price_ma_ratio"] = df["adj_fut_close"] / df["f_fut_ma_5"]

    # --- EMA ---
    df['f_fut_ema_5'] = df['adj_fut_close'].ewm(span=5).mean()
    df['f_fut_ema_20'] = df['adj_fut_close'].ewm(span=20).mean()

    df['f_fut_trend'] = df['f_fut_ema_5'] - df['f_fut_ema_20']

    # -- OI features
    #df['oi_change'] = df['fut_oi'].pct_change()
    df['f_oi_change'] = np.log(df['fut_oi'] / df['fut_oi'].shift(1))
    df['f_oi_change'] = df['f_oi_change'].clip(-0.5, 0.5)
    df['f_oi_vol'] = df['f_oi_change'].rolling(10).std()
    df['f_oi_momentum'] = df['f_oi_change'].rolling(5).mean()
    df['f_oi_zscore'] = (df['fut_oi'] - df['fut_oi'].rolling(30).mean()) / df['fut_oi'].rolling(30).std()
    df['f_price_oi_signal'] = df['f_fut_return_1'] * df['f_oi_change']
    df['f_trend_strength'] = np.sign(df['f_fut_return_1']) * np.sign(df['f_oi_change'])

    # --- MEAN REVERSION (Z-SCORE) ---
    rolling_mean = df["adj_fut_close"].rolling(20).mean()
    rolling_std = df["adj_fut_close"].rolling(20).std()
    
    df["f_fut_zscore"] = (df["adj_fut_close"] - rolling_mean) / rolling_std

    # --- RSI ---
    delta = df["adj_fut_close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["f_fut_rsi"] = 100 - (100 / (1 + rs))

    # Range
    df["f_fut_rangeocp"] = 100 * (df["adj_fut_open"] - df["adj_fut_close"])/df["adj_fut_close"]

    df["f_fut_rangehlp"] = 100 * (df["adj_fut_high"] - df["adj_fut_low"])/df["adj_fut_close"]

    # VWAP
    if "f_fut_vwap_d" not in df.columns: 
        df.ta.vwap(close=df["adj_fut_close"], high=df["adj_fut_high"], low=df["adj_fut_low"], volume=df["fut_volume"]  ,append=True)
        df = df.rename(columns={"VWAP_D": "f_fut_vwap_d"})

    # ATR
    if "f_fut_atrr_14" not in df.columns: 
        df.ta.atr(close=df["adj_fut_close"], high=df["adj_fut_high"], low=df["adj_fut_low"], append=True)
        df = df.rename(columns={"ATRr_14": "f_fut_atrr_14"})

    # --- GARCH VOLATILITY ---
    garch = GARCHModel()

    df = garch.fit_predict(df)

    # fallback if NaN
    df["f_fut_garch_vol"] = df["f_fut_garch_vol"].fillna(df["f_fut_volatility_10"])

    # volatility change
    df["f_fut_garch_vol_change"] = df["f_fut_garch_vol"] - df["f_fut_volatility_10"]

    # volatility ratio
    df["f_fut_vol_ratio"] = df["f_fut_garch_vol"] / (df["f_fut_volatility_10"] + 1e-9)

    # -- VIX features
    # vix = vix_close
    if 'f_vix_ret' not in df.columns:
        df['f_vix_ret'] = np.log(df['vix_close'] / df['vix_close'].shift(1))

    # interaction features
    df['f_fut_vix_divergence'] = df['f_vix_ret'] + df['f_fut_return_1']
    df['f_fut_price_vix_signal'] = df['f_fut_return_1'] * df['f_vix_ret']
    df["f_ret_divergence"] = df['f_fut_return_1'] - df['f_return_1']


    #also include sport features?

    # --- CLEAN ---
    #df = df.replace([np.inf, -np.inf], np.nan)
    #df = df.dropna().reset_index(drop=False)

    print(f"Index is : {df.index} number of records: {len(df)}")

    return df

def fut_add_select_features(dfi, config=None):
    df = dfi[['fut_expiry', 'open','high','close','low', 'volume', 
              'fut_open','fut_high','fut_low','fut_close','fut_prevclose', 'fut_volume',
              'fut_oi','fut_chgoi','lot','spot',
             'adj_fut_open','adj_fut_high','adj_fut_low','adj_fut_close','adj_fut_prevclose',
              'vix_open','vix_high','vix_low','vix_close','vix_prevclose' ]].copy()
    safe_add_feature("date", df, dfi)
    #df = dfi.copy()
    #df = pd.DataFrame()
    if 'date' not in dfi.index and 'date' in dfi.columns and 'fut_expiry' in dfi.columns:
        df['date'] = pd.to_datetime(dfi['date'])
        df['fut_expiry'] = pd.to_datetime(dfi['fut_expiry'])
        df = df.set_index('date')
    #print(f"Index is : {df.index} number of records: {len(df)}")
    use = config or {}
    #print(f"Index is : {df.index}")

    # RETURNS
    if use.get("fut_returns", False):
        #df['f_return_1'] = dfi['f_return_1']
        df['f_return_1'] = dfi['f_fut_return_1']
        df["f_fut_return_1"] = dfi["f_fut_return_1"]
        df["f_fut_return_5"] = dfi["f_fut_return_5"]
        df["f_fut_return_10"] = dfi["f_fut_return_10"]

    # MOMENTUM
    if use.get("fut_momentum", False):
        df["f_fut_momentum_5"] = dfi["f_fut_momentum_5"]
        safe_add_feature("f_fut_zscore", df, dfi)
        safe_add_feature("f_fut_rsi", df, dfi)

    if use.get("fut_momentum2", False):
        df["f_price_ma_ratio"] = dfi["f_price_ma_ratio"]

    # VOLATILITY
    if use.get("fut_volatility", False):
        df["f_fut_volatility_10"] = dfi["f_fut_volatility_10"]

        safe_add_feature("f_fut_rangeocp", df, dfi)
        safe_add_feature("f_fut_rangehlp", df, dfi)
        safe_add_feature("f_atrr_14", df, dfi)

    # TREND
    if use.get("fut_trend", False):
        df["f_fut_ma_10"] = dfi["f_fut_ma_10"]
        df["f_fut_ma_20"] = dfi["f_fut_ma_20"]
        df["f_fut_ma_ratio"] = dfi["f_fut_ma_ratio"] 

    # MEAN REVERSION
    if use.get("fut_mean_reversion", False):
        safe_add_feature("f_fut_zscore", df, dfi)

    # Range
    if use.get("fut_day_range_oc", False):
        safe_add_feature("f_fut_rangeocp", df, dfi)

    if use.get("fut_day_range_hl", False):
        safe_add_feature("f_fut_rangehlp", df, dfi)

    # VWAP
    if use.get("vwap", False):
        safe_add_feature("f_fut_vwap_d", df, dfi)

    # ATR
    if use.get("atr", False):
        safe_add_feature("f_fut_atrr_14", df, dfi)


    # --- GARCH VOLATILITY ---

    if use.get("garch_1", False):
        # fallback if NaN
        df["f_fut_garch_vol"] = dfi["f_fut_garch_vol"]


    if use.get("garch_2", False):

        # fallback if NaN
        df["f_fut_garch_vol"] = dfi["f_fut_garch_vol"]

        # volatility change
        df["f_fut_garch_vol_change"] = dfi["f_fut_garch_vol_change"]

    if use.get("garch_3", False):

        # fallback if NaN
        df["f_fut_garch_vol"] = dfi["f_fut_garch_vol"]

        # volatility change
        df["f_fut_garch_vol_change"] = dfi["f_fut_garch_vol_change"]

        # volatility ratio
        df["f_fut_vol_ratio"] = dfi["f_fut_vol_ratio"]        

    # -- VIX features
    if use.get("vix", False) and 'f_vix_ret' not in df.columns:
        df['f_vix_ret'] = dfi['f_vix_ret']

    # -- VIX and vol features
    if use.get("vixvol", False) and 'f_vix_ret' not in df.columns:
        df['f_vix_ret'] = dfi['f_vix_ret']
        df["f_fut_garch_vol"] = dfi["f_fut_garch_vol"]

    if use.get("oi", False):
        #df['oi_change'] = df['fut_oi'].pct_change()
        df['f_oi_change'] = dfi['f_oi_change']

    # --- OI ---
    if use.get("oi2", False):
        #df['oi_change'] = df['fut_oi'].pct_change()
        df['f_oi_change'] = dfi['f_oi_change']
        df['f_price_oi_signal'] = dfi['f_price_oi_signal']
        df['f_trend_strength'] = dfi['f_trend_strength']
        df['f_oi_vol'] = dfi['f_oi_vol']
        df['f_oi_momentum'] = dfi['f_oi_momentum']

    if use.get("voloi", False):
        #df['oi_change'] = df['fut_oi'].pct_change()
        df['f_oi_change'] = dfi['f_oi_change']
        df["f_fut_garch_vol"] = dfi["f_fut_garch_vol"]

    # --- basis ---
    if use.get("basis", False):

        df["f_basis"] = dfi["f_basis"]

        df['f_basis_pct'] = dfi['f_basis_pct']
        df['f_basis_smooth'] = dfi['f_basis_smooth']

        df['f_basis_zscore'] = dfi['f_basis_zscore']

    # --- structure
    if use.get("structure", False):

        df["f_basis"] = dfi["f_basis"]

    # --- POSITIONING     ----

    if use.get("interaction", False):
        safe_add_feature('f_fut_price_vix_signal', df, dfi)

    if use.get("interaction2", False):
        safe_add_feature("f_ret_divergence", df, dfi)
        safe_add_feature('f_fut_price_vix_signal', df, dfi)

    if use.get("interaction3", False):
        df['f_price_oi_signal'] = dfi['f_price_oi_signal']
        safe_add_feature('f_fut_price_vix_signal', df, dfi)

    
    #print(f"Index is : {df.index} number of records: {len(df)}")
    #print(df.info())
    return df

def get_features(config):

    features = []

    if config.get("returns"):
        features += ["ret_1", "ret_5"]

    if config.get("momentum"):
        features += ["zscore", "price_ma_ratio"]

    if config.get("volatility"):
        features += ["garch_vol", "garch_vol_change", "vol_ratio"]

    if config.get("vix"):
        features += ["vix", "vix_ret"]

    if config.get("oi"):
        features += ["oi_change"]

    if config.get("structure"):
        features += ["basis_pct", "dte_norm"]

    # ADD THIS BLOCK
    if config.get("interaction_features"):
        features += [
            "price_vix_signal",
            "price_oi_signal"
        ]

    return features

def add_basic_features(df):
    df = df.copy()

    # rolling volatility
    df["volatility"] = df["returns"].rolling(10).std()

    # momentum
    df["momentum"] = df["returns"].rolling(5).mean()

    df = df.dropna().reset_index(drop=False)
    print(f"Index is : {df.index} number of records: {len(df)}")

    return df