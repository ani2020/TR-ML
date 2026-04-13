import yfinance as yf
import pandas as pd

def load_data(symbol="^NSEI", start="2015-01-01", end=None):
    #df = yf.download(symbol, start=start, end=end)
    df = pd.read_csv("data/processed/NIFTY_full_data.csv")

    #print(df.info())
    #df = df.reset_index()
    #df.columns = ['timestamp', 'adj_fut_close', 'adj_fut_high', 'adj_fut_low', 'adj_fut_open', 'fut_volume', "fut_oi", "fut_chgoi"]
    df = df.rename(columns={
        "timestamp": "date"
    })
    print(f"loader data info: {df.info()}")
    # --- FORCE NUMERIC --- not necessary
    df["adj_fut_close"] = df["adj_fut_close"].apply(pd.to_numeric, errors="coerce")
    df["adj_fut_open"] = df["adj_fut_open"].apply(pd.to_numeric, errors="coerce")
    df["adj_fut_high"] = df["adj_fut_high"].apply(pd.to_numeric, errors="coerce")
    df["adj_fut_low"] = df["adj_fut_low"].apply(pd.to_numeric, errors="coerce")
    df["fut_volume"] = df["fut_volume"].apply(pd.to_numeric, errors="coerce")

    #return df[["date", "adj_fut_open", "adj_fut_high", "adj_fut_low", "adj_fut_close", "fut_volume", "fut_oi", "fut_chgoi"]]
    return df