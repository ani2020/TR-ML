import yfinance as yf
import pandas as pd

def load_data(symbol="^NSEI", start="2015-01-01", end=None):
    df = yf.download(symbol, start=start, end=end)
    df = df.reset_index()
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close"
    })
    print(f"loader data info: {df.info()}")
    # --- FORCE NUMERIC --- not necessary
    df["close"] = df["close"].apply(pd.to_numeric, errors="coerce")
    df["open"] = df["open"].apply(pd.to_numeric, errors="coerce")
    df["high"] = df["high"].apply(pd.to_numeric, errors="coerce")
    df["low"] = df["low"].apply(pd.to_numeric, errors="coerce")

    return df[["date", "open", "high", "low", "close"]]