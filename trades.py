import numpy as np

def extract_trades(df):
    trades = []
    position = 0
    entry_price = None

    for i in range(len(df)):
        pos = df["position"].iloc[i]
        price = float(df["close"].iloc[i])

        if position == 0 and pos != 0:
            position = pos
            entry_price = price

        elif position != 0 and pos == 0:
            pnl = position * (price - entry_price)
            trades.append(pnl)
            position = 0

        elif position != 0 and pos != position:
            pnl = position * (price - entry_price)
            trades.append(pnl)
            position = pos
            entry_price = price

    return np.array(trades)