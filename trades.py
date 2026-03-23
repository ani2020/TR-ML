import numpy as np

import numpy as np

def extract_trades(df):
    trades = []
    position = 0
    entry_price = 0.0

    for i in range(len(df)):
        current_price = float(df["close"].iloc[i])  # force float

        if position == 0 and df["position"].iloc[i] != 0:
            position = df["position"].iloc[i]
            entry_price = current_price

        elif position != 0 and df["position"].iloc[i] == 0:
            exit_price = current_price

            pnl = position * (exit_price - entry_price)
            trades.append(pnl)

            position = 0

    return np.array(trades)