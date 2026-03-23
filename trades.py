import numpy as np

def extract_trades(df):
    trades = []
    position = 0
    entry_price = None

    for i in range(len(df)):
        current_pos = df["position"].iloc[i]
        price = float(df["close"].iloc[i])

        # ENTRY
        if position == 0 and current_pos != 0:
            position = current_pos
            entry_price = price

        # EXIT
        elif position != 0 and current_pos == 0:
            pnl = position * (price - entry_price)
            trades.append(pnl)
            position = 0
            entry_price = None

        # FLIP (IMPORTANT FIX)
        elif position != 0 and current_pos != position:
            pnl = position * (price - entry_price)
            trades.append(pnl)

            # new position
            position = current_pos
            entry_price = price

    return np.array(trades)