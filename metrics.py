import numpy as np

import numpy as np

def compute_metrics(df, trades):
    returns = df["net_return"].dropna()

    # --- Sharpe ---
    if returns.std() == 0 or len(returns) < 2:
        sharpe = 0
    else:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()

    # --- Sortino ---
    downside = returns[returns < 0]
    if downside.std() == 0 or len(downside) < 2:
        sortino = 0
    else:
        sortino = np.sqrt(252) * returns.mean() / downside.std()

    # --- Drawdown ---
    equity = df["equity"].dropna()
    if len(equity) == 0:
        max_dd = 0
    else:
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

    # --- CAGR ---
    if len(equity) < 2:
        cagr = 0
    else:
        years = len(equity) / 252
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1

    # --- Calmar ---
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # --- Profit Factor ---
    if len(trades) == 0:
        profit_factor = 0
    else:
        profits = trades[trades > 0].sum()
        losses = abs(trades[trades < 0].sum())
        profit_factor = profits / losses if losses != 0 else 0

    # --- Expectancy ---
    if len(trades) == 0:
        expectancy = 0
        win_rate = 0
    else:
        wins = trades[trades > 0]
        losses_arr = trades[trades < 0]

        win_rate = len(wins) / len(trades)

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses_arr.mean() if len(losses_arr) > 0 else 0

        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    print("Returns std:", returns.std())
    print("Num trades:", len(trades))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "cagr": cagr,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "win_rate": win_rate,
        "num_trades": len(trades)
    }