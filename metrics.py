import numpy as np

def compute_brier_score(df):
    df = df.copy()

    df["actual"] = (df["f_return_1"].shift(-1) > 0).astype(int)

    df = df.dropna()

    brier = ((df["xgb_prob"] - df["actual"]) ** 2).mean()

    return brier

def compute_metrics(df, trades):

    returns = df["f_return_1"] * df["position"]

    # --- Sharpe ---
    sharpe = np.sqrt(252) * returns.mean() / (returns.std() + 1e-9)

    # --- Sortino ---
    downside = returns[returns < 0]
    sortino = np.sqrt(252) * returns.mean() / (downside.std() + 1e-9)

    # --- Equity Curve ---
    equity = (1 + returns).cumprod()

    # --- Drawdown ---
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_drawdown = drawdown.min()

    # --- CAGR ---
    total_return = equity.iloc[-1]
    n_years = len(df) / 252
    cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

    # --- Calmar ---
    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0

    if abs(max_drawdown) < 1e-4:
        calmar = 0
    

    # --- Profit Factor ---
    profits = trades[trades > 0].sum()
    losses = -trades[trades < 0].sum()

    if losses == 0:
        profit_factor = np.inf if profits > 0 else 0
    else:
        profit_factor = profits / losses

    # --- Win Rate ---
    win_rate = (trades > 0).mean() if len(trades) > 0 else 0

    # --- Expectancy ---
    expectancy = trades.mean() if len(trades) > 0 else 0

    brier = compute_brier_score(df)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "num_trades": len(trades),
        "brier_score": brier
    }