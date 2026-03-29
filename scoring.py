import numpy as np

def compute_score(metrics):

    score = (
        metrics["sharpe"] * 0.3 +
        metrics["sortino"] * 0.2 +
        metrics["calmar"] * 0.3 +
        metrics["profit_factor"] * 0.1 +
        metrics["expectancy"] * 0.1
    )

    # penalize low trades
    if metrics["num_trades"] < 10:
        score *= 0.5
    
    if np.isinf(score):
        score = 0

    return score