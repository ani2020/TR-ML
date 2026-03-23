
def compute_score(metrics):
    return (
        0.25 * metrics["sharpe"] +
        0.25 * metrics["sortino"] +
        0.2 * metrics["calmar"] +
        0.2 * metrics["profit_factor"] +
        0.1 * metrics["expectancy"]
    )