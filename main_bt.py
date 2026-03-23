from backtester import Backtester
from trades import extract_trades
from metrics import compute_metrics
from scoring import compute_score
from logger import log_results

import pandas as pd

# Load your prepared dataframe
df = pd.read_csv("data/processed/sample_data.csv")

bt = Backtester()

df_result = bt.run(df)

trades = extract_trades(df_result)

metrics = compute_metrics(df_result, trades)

score = compute_score(metrics)

log_results(
    run_id="run_001",
    params={"example_param": 1},
    metrics=metrics,
    score=score
)

print(metrics)
print("Score:", score)