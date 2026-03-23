import pandas as pd
import json
import os

def log_results(run_id, params, metrics, score, file_path="results/summary.csv"):
    row = {
        "run_id": run_id,
        "params": json.dumps(params),
        "metrics": json.dumps(metrics),
        "score": score
    }

    df = pd.DataFrame([row])

    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode="a", header=False, index=False)