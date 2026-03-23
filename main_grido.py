from datetime import datetime

import pandas as pd
from grid_search import GridSearchOptimizer
from signal_generator import hmm_xgb_pipeline
from walk_forward import WalkForward

# Load data
df = pd.read_csv("data/processed/sample_data.csv")

# Walk-forward setup
wf = WalkForward(
    train_size=500,
    test_size=100,
    step_size=100
)

# Grid
param_grid = {
    "n_components": [2, 3],
    "covariance_type": ["diag"],
    "xgb_params": [
        {"max_depth": 3, "learning_rate": 0.1},
        {"max_depth": 5, "learning_rate": 0.05}
    ]
}

# Run optimization
optimizer = GridSearchOptimizer(param_grid)

results_df = optimizer.run(
    df=df,
    walkforward=wf,
    pipeline_fn=hmm_xgb_pipeline
)

# View best results
results_df = results_df.sort_values("score", ascending=False)
time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_df_file = f"results/results_df_{time_stamp}.csv"
results_df.to_csv(results_df_file)

print(results_df.head())