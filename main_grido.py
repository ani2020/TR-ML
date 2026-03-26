from datetime import datetime

import pandas as pd
from grid_search import GridSearchOptimizer
from signal_generator import go_hmm_xgb_pipeline
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
    "long_threshold": [0.55, 0.6, 0.65],
    "short_threshold": [0.45, 0.4, 0.35],
    "xgb_params": [
        {"max_depth": 3, "learning_rate": 0.1},
        {"max_depth": 5, "learning_rate": 0.05}
    ],
    "feature_config": [
        {"returns": True, "momentum": True, "volatility": True},
        {"returns": True, "trend": True, "mean_reversion": True},
        {"returns": True, "momentum": True, "trend": True}
    ]
}

# Run optimization
optimizer = GridSearchOptimizer(param_grid)

results_df = optimizer.run(
    df=df,
    walkforward=wf,
    pipeline_fn=go_hmm_xgb_pipeline
)

# View best results
results_df = results_df.sort_values("score", ascending=False)
feature_impact = results_df.groupby("feature_config")["score"].mean()
threshold_impact = results_df.groupby("long_threshold")["score"].mean()

print(f"feature_impact: {feature_impact}" )
print(f"threshold_impact: {threshold_impact}" )

time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_df_file = f"results/results_df_{time_stamp}.csv"
results_df.to_csv(results_df_file)

print(results_df.head())