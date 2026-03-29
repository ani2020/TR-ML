import pandas as pd
from grid_search import GridSearchOptimizer
from walk_forward import WalkForward
from signal_generator import hmm_xgb_pipeline
from datetime import datetime

df = pd.read_csv("data/processed/sample_data.csv")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")

wf = WalkForward(train_size=500, test_size=100, step_size=100)

param_grid = {
    "n_components": [2],
    "long_threshold": [0.68], #[0.64, 0.66, 0.68]
    "short_threshold": [0.36], #[0.38, 0.36, 0.34]
    "xgb_params": [{"max_depth": 5, "learning_rate": 0.05}],
    "feature_config": [
        {"returns": True, "momentum": True},
        #{"returns": True, "momentum": True, "vwap": True},
        #{"returns": True, "momentum": True, "atr": True},
        #{"returns": True, "trend": True, "mean_reversion": True}
        #{"returns": True, "trend": True},
        #{"returns": True, "trend": True, "volatility": True},
    ]
}

optimizer = GridSearchOptimizer(param_grid)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

optimizer.run(df, wf, hmm_xgb_pipeline, f"results/summary_{timestamp}.csv")