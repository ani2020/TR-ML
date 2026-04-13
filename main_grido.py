import pandas as pd
from grid_search import GridSearchOptimizer
from walk_forward import WalkForward
from signal_generator import hmm_xgb_pipeline
from datetime import datetime

df = pd.read_csv("data/processed/NIFTY_full_data_prec_f.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

wf = WalkForward(train_size=500, test_size=100, step_size=100)

# param_grid = {
#     "n_components": [2],
#     "long_threshold": [0.68], #[0.64, 0.66, 0.68]
#     "short_threshold": [0.36], #[0.38, 0.36, 0.34]
#     "xgb_params": [{"max_depth": 5, "learning_rate": 0.05}],
#     "feature_config": [
#         {"returns": True, "momentum": True},
#         #{"returns": True, "momentum": True, "vwap": True},
#         #{"returns": True, "momentum": True, "atr": True},
#         #{"returns": True, "trend": True, "mean_reversion": True}
#         #{"returns": True, "trend": True},
#         #{"returns": True, "trend": True, "volatility": True},
#     ]
# }

param_grid = {

    "n_components": [2, 3],

    "long_threshold": [0.64, 0.66, 0.68],
    "short_threshold": [0.34, 0.36, 0.38],

    "xgb_params": [
        {"max_depth": 5, "learning_rate": 0.05}
    ],

    "feature_config": [

        # ---------------- BASELINE ----------------
        {"fut_returns": True},

        # ---------------- MOMENTUM ----------------
        {"fut_returns": True, "fut_momentum": True},
        #{"fut_returns": True, "fut_momentum2": ["price_ma_ratio"]},

        # ---------------- VOLATILITY (CORE) ----------------
        {"fut_returns": True, "garch_1": True},
        {"fut_returns": True, "garch_2": True},

        # ---------------- VOLATILITY (RANGE ALT) ----------------
        #{"fut_returns": True, "garch_3": True},
        #{"fut_returns": True, "fut_volatility": True},

        # ---------------- VIX ----------------
        #{"fut_returns": True, "vix": True},
        #{"fut_returns": True, "vixvol": True},

        # ---------------- STRUCTURE ----------------
        {"fut_returns": True, "structure": True},
        #{"fut_returns": True, "structure": ["basis_pct", "dte_norm"]},

        # ---------------- POSITIONING ----------------
        #{"fut_returns": True, "oi": True},
        #{"fut_returns": True, "voloi": True},

        # ---------------- INTERACTION ONLY ----------------
        {"fut_returns": True, "interaction": True},
        {"fut_returns": True, "interaction3": True}
    ]
}


optimizer = GridSearchOptimizer(param_grid)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

optimizer.run(df, wf, hmm_xgb_pipeline, f"results/summary_{timestamp}.csv")