from datetime import datetime

from walk_forward import WalkForward
#from signal_generator import hmm_pipeline
from signal_generator import hmm_xgb_pipeline
#from signal_generator import random_signal_pipeline
from scoring import compute_score
from plot_signals import plot_signals
#from candlestick_plot import plot_candlestick_with_signals
from candlestick_plot_plotly import plot_candlestick_with_signals
from validation import validate_dataframe

import pandas as pd

df = pd.read_csv("data/processed/NIFTY_full_data_prec_f.csv")
df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

#df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["close"] = df["close"].apply(pd.to_numeric, errors="coerce")
df["open"] = df["open"].apply(pd.to_numeric, errors="coerce")
df["high"] = df["high"].apply(pd.to_numeric, errors="coerce")
df["low"] = df["low"].apply(pd.to_numeric, errors="coerce")

#df["returns"] = pd.to_numeric(df["returns"], errors="coerce")
# df["volatility"] = pd.to_numeric(df["volatility"], errors="coerce")

df["f_return_1"] = pd.to_numeric(df["f_return_1"], errors="coerce")
df["f_return_5"] = pd.to_numeric(df["f_return_5"], errors="coerce")
df["f_return_10"] = pd.to_numeric(df["f_return_10"], errors="coerce")

df["f_momentum_5"] = pd.to_numeric(df["f_momentum_5"], errors="coerce")
df["f_momentum_10"] = pd.to_numeric(df["f_momentum_10"], errors="coerce")

df["f_ma_10"] = pd.to_numeric(df["f_ma_10"], errors="coerce")
df["f_ma_20"] = pd.to_numeric(df["f_ma_20"], errors="coerce")
df["f_ma_ratio"] = pd.to_numeric(df["f_ma_ratio"], errors="coerce")
df["f_price_ma_ratio"] = pd.to_numeric(df["f_price_ma_ratio"], errors="coerce")
df["f_zscore"] = pd.to_numeric(df["f_zscore"], errors="coerce")
df["f_rsi"] = pd.to_numeric(df["f_rsi"], errors="coerce")

df["f_volatility_5"] = pd.to_numeric(df["f_volatility_5"], errors="coerce")
df["f_volatility_10"] = pd.to_numeric(df["f_volatility_10"], errors="coerce")

validate_dataframe(df, "input data")

wf = WalkForward(
    train_size=500,
    test_size=100,
    step_size=100
)

params = {
    "n_components": 2,
    "long_threshold": 0.64,
    "short_threshold": 0.36,
    "xgb_params": {
        "max_depth": 5, 
        "learning_rate": 0.05,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        },
    "feature_config":{"fut_returns": True, "fut_momentum": True},
    "covariance_type": "diag"
}

result = wf.run(
    df=df.tail(1000),
    pipeline_fn=hmm_xgb_pipeline, #hmm_pipeline
    params=params
)


metrics = None
full_data = None
i = 0
for r in result:
    if i == 0:
        row = {
            **r["metrics"],
            **r["feature_importance"],
            "score": r["score"]
        }
        metrics = pd.DataFrame([row])
    else:
        row = {
            **r["metrics"],
            **r["feature_importance"],
            "score": r["score"]
        }
        dr = pd.DataFrame([row])
        metrics = pd.concat([metrics, dr], ignore_index=True)
    i+=1

#metrics = result[0]["metrics"]
#score = compute_score(metrics)

full_data = result[len(result)-1]["full_data"]

#print("Metrics:", metrics)
#print("Score:", score)

time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
full_data_file = f"results/full_data_{time_stamp}.csv"
metrics_file = f"results/metrics_data_{time_stamp}.csv"

full_data.to_csv(full_data_file, index=False)
metrics.to_csv(metrics_file, index=False)
print("Metrics:", metrics)

# df_plot = result["full_data"].copy()
# time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# results_df_file = f"results/full_data_df_{time_stamp}.csv"
# df_plot.to_csv(results_df_file)

# print(f"signal sanity check (only -1, 0, 1): {df_plot["signal"].value_counts()}")
# print(f"signal transition check: {print(df_plot["signal"].diff().value_counts())}")
# print(df_plot[["return_1", "position", "net_return"]].head(20))

# df_plot["date"] = pd.to_datetime(df_plot["date"], format="%Y-%m-%d")

# df_plot = df_plot.sort_values("date").reset_index(drop=True)
# df_plot = df_plot.drop_duplicates(subset=["date"], keep="last")
# df_plot = df_plot.reset_index(drop=True)
# #df_plot = df_plot.tail(500)
# #plot_signals(df_plot)
# plot_candlestick_with_signals(df_plot)