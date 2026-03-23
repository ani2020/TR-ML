from walk_forward import WalkForward
#from signal_generator import hmm_pipeline
from signal_generator import hmm_xgb_pipeline
from signal_generator import random_signal_pipeline
from scoring import compute_score
from plot_signals import plot_signals
#from candlestick_plot import plot_candlestick_with_signals
from candlestick_plot_plotly import plot_candlestick_with_signals
from validation import validate_dataframe

import pandas as pd

df = pd.read_csv("data/processed/sample_data.csv")

#df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["close"] = df["close"].apply(pd.to_numeric, errors="coerce")
df["open"] = df["open"].apply(pd.to_numeric, errors="coerce")
df["high"] = df["high"].apply(pd.to_numeric, errors="coerce")
df["low"] = df["low"].apply(pd.to_numeric, errors="coerce")

df["returns"] = pd.to_numeric(df["returns"], errors="coerce")
df["volatility"] = pd.to_numeric(df["volatility"], errors="coerce")

# df["return_1"] = pd.to_numeric(df["return_1"], errors="coerce")
# df["return_3"] = pd.to_numeric(df["return_3"], errors="coerce")
# df["return_5"] = pd.to_numeric(df["return_5"], errors="coerce")

# df["momentum_5"] = pd.to_numeric(df["momentum_5"], errors="coerce")
# df["momentum_10"] = pd.to_numeric(df["momentum_10"], errors="coerce")

# df["ma_10"] = pd.to_numeric(df["ma_10"], errors="coerce")
# df["ma_20"] = pd.to_numeric(df["ma_20"], errors="coerce")
# df["ma_ratio"] = pd.to_numeric(df["ma_ratio"], errors="coerce")
# df["price_ma_ratio"] = pd.to_numeric(df["price_ma_ratio"], errors="coerce")
# df["zscore"] = pd.to_numeric(df["zscore"], errors="coerce")
# df["rsi"] = pd.to_numeric(df["rsi"], errors="coerce")

# df["volatility_5"] = pd.to_numeric(df["volatility_5"], errors="coerce")
# df["volatility_10"] = pd.to_numeric(df["volatility_10"], errors="coerce")

validate_dataframe(df, "input data")

wf = WalkForward(
    train_size=500,
    test_size=100,
    step_size=100
)

params = {
    "n_components": 3,
    "covariance_type": "full",
    "xgb_params": {
        "max_depth": 5,
        "learning_rate": 0.1
    }
}

result = wf.run(
    df=df,
    pipeline_fn=hmm_xgb_pipeline, #hmm_pipeline
    params=params
)

metrics = result["metrics"]
score = compute_score(metrics)

print("Metrics:", metrics)
print("Score:", score)

df_plot = result["full_data"].copy()

print(f"signal sanity check (only -1, 0, 1): {df_plot["signal"].value_counts()}")
print(f"signal transition check: {print(df_plot["signal"].diff().value_counts())}")
print(df_plot[["returns", "position", "net_return"]].head(20))

df_plot["date"] = pd.to_datetime(df_plot["date"], format="%Y-%m-%d")

df_plot = df_plot.sort_values("date").reset_index(drop=True)
df_plot = df_plot.drop_duplicates(subset=["date"], keep="last")
df_plot = df_plot.reset_index(drop=True)
#df_plot = df_plot.tail(500)
#plot_signals(df_plot)
plot_candlestick_with_signals(df_plot)