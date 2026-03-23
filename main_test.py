from walk_forward import WalkForward
#from signal_generator import hmm_pipeline
from signal_generator import hmm_xgb_pipeline
from signal_generator import random_signal_pipeline
from signal_generator import perfect_foresight_pipeline
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
    pipeline_fn=random_signal_pipeline,
    params={}
)

metrics = result["metrics"]

print("Random Test Metrics:", metrics)

result = wf.run(
    df=df,
    pipeline_fn=perfect_foresight_pipeline,
    params={}
)

metrics = result["metrics"]

print("Perfect Foresight Metrics:", metrics)