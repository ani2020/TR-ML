from hmm_model import HMMModel
from validation import validate_dataframe
from xgboost_model import XGBoostModel
import numpy as np

def hmm_xgb_pipeline(train_df, test_df, params):

    # --- Clean ---
    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)

    if len(train_df) < 100 or len(test_df) < 20:
        test_df["signal"] = 0
        return test_df

    # --- HMM ---
    hmm = HMMModel(
        n_components=params.get("n_components", 3),
        covariance_type=params.get("covariance_type", "full")
    )

    hmm.fit(train_df)

    train_df = hmm.predict(train_df)
    test_df = hmm.predict(test_df)

    mapping = hmm.derive_state_mapping(train_df)

    train_df = hmm.apply_state_mapping(train_df, mapping)
    test_df = hmm.apply_state_mapping(test_df, mapping)

    # --- XGBoost ---
    xgb_model = XGBoostModel(params.get("xgb_params", None))

    xgb_model.fit(train_df)
    test_df = xgb_model.predict(test_df)

    # --- Final Signal Logic ---
    test_df["signal"] = 0

    # Only trade in certain regimes
    test_df.loc[
        (test_df["regime"] == "bull") &
        (test_df["xgb_prob"] > 0.55),
        "signal"
    ] = 1

    test_df.loc[
        (test_df["regime"] == "bear") &
        (test_df["xgb_prob"] < 0.45),
        "signal"
    ] = -1

    validate_dataframe(train_df, "train data")

    validate_dataframe(test_df, "test data")

    assert train_df["date"].max() < test_df["date"].min(), \
    "Lookahead bias detected!"

    print(test_df["xgb_prob"].describe())

    return test_df

def random_signal_pipeline(train_df, test_df, params):
    test_df = test_df.copy()

    np.random.seed(42)  # reproducible

    test_df["signal"] = np.random.choice(
        [-1, 0, 1],
        size=len(test_df)
    )

    return test_df

def perfect_foresight_pipeline(train_df, test_df, params):
    test_df = test_df.copy()

    # Use FUTURE return (this is intentional cheating)
    future_return = test_df["returns"].shift(-1)

    test_df["signal"] = 0
    test_df.loc[future_return > 0, "signal"] = 1
    test_df.loc[future_return < 0, "signal"] = -1

    # drop last row (NaN target)
    test_df = test_df.dropna().reset_index(drop=True)

    return test_df

# def simple_pipeline(train_df, test_df, params):
#     # Example: momentum strategy
#     test_df["signal"] = (test_df["returns"].rolling(5).mean() > 0).astype(int)

#     return test_df

